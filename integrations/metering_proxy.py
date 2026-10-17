"""Text-only chat proxy with conservative, persistent spending reservations.

Only this process holds the provider key. Each sent request reserves the full
configured prompt-context ceiling plus its clamped output-token limit. This
bound assumes the provider honors those token limits and the pinned prices.
Reported usage settles the reservation, including on HTTP errors. Ambiguous
post-send failures retain it; a connection failure before sending is free.

Missing usage, reported bound violations, or persistent upstream failures stop
forwarding. Locally refused requests return an empty completion so Shinka does
not keep retrying them. Provider billing remains the authoritative cost record.
"""

from decimal import Decimal
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import secrets
import ssl
import socket
import threading
import time
from urllib.parse import urlsplit

from search.spend import BudgetExceeded, LedgerError

MAX_REQUEST_BYTES = 8 * 2 ** 20
MAX_RESPONSE_BYTES = 16 * 2 ** 20
INPUT_MARGIN_TOKENS = 2048  # chat templates and provider-side system prompts
PER_MESSAGE_TOKENS = 64
MAX_TRANSPORT_FAILURES = 3
CLIENT_READ_TIMEOUT = 10
DEFAULT_MAX_INPUT_TOKENS = 1_048_576
MILLION = Decimal(1000000)


def token_cost(prices, prompt_tokens, completion_tokens, cached_tokens=0):
    cached = min(cached_tokens, prompt_tokens)
    return (Decimal(prompt_tokens - cached) * prices["input"] + Decimal(cached) * prices["cached_input"]
            + Decimal(completion_tokens) * prices["output"]) / MILLION


class MeteringProxy:
    def __init__(self, upstream, api_key, model, prices, ledger, max_output_tokens, log_path=None,
                 timeout=600, max_input_tokens=DEFAULT_MAX_INPUT_TOKENS):
        parts = urlsplit(upstream)
        if not (parts.scheme == "https" or (parts.scheme == "http" and parts.hostname == "127.0.0.1")):
            raise ValueError("The upstream must be https, or http on 127.0.0.1 for tests.")
        if not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("The upstream must have a hostname and no credentials, query, or fragment.")
        if not isinstance(api_key, str) or not api_key or any(not 33 <= ord(c) <= 126 for c in api_key):
            raise ValueError("A nonempty ASCII provider key without whitespace is required.")
        if any(type(value) is not int or value < 1 for value in (max_output_tokens, max_input_tokens)):
            raise ValueError("Token ceilings must be positive integers.")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("The transport timeout must be finite and positive.")
        self.upstream = parts
        self.model = model
        self.prices = prices
        self.ledger = ledger
        self.max_output_tokens = max_output_tokens
        self.max_input_tokens = max_input_tokens
        self.timeout = timeout
        self.token = secrets.token_urlsafe(32)
        self._api_key = api_key
        self._log_path = log_path
        self._lock = threading.Lock()
        self._failures = 0
        self.stopped = None  # why forwarding stopped, if it has
        self.stats = {"forwarded": 0, "successful": 0, "refused": 0, "upstream_errors": 0, "transport_errors": 0,
                      "usage_missing": 0, "usage_overruns": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "cached_tokens": 0, "charged_usd": Decimal(0)}
        self._server = _Server(self)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10)

    def summary(self):
        with self._lock:
            stats = dict(self.stats, charged_usd=str(self.stats["charged_usd"]))
            return dict(stats, stopped=self.stopped, model=self.model,
                        max_output_tokens=self.max_output_tokens, max_input_tokens=self.max_input_tokens)

    def handle(self, headers, raw):
        """Return (status, body bytes) for one client request."""
        supplied = headers.get("Authorization", "")
        if not hmac.compare_digest(supplied.encode("utf-8"), f"Bearer {self.token}".encode("utf-8")):
            return 401, _error("Missing or invalid proxy token.")
        try:
            body = _request(raw, self.model, self.max_output_tokens)
            # This estimate rejects excessive text. It never reduces the full
            # context reservation, so tokenizer/template differences cannot
            # silently shrink the budget guard.
            estimate = (len(json.dumps(body["messages"], ensure_ascii=False).encode("utf-8"))
                        + PER_MESSAGE_TOKENS * len(body["messages"]) + INPUT_MARGIN_TOKENS)
            if estimate > self.max_input_tokens:
                raise ValueError("text exceeds the conservative input allowance")
        except (UnicodeError, ValueError, RecursionError) as error:
            return 400, _error(f"Rejected by the metering proxy: {error}")
        worst = token_cost(self.prices, self.max_input_tokens, body["max_tokens"])

        with self._lock:
            stopped = self.stopped
        if stopped:
            return self._refuse(stopped)
        try:
            reservation = self.ledger.reserve(worst, model=self.model)
        except BudgetExceeded as error:
            return self._refuse(f"budget: {error}", latch=True)
        except LedgerError as error:
            return self._refuse(f"ledger: {error}", latch=True)

        started = time.monotonic()
        status, payload, charge, outcome, usage = self._forward(body, worst)
        if usage and (usage[0] > self.max_input_tokens or usage[1] > body["max_tokens"] or charge > worst):
            outcome = "usage_overrun"
            status, payload = 502, _error("Provider usage exceeded the reserved token/cost bound; forwarding stopped.")
            with self._lock:
                self.stopped = "provider usage exceeded the reserved token/cost bound"
        try:
            self.ledger.settle(reservation, charge, outcome=outcome)
        except LedgerError as error:
            with self._lock:
                self.stopped = f"ledger: {error}"
        self._record(outcome, charge, usage, worst, status, time.monotonic() - started)
        return status, payload

    def _forward(self, body, worst):
        """Send one request; never return provider error details or its key."""
        parts = self.upstream
        cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        kwargs = {"context": ssl.create_default_context()} if parts.scheme == "https" else {}
        connection = cls(parts.hostname, parts.port, timeout=self.timeout, **kwargs)
        try:
            connection.connect()
        except (OSError, http.client.HTTPException):  # no request was sent
            connection.close()
            return 502, _error("Upstream connection failed."), Decimal(0), "connect_failed", None
        # A socket's ordinary timeout resets after each read. An absolute
        # deadline also interrupts a peer that keeps dribbling response bytes.
        transport_socket = connection.sock
        deadline = threading.Timer(self.timeout, _close_socket, args=(transport_socket,))
        deadline.daemon = True
        deadline.start()
        try:
            connection.request(
                "POST", parts.path.rstrip("/") + "/chat/completions",
                body=json.dumps(body).encode("utf-8"),
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json",
                         "Accept": "application/json"})
            response = connection.getresponse()
            payload = response.read(MAX_RESPONSE_BYTES + 1)
            status = response.status
        except (OSError, http.client.HTTPException):
            return 502, _error("Upstream transport failed."), worst, "transport_error", None
        finally:
            deadline.cancel()
            connection.close()
        if len(payload) > MAX_RESPONSE_BYTES:
            return 502, _error("Upstream response too large."), worst, "oversized_response", None
        usage = _usage(payload)
        charge = token_cost(self.prices, *usage) if usage is not None else worst
        if status != 200:
            return status, _error(f"Upstream returned HTTP {status}."), charge, "upstream_error", usage
        if usage is None:
            return 502, _error("Upstream usage missing or invalid; forwarding stopped."), worst, "usage_missing", None
        return status, _redact_payload(payload, self._api_key), charge, "ok", usage

    def _refuse(self, reason, latch=False):
        with self._lock:
            if latch and not self.stopped:
                self.stopped = reason
            self.stats["refused"] += 1
        return 200, json.dumps({
            "id": "metering-proxy-refusal", "object": "chat.completion", "created": 0, "model": self.model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": ""}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }).encode("utf-8")

    def _record(self, outcome, charge, usage, worst, status, seconds):
        with self._lock:
            stats = self.stats
            stats["charged_usd"] += charge
            if outcome != "connect_failed":
                stats["forwarded"] += 1  # requests attempted after connecting
            if outcome == "ok":
                stats["successful"] += 1
                self._failures = 0
            else:
                self._failures += 1
            if outcome == "usage_missing":
                stats["usage_missing"] += 1
                self.stopped = self.stopped or "provider usage missing or invalid"
            elif outcome == "usage_overrun":
                stats["usage_overruns"] += 1
            elif outcome == "upstream_error":
                stats["upstream_errors"] += 1
                if usage is None:
                    stats["usage_missing"] += 1
                if status in (400, 401, 403, 404):
                    self.stopped = self.stopped or f"upstream rejected the request (HTTP {status})"
            elif outcome in ("connect_failed", "transport_error", "oversized_response"):
                stats["transport_errors"] += 1
            if self._failures >= MAX_TRANSPORT_FAILURES and not self.stopped:
                self.stopped = f"upstream unavailable after {self._failures} consecutive failures"
            if usage:
                stats["prompt_tokens"] += usage[0]
                stats["completion_tokens"] += usage[1]
                stats["cached_tokens"] += usage[2]
            if self._log_path is not None:
                entry = {"time": time.time(), "outcome": outcome, "status": status,
                         "reserved_usd": str(worst), "charged_usd": str(charge),
                         "usage": list(usage) if usage else None, "seconds": round(seconds, 3)}
                with open(self._log_path, "a", encoding="utf-8") as log:
                    log.write(json.dumps(entry) + "\n")


def _request(raw, model, max_output_tokens):
    """Reject any unpriced modality or token-bearing field outside plain text."""
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("request too large")
    body = json.loads(raw.decode("utf-8"))
    allowed = {"model", "messages", "max_tokens", "max_completion_tokens", "temperature",
               "top_p", "frequency_penalty", "presence_penalty", "seed", "stop", "stream", "n"}
    if not isinstance(body, dict) or set(body) - allowed:
        raise ValueError("only text chat and supported sampling parameters are allowed")
    if body.get("model") != model:
        raise ValueError("the requested model does not match this proxy")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    for message in messages:
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or message.get("role") not in ("system", "developer", "user", "assistant")
                or not isinstance(message.get("content"), str)):
            raise ValueError("each message must contain only a supported role and text content")
    if "stream" in body and body["stream"] is not False:
        raise ValueError("streaming is not supported")
    if "n" in body and (type(body["n"]) is not int or body["n"] != 1):
        raise ValueError("exactly one completion is supported")
    for field, low, high in (("temperature", 0, 2), ("top_p", 0, 1),
                             ("frequency_penalty", -2, 2), ("presence_penalty", -2, 2)):
        if field in body:
            value = body[field]
            if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
                raise ValueError(f"{field} must be finite and between {low} and {high}")
    if "seed" in body and (type(body["seed"]) is not int or not 0 <= body["seed"] < 2 ** 63):
        raise ValueError("seed must be an unsigned 63-bit integer")
    if "stop" in body:
        stops = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]
        if (not 1 <= len(stops) <= 4
                or any(not isinstance(stop, str) or not 1 <= len(stop) <= 256 for stop in stops)):
            raise ValueError("stop must be one to four strings of at most 256 characters")
    requested = [body[key] for key in ("max_tokens", "max_completion_tokens") if key in body]
    if any(type(value) is not int or value < 1 for value in requested):
        raise ValueError("max_tokens must be a positive integer")
    body.pop("max_completion_tokens", None)
    body["max_tokens"] = min(requested + [max_output_tokens])
    return body


def _usage(payload):
    """Validated (prompt, completion, cached) counts, or None when ambiguous."""
    try:
        usage = json.loads(payload.decode("utf-8"))["usage"]
        prompt, completion = usage["prompt_tokens"], usage["completion_tokens"]
        details = usage.get("prompt_tokens_details")
        details = {} if details is None else details
        cached = details.get("cached_tokens", 0)
        if not all(type(value) is int and value >= 0 for value in (prompt, completion, cached)):
            return None
        if cached > prompt:
            return None
        if "total_tokens" in usage and (type(usage["total_tokens"]) is not int
                                         or usage["total_tokens"] != prompt + completion):
            return None
        return prompt, completion, cached
    except (UnicodeError, ValueError, KeyError, TypeError, AttributeError, RecursionError):
        return None


def _redact_payload(payload, secret):
    """Redact decoded JSON strings too, including Unicode-escaped secrets."""
    def scrub(value):
        if isinstance(value, str):
            return value.replace(secret, "[REDACTED]")
        if isinstance(value, list):
            return [scrub(item) for item in value]
        if isinstance(value, dict):
            return {scrub(key): scrub(item) for key, item in value.items()}
        return value
    return json.dumps(scrub(json.loads(payload.decode("utf-8")))).encode("utf-8")


def _close_socket(transport_socket):
    try:
        transport_socket.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


def _error(message):
    return json.dumps({"error": {"message": message, "type": "metering_proxy"}}).encode("utf-8")


class _Handler(BaseHTTPRequestHandler):
    server_version = "metering-proxy/1"

    def setup(self):
        super().setup()
        self.connection.settimeout(CLIENT_READ_TIMEOUT)

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._reply(404, _error("Only /v1/chat/completions is served."))
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_REQUEST_BYTES:
            self._reply(413, _error("Request too large."))
            return
        status, body = self.server.proxy.handle(self.headers, self.rfile.read(length))
        self._reply(status, body)

    def _reply(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            pass  # billing was already settled even if the client disconnected

    def log_message(self, *args):
        pass


class _Server(ThreadingHTTPServer):
    # server_close joins active handlers before the engine records final cost.
    daemon_threads = False
    block_on_close = True

    def __init__(self, proxy):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.proxy = proxy
