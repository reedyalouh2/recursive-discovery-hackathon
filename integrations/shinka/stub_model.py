"""Deterministic offline stand-in for an OpenAI-compatible chat model.

Serves POST /v1/chat/completions on 127.0.0.1 only. Every reply is a
ShinkaEvolve SEARCH/REPLACE diff: double TRIALS in the current program, or
increment SEED once TRIALS reaches MAX_TRIALS. A run therefore exercises
prompting, patch parsing, evaluation and archiving without any paid model.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import threading

MODEL_NAME = "offline-stub-v1"
MAX_TRIALS = 64
MAX_REQUEST_BYTES = 8 * 1024 * 1024
_CURRENT_PROGRAM = re.compile(r"# Current program.*?```[\w+-]*\n(.*?)\n```", re.S)
_CODE_BLOCK = re.compile(r"```[\w+-]*\n(.*?)\n```", re.S)
_TRIALS = re.compile(r"^TRIALS = (\d+)$", re.M)
_SEED = re.compile(r"^SEED = (\d+)$", re.M)


def propose(prompt):
    """The stub's reply to one ShinkaEvolve mutation prompt."""
    match = _CURRENT_PROGRAM.search(prompt)
    blocks = _CODE_BLOCK.findall(prompt)
    code = match.group(1) if match else (blocks[-1] if blocks else "")
    trials = _TRIALS.search(code)
    seed = _SEED.search(code)
    if trials and int(trials.group(1)) < MAX_TRIALS:
        old, name = trials.group(0), "double_trials"
        new = f"TRIALS = {min(max(1, 2 * int(trials.group(1))), MAX_TRIALS)}"
    elif seed:
        old, name = seed.group(0), "next_seed"
        new = f"SEED = {int(seed.group(1)) + 1}"
    else:
        return ("<NAME>\nno_change\n</NAME>\n\n<DESCRIPTION>\n"
                "The stub found no TRIALS or SEED line to edit.\n</DESCRIPTION>\n\n<DIFF>\n</DIFF>\n")
    return (f"<NAME>\n{name}\n</NAME>\n\n"
            f"<DESCRIPTION>\nDeterministic stub edit: {old} -> {new}.\n</DESCRIPTION>\n\n"
            f"<DIFF>\n<<<<<<< SEARCH\n{old}\n=======\n{new}\n>>>>>>> REPLACE\n</DIFF>\n")


class _Handler(BaseHTTPRequestHandler):
    server_version = "offline-stub/1"

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self.send_error(404, "Only /v1/chat/completions is served.")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= MAX_REQUEST_BYTES:
                raise ValueError("request too large")
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            messages = body.get("messages", [])
        except (ValueError, AttributeError):
            self.send_error(400, "Expected a JSON chat-completions request.")
            return
        prompt = next((m.get("content") for m in reversed(messages)
                       if isinstance(m, dict) and m.get("role") == "user"), "")
        prompt = prompt if isinstance(prompt, str) else ""
        content = propose(prompt)
        number = self.server.count_request()
        usage = {"prompt_tokens": len(prompt.split()), "completion_tokens": len(content.split())}
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        reply = json.dumps({
            "id": f"stub-{number}",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", MODEL_NAME),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": usage,
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):
        pass


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.requests = 0
        self._lock = threading.Lock()

    def count_request(self):
        with self._lock:
            self.requests += 1
            return self.requests


class StubServer:
    """Context manager running the stub on an ephemeral localhost port."""

    def __init__(self):
        self._server = _Server()
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    @property
    def requests(self):
        return self._server.requests

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
