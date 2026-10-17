"""Loopback stand-in for Crusoe's chat-completions endpoint, for tests (no network)."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import threading
import time

from integrations.shinka.stub_model import propose


def reply_for(messages):
    """A ShinkaEvolve-compatible edit: a SEARCH/REPLACE diff, or full code when asked for <CODE>."""
    system = next((m["content"] for m in messages if m.get("role") == "system"), "")
    user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    diff = propose(user)
    if "<CODE>" not in system or "<<<<<<< SEARCH" not in diff:
        return diff
    code = re.search(r"# Current program.*?```[\w+-]*\n(.*?)\n```", user, re.S).group(1)
    search = diff.split("<<<<<<< SEARCH\n", 1)[1].split("\n=======\n", 1)[0]
    replace = diff.split("\n=======\n", 1)[1].split("\n>>>>>>> REPLACE", 1)[0]
    return ("<NAME>\nfull_rewrite\n</NAME>\n\n<DESCRIPTION>\nFake full rewrite.\n</DESCRIPTION>\n\n"
            f"<CODE>\n```python\n{code.replace(search, replace, 1)}\n```\n</CODE>\n")


class FakeCrusoe:
    """mode: "ok", "max_usage" (bill the full max_tokens), "no_usage", "error" or "slow"."""

    def __init__(self, mode="ok", status=500, delay=0.0):
        self.mode, self.status, self.delay = mode, status, delay
        self.requests = []
        self._lock = threading.Lock()
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

    def respond(self, headers, body):
        with self._lock:
            self.requests.append({"headers": dict(headers), "body": body})
        if self.mode == "slow":
            time.sleep(self.delay)
        if self.mode == "error":
            return self.status, {"error": {"message": "fake upstream failure"}}
        content = reply_for(body["messages"])
        prompt_tokens = len(json.dumps(body["messages"])) // 4
        completion_tokens = body["max_tokens"] if self.mode == "max_usage" else len(content) // 4
        reply = {"id": "fake", "object": "chat.completion", "created": 0, "model": body["model"],
                 "choices": [{"index": 0, "finish_reason": "stop",
                              "message": {"role": "assistant", "content": content}}],
                 "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                           "total_tokens": prompt_tokens + completion_tokens,
                           "prompt_tokens_details": {"cached_tokens": prompt_tokens // 2}}}
        if self.mode == "no_usage":
            del reply["usage"]
        return 200, reply


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        status, reply = self.server.fake.respond(self.headers, body)
        data = json.dumps(reply).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            pass  # the client gave up (timeout tests)

    def log_message(self, *args):
        pass


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, fake):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.fake = fake
