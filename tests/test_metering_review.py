"""Regression checks use only synthetic credentials and localhost providers."""
from decimal import Decimal
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import time
import unittest
import urllib.request

from integrations import crusoe
from integrations.metering_proxy import MeteringProxy, token_cost
from search.spend import SpendLedger
from search_tests.fake_crusoe import FakeCrusoe

MODEL = "nvidia/Nemotron-3-Super-120B-A12B"
KEY = "synthetic-review-key-only"
NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class FixedProvider(FakeCrusoe):
    def __init__(self, status=200, usage=None, content="ok", delay=0):
        self.reply_status = status
        self.reply_usage = usage if usage is not None else {"prompt_tokens": 100, "completion_tokens": 1}
        self.content = content
        self.entered = threading.Event()
        super().__init__(delay=delay)

    def respond(self, headers, body):
        self.requests.append({"headers": dict(headers), "body": body})
        self.entered.set()
        time.sleep(self.delay)
        reply = {"choices": [{"message": {"role": "assistant", "content": self.content}}],
                 "usage": self.reply_usage}
        if self.reply_status != 200:
            reply["error"] = {"message": "Provider echoed " + headers.get("Authorization", "")}
        return self.reply_status, reply


class MeteringReviewTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.tmp = Path(temp.name)
        self.ledger = SpendLedger("review", "1", path=self.tmp / "ledger.jsonl")

    def proxy(self, fake, **kwargs):
        return MeteringProxy(fake.base_url, KEY, MODEL, crusoe.prices(MODEL), self.ledger, 1024,
                             max_input_tokens=4096, **kwargs)

    def request(self, proxy, body=None):
        body = body if body is not None else {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
        return proxy.handle({"Authorization": "Bearer " + proxy.token}, json.dumps(body).encode())

    def test_rejects_unmetered_payloads_before_reserving_or_forwarding(self):
        variants = [
            {"tools": [{"type": "function", "function": {"description": "x" * 100000}}]},
            {"response_format": {"type": "json_schema", "json_schema": {"description": "x" * 100000}}},
            {"extra_body": {"max_tokens": 999999}},
            {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.invalid/a.png"}}]}]},
            {"messages": [{"role": "user", "content": "hi", "tool_calls": []}]},
            {"temperature": float("nan")}, {"top_p": 2}, {"seed": -1}, {"n": True},
        ]
        with FixedProvider() as fake, self.proxy(fake) as proxy:
            for extra in variants:
                body = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
                body.update(extra)
                with self.subTest(extra=list(extra)):
                    self.assertEqual(self.request(proxy, body)[0], 400)
        self.assertEqual(fake.requests, [])
        self.assertEqual(self.ledger.totals()["run_committed"], 0)

    def test_full_context_is_reserved_even_for_a_short_prompt(self):
        with FixedProvider() as fake, self.proxy(fake) as proxy:
            self.assertEqual(self.request(proxy)[0], 200)
        entries = [json.loads(line) for line in self.ledger.path.read_text().splitlines()]
        self.assertEqual(Decimal(entries[0]["usd"]), token_cost(crusoe.prices(MODEL), 4096, 1024))
        self.assertEqual(proxy.summary()["max_input_tokens"], 4096)
        self.assertEqual(proxy.summary()["successful"], 1)

    def test_usage_overrun_is_recorded_and_stops_further_requests(self):
        with FixedProvider(usage={"prompt_tokens": 100000, "completion_tokens": 1}) as fake, self.proxy(fake) as proxy:
            self.assertEqual(self.request(proxy)[0], 502)
            self.assertEqual(self.request(proxy)[0], 200)  # local empty refusal
        self.assertEqual(len(fake.requests), 1)
        self.assertEqual(self.ledger.totals()["run_committed"], Decimal("0.0300024"))
        self.assertIsNotNone(proxy.summary()["stopped"])
        self.assertEqual(proxy.summary()["usage_overruns"], 1)
        self.assertEqual(proxy.summary()["forwarded"], 1)
        self.assertEqual(proxy.summary()["successful"], 0)

    def test_error_usage_is_charged_without_leaking_provider_details(self):
        with FixedProvider(status=500) as fake, self.proxy(fake) as proxy:
            status, payload = self.request(proxy)
        self.assertEqual(status, 500)
        self.assertNotIn(KEY.encode(), payload)
        self.assertEqual(self.ledger.totals()["run_committed"], Decimal("0.0000324"))
        self.assertEqual(proxy.summary()["forwarded"], 1)
        self.assertEqual(proxy.summary()["successful"], 0)

    def test_errors_without_usage_keep_reservation_and_stop_after_three(self):
        with FixedProvider(status=500, usage={}) as fake, self.proxy(fake) as proxy:
            for _ in range(4):
                self.request(proxy)
        self.assertEqual(len(fake.requests), 3)
        self.assertEqual(self.ledger.totals()["run_committed"], 3 * token_cost(crusoe.prices(MODEL), 4096, 1024))
        self.assertEqual(proxy.summary()["forwarded"], 3)
        self.assertIsNotNone(proxy.summary()["stopped"])

    def test_success_payload_redacts_provider_key_in_json_strings(self):
        with FixedProvider(content="Provider said " + KEY) as fake, self.proxy(fake) as proxy:
            status, payload = self.request(proxy)
        self.assertEqual(status, 200)
        self.assertNotIn(KEY.encode(), payload)
        self.assertIn(b"[REDACTED]", payload)

    def test_invalid_cache_usage_keeps_reservation_and_stops(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 101}}
        with FixedProvider(usage=usage) as fake, self.proxy(fake) as proxy:
            self.assertEqual(self.request(proxy)[0], 502)
            self.assertEqual(self.request(proxy)[0], 200)
        self.assertEqual(len(fake.requests), 1)
        self.assertEqual(self.ledger.totals()["run_committed"], token_cost(crusoe.prices(MODEL), 4096, 1024))
        self.assertIsNotNone(proxy.summary()["stopped"])

    def test_absolute_transport_deadline_interrupts_a_slow_drip(self):
        class DripHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Length", "100")
                self.end_headers()
                try:
                    for _ in range(20):
                        self.wfile.write(b"x")
                        self.wfile.flush()
                        time.sleep(0.05)  # each byte arrives inside the socket timeout
                except OSError:
                    pass
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), DripHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with MeteringProxy(f"http://127.0.0.1:{server.server_address[1]}/v1", KEY,
                    MODEL, crusoe.prices(MODEL), self.ledger, 1024, timeout=0.2,
                    max_input_tokens=4096) as proxy:
                started = time.monotonic()
                self.assertEqual(self.request(proxy)[0], 502)
                self.assertLess(time.monotonic() - started, 0.8)
            self.assertEqual(self.ledger.totals()["run_committed"], token_cost(crusoe.prices(MODEL), 4096, 1024))
            self.assertEqual(proxy.summary()["forwarded"], 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(3)

    def test_shutdown_waits_for_forwarded_request_and_settlement(self):
        results = []
        with FixedProvider(delay=0.75) as fake:
            with self.proxy(fake, timeout=3) as proxy:
                request = urllib.request.Request(proxy.base_url + "/chat/completions",
                    data=json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}).encode(),
                    headers={"Authorization": "Bearer " + proxy.token})
                def client():
                    with NO_PROXY.open(request, timeout=5) as response:
                        results.append(response.status)
                thread = threading.Thread(target=client)
                thread.start()
                self.assertTrue(fake.entered.wait(3))
            self.assertEqual(proxy.summary()["forwarded"], 1)
            self.assertEqual(proxy.summary()["successful"], 1)
            self.assertEqual(self.ledger.totals()["run_open_reservations"], 0)
            thread.join(5)
        self.assertEqual(results, [200])


if __name__ == "__main__":
    unittest.main()
