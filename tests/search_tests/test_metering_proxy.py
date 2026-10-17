from decimal import Decimal
import json
from pathlib import Path
import socket
import tempfile
import unittest
import urllib.error
import urllib.request

from integrations import crusoe
from integrations.metering_proxy import MeteringProxy, token_cost
from search.spend import SpendLedger

from .fake_crusoe import FakeCrusoe

MODEL = "nvidia/Nemotron-3-Super-120B-A12B"
KEY = "test-key-not-real"
NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class MeteringProxyTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)

    def proxy(self, upstream, cap="1.00", max_output_tokens=1024, timeout=10):
        self.ledger = SpendLedger("test-run", cap, path=self.tmp / "ledger.jsonl")
        return MeteringProxy(upstream, KEY, MODEL, crusoe.prices(MODEL), self.ledger, max_output_tokens,
                             log_path=self.tmp / "spend.jsonl", timeout=timeout, max_input_tokens=4096)

    def post(self, proxy, body=None, token=None):
        body = body or {"model": MODEL, "messages": [{"role": "user", "content": "# Current program\n\n"
                                                      "```python\nTRIALS = 1\n```\n"}]}
        request = urllib.request.Request(
            proxy.base_url + "/chat/completions", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {proxy.token if token is None else token}"})
        try:
            with NO_PROXY.open(request, timeout=30) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            with error:
                return error.code, json.loads(error.read().decode("utf-8"))

    def test_forwards_with_provider_key_and_bills_reported_usage(self):
        with FakeCrusoe() as fake, self.proxy(fake.base_url) as proxy:
            status, reply = self.post(proxy, {"model": MODEL, "max_tokens": 99999,
                                              "max_completion_tokens": 50000,
                                              "messages": [{"role": "user", "content": "# Current program\n\n"
                                                            "```python\nTRIALS = 1\n```\n"}]})
        self.assertEqual(status, 200)
        self.assertIn("TRIALS = 2", reply["choices"][0]["message"]["content"])
        sent = fake.requests[0]
        self.assertEqual(sent["headers"]["Authorization"], f"Bearer {KEY}")
        self.assertNotIn(proxy.token, json.dumps(sent))
        self.assertEqual(sent["body"]["max_tokens"], 1024)  # clamped to the proxy ceiling
        self.assertNotIn("max_completion_tokens", sent["body"])
        usage = reply["usage"]
        expected = token_cost(crusoe.prices(MODEL), usage["prompt_tokens"], usage["completion_tokens"],
                              usage["prompt_tokens_details"]["cached_tokens"])
        summary = proxy.summary()
        self.assertEqual(Decimal(summary["charged_usd"]), expected)
        self.assertEqual(summary["forwarded"], 1)
        self.assertEqual(self.ledger.totals()["run_committed"], expected)
        self.assertNotIn(KEY, (self.tmp / "spend.jsonl").read_text())

    def test_rejects_bad_tokens_and_unsupported_requests(self):
        with FakeCrusoe() as fake, self.proxy(fake.base_url) as proxy:
            self.assertEqual(self.post(proxy, token="wrong")[0], 401)
            for body in ({"model": "zai/GLM-5.3", "messages": []},
                         {"model": MODEL, "messages": [], "stream": True},
                         {"model": MODEL, "messages": [], "n": 2},
                         {"model": MODEL, "messages": [], "max_tokens": 0},
                         {"model": MODEL}):
                with self.subTest(body=body):
                    self.assertEqual(self.post(proxy, body)[0], 400)
        self.assertEqual(fake.requests, [])
        self.assertEqual(self.ledger.totals()["run_committed"], Decimal(0))

    def test_missing_usage_is_charged_at_the_reservation(self):
        with FakeCrusoe(mode="no_usage") as fake, self.proxy(fake.base_url) as proxy:
            self.assertEqual(self.post(proxy)[0], 502)
        committed = self.ledger.totals()["run_committed"]
        self.assertGreater(committed, Decimal(0))
        self.assertEqual(Decimal(proxy.summary()["charged_usd"]), committed)
        self.assertEqual(proxy.summary()["usage_missing"], 1)

    def test_upstream_error_is_sanitized_and_reservation_retained(self):
        with FakeCrusoe(mode="error", status=500) as fake, self.proxy(fake.base_url) as proxy:
            status, reply = self.post(proxy)
        self.assertEqual(status, 500)
        self.assertEqual(reply["error"]["message"], "Upstream returned HTTP 500.")
        self.assertGreater(self.ledger.totals()["run_committed"], Decimal(0))

    def test_timeout_after_sending_is_charged_at_the_reservation(self):
        with FakeCrusoe(mode="slow", delay=3) as fake, self.proxy(fake.base_url, timeout=1) as proxy:
            status, _ = self.post(proxy)
        self.assertEqual(status, 502)
        self.assertGreater(self.ledger.totals()["run_committed"], Decimal(0))

    def test_unreachable_upstream_is_free_then_stops(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]  # closed again before use
        with self.proxy(f"http://127.0.0.1:{port}/v1") as proxy:
            statuses = [self.post(proxy)[0] for _ in range(4)]
        self.assertEqual(statuses[:3], [502, 502, 502])
        self.assertEqual(statuses[3], 200)  # refused locally once forwarding stopped
        self.assertIn("upstream unavailable", proxy.summary()["stopped"])
        self.assertEqual(self.ledger.totals()["run_committed"], Decimal(0))

    def test_budget_exhaustion_refuses_without_contacting_upstream(self):
        with FakeCrusoe() as fake, self.proxy(fake.base_url, cap="0.0001") as proxy:
            first = self.post(proxy)
            second = self.post(proxy)
        for status, reply in (first, second):
            self.assertEqual(status, 200)
            self.assertEqual(reply["choices"][0]["message"]["content"], "")
        self.assertEqual(fake.requests, [])
        summary = proxy.summary()
        self.assertEqual(summary["refused"], 2)
        self.assertTrue(summary["stopped"].startswith("budget"))
        self.assertEqual(self.ledger.totals()["run_committed"], Decimal(0))

    def test_spending_stops_at_the_run_cap(self):
        with FakeCrusoe(mode="max_usage") as fake, \
                self.proxy(fake.base_url, cap="0.01", max_output_tokens=1024) as proxy:
            replies = [self.post(proxy) for _ in range(10)]
        charged = Decimal(proxy.summary()["charged_usd"])
        self.assertLessEqual(charged, Decimal("0.01"))
        self.assertEqual(self.ledger.totals()["run_committed"], charged)
        self.assertGreater(len(fake.requests), 0)
        self.assertLess(len(fake.requests), 10)
        self.assertEqual(replies[-1][1]["choices"][0]["message"]["content"], "")

    def test_only_https_or_loopback_upstreams(self):
        ledger = SpendLedger("test-run", "1.00", path=self.tmp / "ledger.jsonl")
        for url in ("http://api.example.test/v1", "ftp://127.0.0.1/v1", "http://localhost:1/v1"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                MeteringProxy(url, KEY, MODEL, crusoe.prices(MODEL), ledger, 1024)
        with self.assertRaises(ValueError):
            MeteringProxy("https://api.example.test/v1", "", MODEL, crusoe.prices(MODEL), ledger, 1024)


class CrusoePricingTests(unittest.TestCase):
    def test_price_table_is_complete_and_sourced(self):
        self.assertGreater(len(crusoe.MODELS), 0)
        for model in crusoe.MODELS:
            prices = crusoe.prices(model)
            self.assertTrue(all(isinstance(value, Decimal) and value >= 0 for value in prices.values()))
            record = crusoe.price_record(model)
            self.assertEqual((record["source"], record["retrieved"]),
                             ("https://crusoe.ai/cloud/pricing", "2026-10-17"))
        self.assertEqual(crusoe.prices(MODEL), {"input": Decimal("0.30"), "output": Decimal("2.40"),
                                                "cached_input": Decimal("0.15")})
        self.assertEqual(crusoe.BASE_URL, "https://api.inference.crusoecloud.com/v1")


if __name__ == "__main__":
    unittest.main()
