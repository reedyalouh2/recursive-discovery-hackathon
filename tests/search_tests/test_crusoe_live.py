from decimal import Decimal
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from integrations import crusoe
from integrations.shinka.engine import DEFAULT_PYTHON, ShinkaEngine
from search.config import ConfigError, make_config
from search.spend import SpendLedger

from .fake_crusoe import FakeCrusoe

ROOT = Path(__file__).resolve().parents[2]
MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"
KEY = "test-key-not-real"
SHINKA_READY = Path(os.environ.get("SHINKA_PYTHON", str(DEFAULT_PYTHON))).is_file()


class CrusoeConfigTests(unittest.TestCase):
    def test_documented_alias_uses_exact_api_model_id(self):
        legacy = "nvidia/Nemotron-3-Super-120B-A12B"
        config = make_config(2, 0, 5, engine="shinka", params={"model": legacy}, max_cost_usd=1)
        self.assertEqual(config["engine"]["params"]["model"], MODEL)
        self.assertEqual(crusoe.prices(legacy), crusoe.prices(MODEL))

    def test_paid_models_need_a_budget_within_the_cap(self):
        config = make_config(2, 0, 5, engine="shinka", params={"model": MODEL}, max_cost_usd=1.25)
        self.assertEqual(config["budget"]["max_cost_usd"], 1.25)
        self.assertTrue(ShinkaEngine.is_paid(config["engine"]["params"]))
        for cost in (0, 50.01, -1, float("nan"), float("inf"), True, "1"):
            with self.subTest(cost=cost), self.assertRaises(ConfigError):
                make_config(2, 0, 5, engine="shinka", params={"model": MODEL}, max_cost_usd=cost)
        make_config(2, 0, 5, engine="shinka", params={"model": MODEL}, max_cost_usd=5)

    def test_unpaid_runs_must_not_have_a_budget(self):
        for engine, params in (("offline-ils", None), ("shinka", None)):
            with self.subTest(engine=engine), self.assertRaises(ConfigError):
                make_config(2, 0, 5, engine=engine, params=params, max_cost_usd=0.5)

    def test_only_priced_models_and_sane_output_ceilings(self):
        for params in ({"model": "nvidia/Nemotron-3-Ultra-550B-A55B"}, {"model": "qwen/Qwen3.8-27B"},
                       {"model": MODEL, "max_output_tokens": 100},
                       {"model": MODEL, "max_output_tokens": 65536}):
            with self.subTest(params=params), self.assertRaises(ConfigError):
                make_config(2, 0, 5, engine="shinka", params=params, max_cost_usd=1)

    def test_identity_records_provider_and_prices(self):
        identity = ShinkaEngine({"model": MODEL}).identity()
        self.assertEqual((identity["provider"], identity["model"]), ("crusoe", MODEL))
        self.assertEqual(identity["pricing"], crusoe.price_record(MODEL))


class LiveRunHarness(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)
        self.ledger_path = self.tmp / "ledger.jsonl"

    def environment(self, base_url, **overrides):
        env = dict(os.environ, DISCOVERY_SPEND_LEDGER=str(self.ledger_path), CRUSOE_API_KEY=KEY,
                   DISCOVERY_CRUSOE_BASE_URL=base_url)
        env.update(overrides)
        return {key: value for key, value in env.items() if value is not None}

    def cli(self, env, *args):
        process = subprocess.run([sys.executable, "-m", "search", "run", *map(str, args)], cwd=ROOT,
                                 env=env, capture_output=True, text=True, check=False)
        return process.returncode, json.loads(process.stdout) if process.stdout else None, process.stderr

    def paid_args(self, out, cost="0.50", iterations=4, allow=True):
        args = ["--engine", "shinka", "--model", MODEL, "--dimension", 2, "--seed", 0,
                "--iterations", iterations, "--max-cost-usd", cost, "--out", out]
        return args + (["--allow-paid"] if allow else [])


@unittest.skipUnless(SHINKA_READY, "ShinkaEvolve environment is not installed; set SHINKA_PYTHON")
class CrusoePreflightTests(LiveRunHarness):
    def assert_refused(self, env, args, reason):
        code, summary, stderr = self.cli(env, *args)
        self.assertEqual((code, summary["status"]), (1, "error"), stderr)
        self.assertIn(reason, summary["reason"])
        self.assertFalse(Path(args[args.index("--out") + 1]).exists())

    def test_paid_runs_need_explicit_opt_in_key_and_headroom(self):
        with FakeCrusoe() as fake:
            out = self.tmp / "run"
            self.assert_refused(self.environment(fake.base_url), self.paid_args(out, allow=False),
                                "--allow-paid")
            self.assert_refused(self.environment(fake.base_url, CRUSOE_API_KEY=None),
                                self.paid_args(out), "CRUSOE_API_KEY")
            self.assert_refused(self.environment("https://api.example.test/v1"), self.paid_args(out),
                                "127.0.0.1")
            self.assert_refused(self.environment(fake.base_url), self.paid_args(out, cost="0.001"),
                                "worst-case request")
            earlier = SpendLedger("earlier", "50.00", path=self.ledger_path)
            earlier.settle(earlier.reserve("49.99"), "49.99")
            self.assert_refused(self.environment(fake.base_url), self.paid_args(out, cost="0.50"),
                                "lifetime cap")
        self.assertEqual(fake.requests, [])


@unittest.skipUnless(SHINKA_READY, "ShinkaEvolve environment is not installed; set SHINKA_PYTHON")
class CrusoeEndToEndTests(LiveRunHarness):
    def run_paid(self, fake, cost, iterations):
        out = self.tmp / "run"
        code, summary, stderr = self.cli(self.environment(fake.base_url),
                                         *self.paid_args(out, cost=cost, iterations=iterations))
        self.assertEqual(code, 0, (summary, stderr))
        manifest = json.loads((out / "manifest.json").read_text())
        events = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
        totals = SpendLedger("check", "5.00", path=self.ledger_path).totals()
        return summary, manifest, events, totals

    def test_paid_run_goes_through_the_proxy_and_is_billed(self):
        with FakeCrusoe() as fake:
            summary, manifest, events, totals = self.run_paid(fake, "0.50", 4)
        self.assertEqual(summary["status"], "verified")
        self.assertEqual((manifest["engine"]["provider"], manifest["engine"]["model"]), ("crusoe", MODEL))
        self.assertEqual(manifest["engine"]["pricing"], crusoe.price_record(MODEL))
        spend = manifest["spend"]
        self.assertGreaterEqual(spend["forwarded"], 1)
        self.assertEqual(spend["refused"], 0)
        self.assertIsNone(spend["stopped"])
        charged = Decimal(spend["charged_usd"])
        self.assertGreater(charged, 0)
        self.assertEqual(totals["lifetime_committed"], charged)
        self.assertEqual(Decimal(manifest["cost"]["usd_exact"]), charged)
        self.assertEqual(len(fake.requests), spend["forwarded"])
        for request in fake.requests:
            self.assertEqual(request["headers"]["Authorization"], f"Bearer {KEY}")
            self.assertEqual(request["body"]["model"], MODEL)
            self.assertLessEqual(request["body"]["max_tokens"], 8192)
        names = [event["event"] for event in events]
        self.assertIn("spend_start", names)
        self.assertIn("spend_end", names)
        self.assertTrue(all(event["correct"] for event in events if event["event"] == "generation"))

    def test_budget_exhaustion_stops_spending_but_keeps_verified_results(self):
        with FakeCrusoe(mode="max_usage") as fake:
            summary, manifest, events, totals = self.run_paid(fake, "0.36", 8)
        spend = manifest["spend"]
        self.assertTrue(spend["stopped"].startswith("budget"))
        self.assertGreaterEqual(spend["refused"], 1)
        self.assertLessEqual(Decimal(spend["charged_usd"]), Decimal("0.36"))
        self.assertLessEqual(totals["lifetime_committed"], Decimal("0.36"))
        self.assertEqual(len(fake.requests), spend["forwarded"])
        self.assertEqual(summary["status"], "verified")


if __name__ == "__main__":
    unittest.main()
