"""Live-adapter failure diagnostics tested only against a loopback fake."""

from decimal import Decimal
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from integrations.shinka.engine import PROXY_TOKEN_ENV, ShinkaEngine
from search.domains.c7 import C7StrongPower
from search.errors import EngineError
from search.history import History
from search.rng import Rng
from search.spend import SpendLedger
from search_tests.fake_crusoe import FakeCrusoe


MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"


class LiveFailureIntegrationTests(unittest.TestCase):
    def exercise_driver_failure(self, mode, exit_code):
        with tempfile.TemporaryDirectory() as directory, FakeCrusoe(mode=mode, status=401) as fake:
            root = Path(directory)
            ledger_path = root / "ledger.jsonl"
            workspace = root / "engine"
            engine = ShinkaEngine({"model": MODEL})

            def synthetic_driver(command, **kwargs):
                self.assertNotIn("CRUSOE_API_KEY", kwargs["env"])
                spec = json.loads(Path(command[-1]).read_text())
                request = Request(spec["model"]["base_url"] + "/chat/completions",
                                  data=json.dumps({"model": MODEL, "messages": [
                                      {"role": "user", "content": "synthetic request"}],
                                      "max_tokens": 256}).encode(),
                                  headers={"Authorization": "Bearer " + kwargs["env"][PROXY_TOKEN_ENV],
                                           "Content-Type": "application/json"})
                try:
                    with urlopen(request, timeout=5) as response:
                        response.read()
                except HTTPError as error:
                    error.read()
                    error.close()
                results = Path(spec["results_dir"])
                results.mkdir()
                (results / "driver_summary.json").write_text(json.dumps({
                    "shinka_version": "0.0.7", "python": "synthetic", "model_requests": 0,
                    "api_cost": 0, "generations": []}))
                return subprocess.CompletedProcess(command, exit_code)

            env = {"CRUSOE_API_KEY": "synthetic-provider-key", "DISCOVERY_SPEND_LEDGER": str(ledger_path),
                   "DISCOVERY_CRUSOE_BASE_URL": fake.base_url}
            with patch.dict(os.environ, env), \
                    patch.object(engine, "preflight", return_value=Path(sys.executable)), \
                    patch("integrations.shinka.engine.subprocess.run", side_effect=synthetic_driver), \
                    patch.object(engine, "_best_verified", return_value=[[0], [2], [4]]), \
                    History(root / "history.jsonl") as history:
                with self.assertRaises(EngineError) as raised:
                    engine.run(C7StrongPower(1), {"max_cost_usd": 1, "iterations": 2, "restarts": 1},
                               Rng(0), history, workspace=workspace)
            report = json.loads((workspace / "spend_summary.json").read_text())
            self.assertEqual(report["forwarded"], 1)
            self.assertEqual(Decimal(report["run_open_reservations"]), 0)
            self.assertGreater(Decimal(report["charged_usd"]), 0)
            ledger_total = SpendLedger("audit", 1, path=ledger_path).totals()["lifetime_committed"]
            self.assertEqual(Decimal(report["charged_usd"]), ledger_total)
            self.assertNotIn("synthetic-provider-key", (workspace / "spend_summary.json").read_text())
            return str(raised.exception), report

    def test_driver_crash_preserves_settled_spend_report(self):
        error, report = self.exercise_driver_failure("ok", 1)
        self.assertIn("exited with code 1", error)
        self.assertEqual(report["successful"], 1)

    def test_all_provider_errors_do_not_claim_a_successful_live_run(self):
        error, report = self.exercise_driver_failure("error", 0)
        self.assertIn("No provider request succeeded", error)
        self.assertEqual(report["successful"], 0)

    def test_preflight_stops_on_previous_reservation_violation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            ledger = SpendLedger("earlier", 1, path=path)
            ledger.settle(ledger.reserve("0.10"), "0.20")
            with patch.dict(os.environ, {"CRUSOE_API_KEY": "synthetic-key",
                                         "DISCOVERY_SPEND_LEDGER": str(path)}):
                with self.assertRaisesRegex(EngineError, "prior request exceeded"):
                    ShinkaEngine({"model": MODEL})._authorize_paid({"max_cost_usd": 1}, True)


if __name__ == "__main__":
    unittest.main()
