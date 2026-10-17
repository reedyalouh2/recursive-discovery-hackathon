"""Integration regressions for Shinka's interpreter and evaluator boundaries."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from integrations.shinka.engine import ShinkaEngine, _read_summary
from integrations.shinka.evaluate import evaluate
from search.config import make_config
from search.domains.c7 import C7StrongPower
from search.errors import EngineError
from search.run import execute


def summary(version="0.0.7", rows=None):
    return {"shinka_version": version, "python": "3.12.13", "model_requests": 0,
            "api_cost": 0.0, "generations": [] if rows is None else rows}


class ShinkaIntegrationTests(unittest.TestCase):
    def test_failed_evaluations_do_not_produce_successful_empty_run(self):
        failed = {"generation": 0, "correct": False, "combined_score": 0.0,
                  "error": "candidate timed out", "certificate": None}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "run"

            def driver_failed_candidates(command, **kwargs):
                spec = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
                results = Path(spec["results_dir"])
                results.mkdir(parents=True)
                (results / "driver_summary.json").write_text(
                    json.dumps(summary(rows=[failed])), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0)

            with patch.object(ShinkaEngine, "preflight", return_value=Path(sys.executable)), \
                    patch("integrations.shinka.engine.subprocess.run", side_effect=driver_failed_candidates):
                with self.assertRaisesRegex(EngineError, "no verified candidate"):
                    execute(make_config(1, 0, 1, engine="shinka"), out)
            self.assertFalse((out / "manifest.json").exists())
            self.assertFalse((out / "certificate.json").exists())

    def test_absent_generations_are_not_a_valid_empty_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(EngineError, "no verified candidate"):
                ShinkaEngine._best_verified(C7StrongPower(1), Path(directory), [], Mock(), 0)

    def test_explicitly_evaluated_empty_construction_remains_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certificate = {"format_version": 1, "problem": "c7_strong_power",
                           "dimension": 1, "vertices": []}
            (root / "empty.json").write_text(json.dumps(certificate), encoding="utf-8")
            rows = [{"generation": 0, "correct": True, "combined_score": 0.0,
                     "error": None, "certificate": "empty.json"}]
            self.assertEqual(ShinkaEngine._best_verified(C7StrongPower(1), root, rows, Mock(), 0), [])

    def test_evaluator_rejects_valid_certificate_for_different_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            program = root / "candidate.py"
            certificate = {"format_version": 1, "problem": "c7_strong_power",
                           "dimension": 2, "vertices": [[0, 0], [2, 2], [4, 4]]}
            program.write_text(
                "import json, sys\n"
                "output = sys.argv[sys.argv.index('--output') + 1]\n"
                f"json.dump({certificate!r}, open(output, 'w'))\n", encoding="utf-8")
            metrics, correct, reason = evaluate(program, root / "results", 1, 5)
            self.assertFalse(correct)
            self.assertEqual(metrics["combined_score"], 0.0)
            self.assertIn("does not match requested dimension", reason)

    def test_preflight_rejects_wrong_versions_before_creating_run(self):
        cases = [
            {"version": "9.9.9", "python": [3, 12]},
            {"version": "0.0.7", "python": [3, 8]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            for metadata in cases:
                with self.subTest(metadata=metadata):
                    out = Path(directory) / "never-created"
                    process = subprocess.CompletedProcess([], 0, json.dumps(metadata), "")
                    with patch.dict(os.environ, {"SHINKA_PYTHON": sys.executable}), \
                            patch("integrations.shinka.engine.subprocess.run", return_value=process):
                        with self.assertRaisesRegex(EngineError, "Invalid ShinkaEvolve environment"):
                            execute(make_config(1, 0, 1, engine="shinka"), out)
                    self.assertFalse(out.exists())

    def test_preflight_rejects_missing_package_before_creating_run(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "never-created"
            process = subprocess.CompletedProcess([], 1, "", "PackageNotFoundError")
            with patch.dict(os.environ, {"SHINKA_PYTHON": sys.executable}), \
                    patch("integrations.shinka.engine.subprocess.run", return_value=process):
                with self.assertRaisesRegex(EngineError, "Invalid ShinkaEvolve environment"):
                    execute(make_config(1, 0, 1, engine="shinka"), out)
            self.assertFalse(out.exists())

    def test_driver_cannot_report_a_different_shinka_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "driver_summary.json"
            path.write_text(json.dumps(summary(version="9.9.9")), encoding="utf-8")
            with self.assertRaisesRegex(EngineError, "expected ShinkaEvolve 0.0.7"):
                _read_summary(path)


if __name__ == "__main__":
    unittest.main()
