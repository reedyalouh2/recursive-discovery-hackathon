"""Regression checks for failures at the search/verifier boundary."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from search.config import make_config
from search.domains.c7 import C7StrongPower
from search.rng import Rng
from search.run import negative_control

ROOT = Path(__file__).resolve().parents[1]


class SearchIntegrationTests(unittest.TestCase):
    def test_iterator_candidates_are_checked_for_adjacency(self):
        task = C7StrongPower(1)
        self.assertFalse(task.is_independent(iter([0, 1])))
        self.assertTrue(task.is_independent(iter([0, 2, 4])))

    def test_rng_rejects_impossible_rejection_sampling_bound(self):
        with self.assertRaises(ValueError):
            Rng(0).randbelow((1 << 64) + 1)
        self.assertEqual(Rng(0).randbelow(1 << 64), Rng(0).next_u64())

    def test_malformed_engine_names_return_structured_cli_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            for name in ([], {}, None, True):
                with self.subTest(name=name):
                    config = make_config(1, 0, 1)
                    config["engine"]["name"] = name
                    config_path.write_text(json.dumps(config), encoding="utf-8")
                    process = subprocess.run(
                        [sys.executable, "-m", "search", "run", "--config", str(config_path)],
                        cwd=ROOT, capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(process.returncode, 2)
                    self.assertEqual(json.loads(process.stdout)["status"], "error")
                    self.assertEqual(process.stderr, "")

    def test_checker_failure_does_not_count_as_negative_control_rejection(self):
        cases = [
            {"valid": False, "exit_code": 1, "result": None},
            {"valid": False, "exit_code": -9, "result": None},
            {"valid": False, "exit_code": 1, "result": ["unexpected"]},
            {"valid": False, "exit_code": 1, "result": {"reason": "internal failure"}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            for verdict in cases:
                with self.subTest(verdict=verdict), patch("search.run.official.check", return_value=verdict):
                    result = negative_control(
                        ROOT / "examples/c7_calibration.json", Path(directory) / "negative.json",
                    )
                    self.assertFalse(result["rejected"])


if __name__ == "__main__":
    unittest.main()
