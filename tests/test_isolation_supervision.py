"""Regression checks for supervision failures and scratch cleanup."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from integrations.shinka.evaluate import _run_failure
from search import isolation


class SupervisionFailureTests(unittest.TestCase):
    def test_missing_memory_reader_prevents_launch(self):
        with patch.object(isolation, "_footprint_reader", return_value=None), \
                patch.object(isolation.subprocess, "Popen") as spawn:
            with self.assertRaises(isolation.IsolationUnavailable):
                isolation._supervise_local([], {}, subprocess.DEVNULL,
                                           subprocess.DEVNULL, isolation.Limits())
        spawn.assert_not_called()

    def test_lost_memory_telemetry_stops_and_reaps_live_child(self):
        with patch.object(isolation, "_footprint_reader", return_value=lambda pid: None), \
                patch.object(isolation.subprocess, "Popen", wraps=subprocess.Popen) as spawn:
            result = isolation._supervise_local(
                [sys.executable, "-I", "-S", "-c", "import time; time.sleep(2)"],
                {"PATH": "/usr/bin:/bin"}, subprocess.DEVNULL, subprocess.DEVNULL,
                isolation.Limits(wall_seconds=1))
        self.assertEqual(result["status"], "monitor")
        self.assertLess(result["wall_seconds"], 1)
        self.assertEqual(spawn.call_count, 1)
        self.assertIsNotNone(result["signal"])

    def test_evaluator_rejects_monitor_cancelled_and_unknown_status(self):
        for state in ("monitor", "cancelled", "unexpected"):
            with self.subTest(status=state):
                self.assertIsNotNone(_run_failure({"status": state}, isolation.Limits()))
        self.assertIsNone(_run_failure({"status": "ok"}, isolation.Limits()))

    def test_limits_reject_nonfinite_values_and_fractional_resource_caps(self):
        for name in ("wall_seconds", "cpu_seconds", "memory_bytes", "file_bytes", "open_files"):
            for value in (float("inf"), float("nan"), True):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    isolation.Limits(**{name: value})
            if name != "wall_seconds":
                with self.subTest(name=name), self.assertRaises(ValueError):
                    isolation.Limits(**{name: 0.5})

    def test_cleanup_restores_permissions_without_following_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside"
            outside.mkdir(mode=0o700)
            canary = outside / "canary"
            canary.write_text("synthetic cleanup canary")
            scratch = root / "scratch"
            nested = scratch / "nested"
            nested.mkdir(parents=True)
            (nested / "data").write_text("scratch data")
            (scratch / "external-link").symlink_to(outside, target_is_directory=True)
            nested.chmod(0)
            scratch.chmod(0)
            isolation._cleanup_tree(scratch)
            self.assertFalse(scratch.exists())
            self.assertEqual(canary.read_text(), "synthetic cleanup canary")
            self.assertEqual(outside.stat().st_mode & 0o777, 0o700)


@unittest.skipIf(isolation.unavailable_reason(), "candidate isolation unavailable")
class CandidateCleanupTests(unittest.TestCase):
    def test_candidate_cannot_leave_scratch_by_restricting_permissions(self):
        run = isolation.run_candidate(
            "import os, json\n"
            "os.mkdir('nested')\n"
            "open('nested/data', 'w').write('scratch data')\n"
            "print(json.dumps(os.getcwd()), flush=True)\n"
            "os.chmod('nested', 0)\n"
            "os.chmod('.', 0)\n",
            [], isolation.Limits(wall_seconds=5))
        self.assertEqual(run["status"], "ok", run)
        self.assertFalse(Path(json.loads(run["stdout_tail"])).exists())


if __name__ == "__main__":
    unittest.main()
