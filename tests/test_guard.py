"""Guardian lifecycle tests use trusted short-lived children, not candidates."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from search import guard, isolation


ROOT = Path(__file__).resolve().parents[1]


def environment(directory):
    return {"PATH": "/usr/bin:/bin", "HOME": str(directory), "TMPDIR": str(directory),
            "LANG": "C.UTF-8"}


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@unittest.skipIf(isolation._footprint_reader() is None, "process memory telemetry unavailable")
class GuardianTests(unittest.TestCase):
    def test_normal_completion_returns_result_and_preserves_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scratch = root / "scratch"
            scratch.mkdir()
            with (root / "out").open("wb") as stdout, (root / "err").open("wb") as stderr:
                result = guard.supervise(
                    [sys.executable, "-I", "-S", "-c", "print('trusted child')"],
                    environment(root), stdout, stderr,
                    isolation.Limits(wall_seconds=2, memory_bytes=128 * 2 ** 20),
                    cleanup_paths=[scratch],
                )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["returncode"], 0)
            self.assertEqual((root / "out").read_text().strip(), "trusted child")
            self.assertTrue(scratch.exists())

    def test_evaluator_sigkill_reaps_child_and_cleans_scratch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scratch = root / "scratch"
            scratch.mkdir()
            child_marker, guard_marker = root / "child.pid", root / "guard.pid"
            child_code = ("import os,time; from pathlib import Path; "
                          "Path(%r).write_text(str(os.getpid())); time.sleep(10)" % str(child_marker))
            helper_code = """
import os, subprocess, sys
from pathlib import Path
sys.path.insert(0, %r)
from search import guard, isolation
original_spawn = subprocess.Popen
def tracked_spawn(*args, **kwargs):
    process = original_spawn(*args, **kwargs)
    Path(%r).write_text(str(process.pid))
    return process
guard.subprocess.Popen = tracked_spawn
with open(%r, 'wb') as stdout, open(%r, 'wb') as stderr:
    guard.supervise(%r, %r, stdout, stderr,
                    isolation.Limits(wall_seconds=5, memory_bytes=128 * 2 ** 20),
                    cleanup_paths=[%r])
""" % (str(ROOT), str(guard_marker), str(scratch / "out"), str(scratch / "err"),
       [sys.executable, "-I", "-S", "-c", child_code], environment(root), str(scratch))
            helper = subprocess.Popen([sys.executable, "-I", "-S", "-c", helper_code],
                                      env=environment(root), stdin=subprocess.DEVNULL,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                      start_new_session=True)
            try:
                deadline = time.monotonic() + 5
                while not child_marker.exists() and time.monotonic() < deadline:
                    if helper.poll() is not None:
                        break
                    time.sleep(0.01)
                self.assertTrue(child_marker.exists(), "trusted child never started")
                child_pid = int(child_marker.read_text())
                helper.kill()
                helper.wait(timeout=2)
                deadline = time.monotonic() + 2
                while (alive(child_pid) or scratch.exists()) and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertFalse(alive(child_pid), "parent cancellation did not terminate the child")
                self.assertFalse(scratch.exists(), "guardian did not remove abandoned scratch")
            finally:
                if helper.poll() is None:
                    helper.kill()
                helper.wait(timeout=2)
                for marker in (child_marker, guard_marker):
                    if marker.exists():
                        try:
                            os.killpg(int(marker.read_text()), signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            pass


if __name__ == "__main__":
    unittest.main()
