"""Controlled confidentiality regressions for the candidate OS boundary."""

import errno
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from search import isolation


REASON = isolation.unavailable_reason()


@unittest.skipIf(REASON, "candidate isolation unavailable: %s" % REASON)
class CandidateConfidentialityTests(unittest.TestCase):
    def run_program(self, source, args=()):
        result = isolation.run_candidate(
            source, list(args),
            isolation.Limits(wall_seconds=5, cpu_seconds=3,
                             memory_bytes=128 * 2 ** 20, file_bytes=2 ** 20))
        self.assertEqual(result["status"], "ok", result["stderr_tail"])
        self.assertIsNotNone(result["output"])
        return json.loads(result["output"].decode("utf-8"))

    def test_other_process_environment_is_unreadable(self):
        # Only this helper's explicit synthetic environment is inspected. No
        # parent environment or unrelated process is ever used as the target.
        helper = subprocess.Popen(
            [sys.executable, "-I", "-S", "-c",
             "import time; print('ready', flush=True); time.sleep(30)"],
            env={"PATH": "/usr/bin:/bin", "REVIEW_CANARY_ONLY": "synthetic-canary-29b1"},
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            self.assertEqual(helper.stdout.readline(), b"ready\n")
            result = self.run_program('''
import os, sys, types
# ctypes calls uname on import. This shim keeps the probe meaningful even if
# ordinary sysctl reads are also blocked: it does not change the target syscall.
os.uname = lambda: types.SimpleNamespace(release='25.0.0')
import ctypes, json
libc = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True)
libc.sysctl.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_uint,
                       ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                       ctypes.c_void_p, ctypes.c_size_t]
libc.sysctl.restype = ctypes.c_int
# CTL_KERN / KERN_PROCARGS2, restricted to the synthetic helper's supplied PID.
mib = (ctypes.c_int * 3)(1, 49, int(sys.argv[1]))
buffer = ctypes.create_string_buffer(262144)
size = ctypes.c_size_t(len(buffer))
code = libc.sysctl(mib, 3, buffer, ctypes.byref(size), None, 0)
error = ctypes.get_errno()
found = b'REVIEW_CANARY_ONLY=synthetic-canary-29b1' in buffer.raw[:size.value]
json.dump({'return_code': code, 'errno': error, 'has_canary': found},
          open('output.json', 'w'))
''', [helper.pid])
            self.assertEqual(result["return_code"], -1, result)
            self.assertIn(result["errno"], (errno.EPERM, errno.EACCES), result)
            self.assertFalse(result["has_canary"], result)
        finally:
            if helper.poll() is None:
                helper.terminate()
                try:
                    helper.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    helper.kill()
                    helper.wait(timeout=5)
            helper.stdout.close()

    def test_hardlink_cannot_exfiltrate_external_canary(self):
        with tempfile.TemporaryDirectory(prefix="isolation-review-canary-") as directory:
            canary = Path(directory) / "only-synthetic-canary.txt"
            canary.write_text("synthetic-file-canary-91bf", encoding="utf-8")
            result = self.run_program('''
import json, os, sys
try:
    os.link(sys.argv[1], 'linked-canary')
    result = {'linked': True}
except OSError as error:
    result = {'linked': False, 'errno': error.errno}
json.dump(result, open('output.json', 'w'))
''', [str(canary)])
            self.assertFalse(result["linked"], result)
            self.assertIn(result["errno"], (errno.EPERM, errno.EACCES), result)
            self.assertEqual(canary.read_text(encoding="utf-8"), "synthetic-file-canary-91bf")


if __name__ == "__main__":
    unittest.main()
