import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

from integrations.shinka.evaluate import evaluate
from search import isolation
from search.config import make_config
from search.errors import EngineError
from search.run import execute

ROOT = Path(__file__).resolve().parents[2]
REASON = isolation.unavailable_reason()
requires_isolation = unittest.skipIf(REASON, f"candidate isolation unavailable here: {REASON}")
MIB = 2 ** 20

# Each probe reports the exception class it hit, or "allowed".
ATTEMPT = '''
import json

def attempt(action):
    try:
        action()
        return "allowed"
    except Exception as error:
        return type(error).__name__
'''
REPORT = "\njson.dump(result, open('output.json', 'w'))\n"


def limits(**overrides):
    values = dict(wall_seconds=20, cpu_seconds=10, memory_bytes=256 * MIB, file_bytes=MIB, open_files=64)
    values.update(overrides)
    return isolation.Limits(**values)


@requires_isolation
class IsolationLimitTests(unittest.TestCase):
    def run_candidate(self, program, args=(), **overrides):
        return isolation.run_candidate(program, list(args), limits(**overrides))

    def probe(self, body, **overrides):
        run = self.run_candidate(ATTEMPT + body + REPORT, **overrides)
        self.assertEqual(run["status"], "ok", run["stderr_tail"])
        return json.loads(run["output"].decode("utf-8"))

    def test_ordinary_candidate_runs_with_arguments(self):
        run = self.run_candidate("import json, sys\nprint('hello')\n"
                                 "json.dump(sys.argv[1:], open('output.json', 'w'))\n",
                                 args=["--dimension", 3])
        self.assertEqual(run["status"], "ok")
        self.assertEqual(json.loads(run["output"].decode("utf-8")), ["--dimension", "3"])
        self.assertIn("hello", run["stdout_tail"])

    def test_network_is_blocked(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            result = self.probe(f"""
import socket
result = {{
    "tcp_connect": attempt(lambda: socket.create_connection(("127.0.0.1", {port}), timeout=2)),
    "udp_send": attempt(lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        .sendto(b"x", ("127.0.0.1", {port}))),
    "tcp_bind": attempt(lambda: socket.socket().bind(("127.0.0.1", 0))),
    "unix_connect": attempt(lambda: socket.socket(socket.AF_UNIX).connect("/private/var/run/syslog")),
    "dns": attempt(lambda: socket.getaddrinfo("example.com", 80)),
}}
""")
            listener.settimeout(0.5)
            with self.assertRaises(socket.timeout):
                listener.accept()
        self.assertEqual(result, {"tcp_connect": "PermissionError", "udp_send": "PermissionError",
                                  "tcp_bind": "PermissionError", "unix_connect": "PermissionError",
                                  "dns": "gaierror"})

    def test_filesystem_access_is_limited_to_scratch(self):
        with tempfile.TemporaryDirectory() as directory:
            secret = Path(directory) / "secret.txt"
            secret.write_text("secret", encoding="utf-8")
            canary = Path(directory) / "canary.txt"
            result = self.probe(f"""
import os
result = {{
    "read_secret": attempt(lambda: open({str(secret)!r}).read()),
    "read_verifier": attempt(lambda: open({str(ROOT / "verification" / "c7.py")!r}).read()),
    "list_home": attempt(lambda: os.listdir({str(Path.home())!r})),
    "write_outside": attempt(lambda: open({str(canary)!r}, "w").write("x")),
    "write_scratch": attempt(lambda: open("inside.txt", "w").write("x")),
    "read_scratch": attempt(lambda: open("inside.txt").read()),
}}
""")
            self.assertFalse(canary.exists())
        self.assertEqual(result, {"read_secret": "PermissionError", "read_verifier": "PermissionError",
                                  "list_home": "PermissionError", "write_outside": "PermissionError",
                                  "write_scratch": "allowed", "read_scratch": "allowed"})

    def test_process_creation_is_blocked(self):
        result = self.probe("""
import os, subprocess

def fork():
    if os.fork() == 0:
        os._exit(0)

result = {
    "subprocess": attempt(lambda: subprocess.run(["/bin/echo", "hi"], check=True)),
    "fork": attempt(fork),
    "exec_shell": attempt(lambda: os.execv("/bin/sh", ["sh", "-c", "true"])),
}
""")
        self.assertEqual(result, {"subprocess": "PermissionError", "fork": "PermissionError",
                                  "exec_shell": "PermissionError"})

    def test_cpu_limit_stops_busy_loop(self):
        run = self.run_candidate("while True:\n    pass\n", cpu_seconds=1, wall_seconds=20)
        self.assertEqual(run["status"], "cpu")
        self.assertGreaterEqual(run["cpu_seconds"], 1)
        self.assertLess(run["wall_seconds"], 10)

    def test_memory_limit_kills_large_allocation(self):
        program = ("data = bytearray(1024 * 2 ** 20)\n"
                   "for i in range(0, len(data), 4096):\n    data[i] = 1\n"
                   "import time\ntime.sleep(10)\n")
        run = self.run_candidate(program, memory_bytes=128 * MIB)
        self.assertEqual(run["status"], "memory")
        self.assertGreater(run["peak_footprint_bytes"], 128 * MIB)
        # Polling cannot guarantee how far an allocation overshoots the cap.
        self.assertLess(run["wall_seconds"], 10)

    def test_memory_under_the_limit_is_allowed(self):
        program = ("import json\ndata = bytearray(64 * 2 ** 20)\n"
                   "for i in range(0, len(data), 4096):\n    data[i] = 1\n"
                   "json.dump(len(data), open('output.json', 'w'))\n")
        run = self.run_candidate(program, memory_bytes=256 * MIB)
        self.assertEqual(run["status"], "ok")
        self.assertGreater(run["max_rss_bytes"], 64 * MIB)  # exact kernel peak; polling may miss it

    def test_wall_clock_limit_kills_sleeper(self):
        run = self.run_candidate("import time\ntime.sleep(60)\n", wall_seconds=1)
        self.assertEqual(run["status"], "timeout")
        self.assertGreaterEqual(run["wall_seconds"], 1)
        self.assertLess(run["wall_seconds"], 5)

    def test_file_size_and_open_file_limits(self):
        result = self.probe("""
handles = []
result = {
    "large_file": attempt(lambda: open("big.bin", "wb").write(b"x" * (2 * 2 ** 20))),
    "small_file": attempt(lambda: open("small.bin", "wb").write(b"x" * 1024)),
    "many_files": attempt(lambda: [handles.append(open("f%d" % i, "w")) for i in range(64)]),
}
for handle in handles:
    handle.close()
""", file_bytes=MIB, open_files=32)
        self.assertEqual(result, {"large_file": "OSError", "small_file": "allowed", "many_files": "OSError"})

    def test_environment_carries_no_secrets(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "CRUSOE_API_KEY": "secret"}):
            result = self.probe("import os\nresult = {'keys': sorted(os.environ), 'home': os.environ['HOME'],"
                                " 'cwd': os.getcwd()}\n")
        # macOS CoreFoundation adds __CF_USER_TEXT_ENCODING when framework Python builds start.
        keys = [key for key in result["keys"] if key != "__CF_USER_TEXT_ENCODING"]
        self.assertEqual(keys, ["HOME", "LANG", "PATH", "PYTHONHASHSEED", "TMPDIR"])
        self.assertEqual(result["home"], result["cwd"])
        self.assertNotEqual(Path(result["home"]), Path.home())
        self.assertFalse(Path(result["cwd"]).exists())  # scratch is removed afterwards

    def test_oversized_or_redirected_output_is_ignored(self):
        cases = {
            "oversized": "open('output.json', 'wb').write(b'0' * 4096)\n",
            "symlink": "import os\nos.symlink('/private/etc/passwd', 'output.json')\n",
        }
        for name, program in cases.items():
            with self.subTest(name=name):
                run = isolation.run_candidate(program, [], limits(), max_output_bytes=1024)
                self.assertEqual(run["status"], "ok", run["stderr_tail"])
                self.assertIsNone(run["output"])


class IsolationFailsClosedTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)
        self.canary = self.tmp / "ran.txt"
        self.program = f"open({str(self.canary)!r}, 'w').write('ran')\n"

    def test_run_candidate_refuses_to_run(self):
        with patch.object(isolation, "unavailable_reason", return_value="test: no isolation"):
            with self.assertRaises(isolation.IsolationUnavailable):
                isolation.run_candidate(self.program, [], limits())
        self.assertFalse(self.canary.exists())

    def test_evaluator_scores_zero_without_running(self):
        program = self.tmp / "main.py"
        program.write_text(self.program, encoding="utf-8")
        with patch.object(isolation, "unavailable_reason", return_value="test: no isolation"):
            metrics, correct, error = evaluate(program, self.tmp / "results", 1, 5)
        self.assertFalse(correct)
        self.assertEqual(metrics["combined_score"], 0.0)
        self.assertIn("isolation unavailable", error)
        self.assertFalse(self.canary.exists())

    def test_shinka_preflight_refuses_before_creating_run(self):
        out = self.tmp / "never-created"
        probe = json.dumps({"version": "0.0.7", "python": [3, 12]})
        with patch.dict(os.environ, {"SHINKA_PYTHON": sys.executable}), \
                patch("integrations.shinka.engine.subprocess.run",
                      return_value=__import__("subprocess").CompletedProcess([], 0, probe, "")), \
                patch.object(isolation, "unavailable_reason", return_value="test: no isolation"):
            with self.assertRaisesRegex(EngineError, "isolation is unavailable"):
                execute(make_config(1, 0, 1, engine="shinka"), out)
        self.assertFalse(out.exists())

    def test_other_platforms_are_unavailable(self):
        with patch.object(isolation.sys, "platform", "linux"):
            self.assertIn("macOS only", isolation._probe())

    def test_limits_must_be_positive(self):
        for name in ("wall_seconds", "cpu_seconds", "memory_bytes", "file_bytes", "open_files"):
            for value in (0, -1, "1", None):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    isolation.Limits(**{name: value})


if __name__ == "__main__":
    unittest.main()
