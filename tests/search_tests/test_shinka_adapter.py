import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import urllib.request

from integrations.shinka.engine import DEFAULT_PYTHON, ShinkaEngine
from integrations.shinka.evaluate import evaluate
from integrations.shinka.stub_model import MAX_TRIALS, MODEL_NAME, StubServer, propose
from search import isolation
from search.config import ConfigError, make_config
from verification.c7 import check_file

ROOT = Path(__file__).resolve().parents[2]
SHINKA = ROOT / "integrations" / "shinka"
INITIAL = (SHINKA / "initial_program.py").read_text(encoding="utf-8")
NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def prompt(code, prior=""):
    return (f"{prior}# Current program\n\nHere is the current program:\n\n"
            f"```python\n{code}\n```\n\nHere are the performance metrics of the program:\n")


def apply_reply(code, reply):
    search = reply.split("<<<<<<< SEARCH\n", 1)[1].split("\n=======\n", 1)[0]
    replace = reply.split("\n=======\n", 1)[1].split("\n>>>>>>> REPLACE", 1)[0]
    assert code.count(search) == 1, search
    return code.replace(search, replace)


def cli(*args, env=None):
    process = subprocess.run([sys.executable, "-m", "search", *map(str, args)], cwd=ROOT, env=env,
                             capture_output=True, text=True, check=False)
    return process.returncode, json.loads(process.stdout) if process.stdout else None, process.stderr


class StubModelTests(unittest.TestCase):
    def test_doubles_trials_then_increments_seed(self):
        code, seen = INITIAL, []
        for _ in range(9):
            code = apply_reply(code, propose(prompt(code)))
            seen.append((int(re.search(r"^TRIALS = (\d+)$", code, re.M).group(1)),
                         int(re.search(r"^SEED = (\d+)$", code, re.M).group(1))))
        self.assertEqual(seen, [(2, 0), (4, 0), (8, 0), (16, 0), (32, 0), (MAX_TRIALS, 0),
                                (MAX_TRIALS, 1), (MAX_TRIALS, 2), (MAX_TRIALS, 3)])

    def test_edits_the_current_program_not_prior_programs(self):
        prior = "# Prior programs\n\n```python\nTRIALS = 999\nSEED = 7\n```\n\n"
        self.assertIn("TRIALS = 1\n=======\nTRIALS = 2", propose(prompt(INITIAL, prior)))

    def test_without_editable_line_returns_empty_diff(self):
        reply = propose(prompt("print('no parameters')"))
        self.assertNotIn("SEARCH", reply)
        self.assertIn("<DIFF>\n</DIFF>", reply)

    def test_serves_chat_completions_on_localhost_only(self):
        with StubServer() as server:
            self.assertTrue(server.base_url.startswith("http://127.0.0.1:"))
            body = json.dumps({"model": MODEL_NAME, "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": prompt(INITIAL)}]}).encode("utf-8")
            request = urllib.request.Request(server.base_url + "/chat/completions", data=body,
                                             headers={"Content-Type": "application/json"})
            with NO_PROXY.open(request, timeout=10) as response:
                reply = json.loads(response.read().decode("utf-8"))
            message = reply["choices"][0]["message"]
            self.assertEqual(message["role"], "assistant")
            self.assertIn("TRIALS = 2", message["content"])
            self.assertEqual(server.requests, 1)
            for path, data in (("/embeddings", body), ("/chat/completions", b"not json")):
                with self.subTest(path=path), self.assertRaises(urllib.error.HTTPError):
                    NO_PROXY.open(urllib.request.Request(server.base_url + path, data=data), timeout=10)


class EvaluatorTests(unittest.TestCase):
    def evaluate(self, program, dimension=2, timeout=30):
        with tempfile.TemporaryDirectory() as directory:
            program_path = Path(directory) / "main.py"
            program_path.write_text(program, encoding="utf-8")
            results = Path(directory) / "results"
            process = subprocess.run(
                [sys.executable, str(SHINKA / "evaluate.py"), "--program_path", str(program_path),
                 "--results_dir", str(results), "--dimension", str(dimension), "--timeout", str(timeout)],
                cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(process.returncode, 0, process.stderr)
            metrics = json.loads((results / "metrics.json").read_text())
            correct = json.loads((results / "correct.json").read_text())
            certificate = results / "certificate.json"
            official = check_file(certificate) if certificate.is_file() else None
            return metrics, correct, official

    def write_certificate(self, vertices, dimension=2):
        certificate = {"format_version": 1, "problem": "c7_strong_power",
                       "dimension": dimension, "vertices": vertices}
        return ("import json, sys\n"
                "out = sys.argv[sys.argv.index('--output') + 1]\n"
                f"json.dump({certificate!r}, open(out, 'w'))\n")

    def test_seed_program_scores_the_official_size(self):
        metrics, correct, official = self.evaluate(INITIAL, dimension=3)
        self.assertTrue(correct["correct"])
        self.assertTrue(official["valid"])
        self.assertEqual(metrics["combined_score"], official["size"])
        self.assertEqual(metrics["public"]["size"], official["size"])

    def test_invalid_output_scores_zero(self):
        metrics, correct, official = self.evaluate(self.write_certificate([[0, 0], [0, 1]]))
        self.assertFalse(correct["correct"])
        self.assertFalse(official["valid"])
        self.assertEqual(metrics["combined_score"], 0.0)
        self.assertIn("rejected", correct["error"])

    def test_crash_timeout_and_missing_output_score_zero(self):
        cases = {
            "crash": ("raise RuntimeError('boom')\n", 30, "exited with code"),
            "timeout": ("import time\ntime.sleep(30)\n", 1, "time limit"),
            "no output": ("pass\n", 30, "no certificate"),
        }
        for name, (program, timeout, reason) in cases.items():
            with self.subTest(name=name):
                metrics, correct, _ = self.evaluate(program, timeout=timeout)
                self.assertFalse(correct["correct"])
                self.assertEqual(metrics["combined_score"], 0.0)
                self.assertIn(reason, correct["error"])

    def test_candidate_cannot_reach_the_results_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results"
            results.mkdir()
            program_path = Path(directory) / "main.py"
            program_path.write_text(
                "import json\n"
                f"for name in ('metrics.json', 'correct.json', 'forged.txt'):\n"
                f"    try:\n"
                f"        open({str(results)!r} + '/' + name, 'w').write('{{\"combined_score\": 999}}')\n"
                f"    except PermissionError:\n"
                f"        pass\n"
                "json.dump({'format_version': 1, 'problem': 'c7_strong_power', 'dimension': 2,\n"
                "           'vertices': [[0, 0], [2, 2], [4, 4]]}, open('output.json', 'w'))\n",
                encoding="utf-8")
            if isolation.unavailable_reason():
                self.skipTest(f"candidate isolation unavailable: {isolation.unavailable_reason()}")
            metrics, correct, error = evaluate(program_path, results, 2, 30)
            self.assertTrue(correct, error)
            self.assertEqual(metrics["combined_score"], 3.0)  # from verification.c7, not the candidate
            self.assertEqual(sorted(path.name for path in results.iterdir()), ["certificate.json"])


class ShinkaConfigTests(unittest.TestCase):
    def test_defaults_and_identity(self):
        config = make_config(3, 0, 5, engine="shinka")
        self.assertEqual(config["engine"]["params"],
                         {"model": MODEL_NAME, "eval_timeout_seconds": 60, "eval_cpu_seconds": 60,
                          "eval_memory_mb": 1024, "num_islands": 1, "max_output_tokens": 8192})
        identity = ShinkaEngine(config["engine"]["params"]).identity()
        self.assertEqual(identity["upstream"]["package"], "shinka-evolve==0.0.7")
        self.assertEqual(identity["upstream"]["git_commit"], "c4568adde253cacf185be3a8412c3c2142761ebe")
        self.assertFalse(ShinkaEngine.deterministic)

    def test_rejects_live_models_and_out_of_range_settings(self):
        cases = [
            dict(iterations=5, params={"model": "local/nemotron@https://api.example.test/v1"}),
            dict(iterations=5, params={"model": "gpt-5.4"}),
            dict(iterations=5, params={"eval_timeout_seconds": 0}),
            dict(iterations=5, params={"num_islands": 9}),
            dict(iterations=5, params={"eval_cpu_seconds": 0}),
            dict(iterations=5, params={"eval_memory_mb": 32}),
            dict(iterations=5, params={"eval_memory_mb": 1.5}),
            dict(iterations=5, params={"temperature": 1.0}),
            dict(iterations=0, params={}),
            dict(iterations=501, params={}),
            dict(iterations=5, params={}, restarts=11),
            dict(iterations=5, params={}, initial_sha256="0" * 64),
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ConfigError):
                make_config(2, 0, case["iterations"], case.get("restarts", 1), engine="shinka",
                            params=case["params"], initial_sha256=case.get("initial_sha256"))

    def test_missing_environment_fails_before_writing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, SHINKA_PYTHON=str(Path(directory) / "missing-python"))
            out = Path(directory) / "run"
            code, summary, _ = cli("run", "--engine", "shinka", "--dimension", 2, "--seed", 0,
                                   "--iterations", 2, "--out", out, env=env)
            self.assertEqual((code, summary["status"]), (1, "error"))
            self.assertIn("uv venv", summary["reason"])
            self.assertFalse(out.exists())


@unittest.skipUnless(Path(os.environ.get("SHINKA_PYTHON", str(DEFAULT_PYTHON))).is_file(),
                     "ShinkaEvolve environment is not installed; set SHINKA_PYTHON to reuse one")
class ShinkaEndToEndTests(unittest.TestCase):
    def test_stub_run_is_verified_and_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            code, summary, stderr = cli("run", "--engine", "shinka", "--dimension", 2, "--seed", 0,
                                        "--iterations", 4, "--out", os.path.relpath(out, ROOT))
            self.assertEqual(code, 0, (summary, stderr))
            self.assertEqual(summary["status"], "verified")
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertTrue(manifest["ok"])
            self.assertFalse(manifest["record_claim"])
            self.assertEqual(manifest["run_class"], "baseline")
            self.assertEqual(manifest["engine"]["name"], "shinka")
            self.assertEqual(manifest["engine_runtime"]["shinka_version"], "0.0.7")
            self.assertEqual(manifest["deterministic_artifacts"], ["config.json"])
            self.assertEqual(manifest["cost"]["usd"], 0.0)
            self.assertEqual(manifest["cost"]["llm_calls"], 3)  # generations 1-3; 0 is the seed

            self.assertTrue(check_file(out / "certificate.json")["valid"])
            events = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
            generations = [event for event in events if event["event"] == "generation"]
            self.assertEqual([event["generation"] for event in generations], [0, 1, 2, 3])
            self.assertTrue(all(event["correct"] for event in generations))
            self.assertEqual(manifest["official"]["size"], max(event["score"] for event in generations))
            self.assertTrue((out / "engine" / "restart_0" / "programs.sqlite").is_file())


if __name__ == "__main__":
    unittest.main()
