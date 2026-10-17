import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from search.config import ConfigError, make_config, validate_config

ROOT = Path(__file__).resolve().parents[2]
DETERMINISTIC = ("config.json", "certificate.json", "history.jsonl")


def cli(*args):
    process = subprocess.run([sys.executable, "-m", "search", *map(str, args)], cwd=ROOT,
                             capture_output=True, text=True, check=False)
    return process.returncode, json.loads(process.stdout) if process.stdout else None, process.stderr


def checker(path):
    process = subprocess.run([sys.executable, "-m", "verification.c7", str(path)], cwd=ROOT,
                             capture_output=True, text=True, check=False)
    return process.returncode, json.loads(process.stdout)


class BoundedRunTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.tmp = Path(self._directory.name)

    def tearDown(self):
        self._directory.cleanup()

    def run_search(self, name, *args):
        code, summary, stderr = cli("run", *args, "--out", self.tmp / name)
        self.assertEqual(code, 0, (summary, stderr))
        return summary, self.tmp / name

    def test_run_emits_verified_certificate_and_history(self):
        summary, out = self.run_search("run", "--dimension", 3, "--seed", 7,
                                       "--iterations", 60, "--restarts", 2)
        self.assertEqual(summary["status"], "verified")
        self.assertEqual(summary["run_class"], "baseline")
        self.assertFalse(summary["record_claim"])

        code, result = checker(out / "certificate.json")  # independent of the search's report
        self.assertEqual(code, 0)
        self.assertEqual(result["size"], summary["size"])
        certificate = json.loads((out / "certificate.json").read_text())
        self.assertEqual(set(certificate), {"format_version", "problem", "dimension", "vertices"})

        manifest = json.loads((out / "manifest.json").read_text())
        self.assertTrue(manifest["ok"])
        self.assertFalse(manifest["record_claim"])
        self.assertEqual(manifest["official"]["size"], manifest["search_reported_size"])
        self.assertEqual(manifest["cost"], {"usd": 0.0, "llm_calls": 0,
                                            "input_tokens": 0, "output_tokens": 0})
        self.assertEqual((manifest["engine"]["provider"], manifest["engine"]["model"]), ("none", "none"))
        self.assertEqual(manifest["compute"]["iterations"], 120)

        events = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
        self.assertEqual([event["seq"] for event in events], list(range(len(events))))
        self.assertEqual(events[0]["event"], "run_start")
        self.assertEqual(events[0]["config_sha256"], manifest["config_sha256"])
        self.assertEqual(events[0]["seed"], 7)
        self.assertEqual(events[-1]["event"], "run_end")
        self.assertEqual(events[-1]["search_reported_size"], summary["size"])
        self.assertEqual(sum(event["event"] == "restart_end" for event in events), 2)

    def test_replay_from_saved_config_is_byte_identical(self):
        _, out = self.run_search("run", "--dimension", 3, "--seed", 11, "--iterations", 80)
        _, replay = self.run_search("replay", "--config", out / "config.json")
        for name in DETERMINISTIC:
            with self.subTest(name=name):
                self.assertEqual((out / name).read_bytes(), (replay / name).read_bytes())

    def test_checker_rejects_deliberately_invalid_output(self):
        _, out = self.run_search("run", "--dimension", 2, "--seed", 1, "--iterations", 20)
        certificate = json.loads((out / "certificate.json").read_text())
        first = certificate["vertices"][0]
        corruptions = {
            "duplicate": certificate["vertices"] + [first],
            "adjacent": certificate["vertices"] + [[first[0], (first[1] + 1) % 7]],
            "out_of_range": [[7, 0]] + certificate["vertices"][1:],
        }
        for name, vertices in corruptions.items():
            with self.subTest(name=name):
                path = self.tmp / f"{name}.json"
                path.write_text(json.dumps(dict(certificate, vertices=vertices)))
                code, result = checker(path)
                self.assertEqual(code, 1)
                self.assertFalse(result["valid"])

    def test_initial_construction_is_copied_recorded_and_replayable(self):
        _, out = self.run_search("seeded", "--dimension", 1, "--seed", 0, "--iterations", 5,
                                 "--initial", ROOT / "examples/c7_calibration.json",
                                 "--run-class", "calibration")
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["run_class"], "calibration")
        self.assertEqual(manifest["initial_construction"]["size"], 3)
        self.assertEqual((out / "input.json").read_bytes(),
                         (ROOT / "examples/c7_calibration.json").read_bytes())
        _, replay = self.run_search("seeded-replay", "--config", out / "config.json")
        for name in DETERMINISTIC + ("input.json",):
            self.assertEqual((out / name).read_bytes(), (replay / name).read_bytes())

    def test_tampered_or_invalid_initial_construction_is_refused(self):
        _, out = self.run_search("seeded", "--dimension", 1, "--seed", 0, "--iterations", 1,
                                 "--initial", ROOT / "examples/c7_calibration.json")
        (out / "input.json").write_text(
            '{"format_version": 1, "problem": "c7_strong_power", "dimension": 1, "vertices": [[0], [3]]}')
        code, summary, _ = cli("run", "--config", out / "config.json", "--out", self.tmp / "tampered")
        self.assertEqual((code, summary["status"]), (2, "error"))
        self.assertIn("sha256", summary["reason"])

        adjacent = self.tmp / "adjacent.json"
        adjacent.write_text(
            '{"format_version": 1, "problem": "c7_strong_power", "dimension": 1, "vertices": [[0], [1]]}')
        code, summary, _ = cli("run", "--dimension", 1, "--seed", 0, "--iterations", 1,
                               "--initial", adjacent, "--out", self.tmp / "adjacent-run")
        self.assertEqual((code, summary["status"]), (2, "error"))
        self.assertIn("rejected", summary["reason"])
        self.assertFalse((self.tmp / "adjacent-run").exists())

    def test_cli_rejects_bad_arguments(self):
        config_path = self.tmp / "record.json"
        record = make_config(2, 0, 10)
        record["run_class"] = "record"
        config_path.write_text(json.dumps(record))
        cases = [
            ("run", "--dimension", 6, "--seed", 0, "--iterations", 1),
            ("run", "--dimension", 2, "--seed", -1, "--iterations", 1),
            ("run", "--dimension", 2, "--seed", 0, "--iterations", -5),
            ("run", "--dimension", 2, "--seed", 0),
            ("run", "--config", config_path),
            ("run", "--config", config_path, "--seed", 3),
            ("run", "--config", self.tmp / "missing.json"),
        ]
        for args in cases:
            with self.subTest(args=args):
                code, summary, _ = cli(*args, "--out", self.tmp / "never")
                self.assertEqual((code, summary["status"]), (2, "error"))
        self.assertFalse((self.tmp / "never").exists())

    def test_refuses_non_empty_output_directory(self):
        occupied = self.tmp / "occupied"
        occupied.mkdir()
        (occupied / "keep.txt").write_text("existing work")
        code, summary, _ = cli("run", "--dimension", 1, "--seed", 0, "--iterations", 1, "--out", occupied)
        self.assertEqual((code, summary["status"]), (2, "error"))
        self.assertEqual(sorted(path.name for path in occupied.iterdir()), ["keep.txt"])

    def test_smoke(self):
        code, summary, stderr = cli("smoke", "--out", self.tmp / "smoke")
        self.assertEqual(code, 0, (summary, stderr))
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["run_class"], "calibration")
        self.assertFalse(summary["record_claim"])
        self.assertTrue(summary["replay_identical"])
        self.assertTrue(summary["negative_control_rejected"])
        self.assertTrue((self.tmp / "smoke" / "smoke.json").is_file())


class ConfigValidationTests(unittest.TestCase):
    def test_normalizes_defaults(self):
        config = make_config(3, 5, 100, 2)
        self.assertEqual(config["engine"]["params"], {"max_perturbation": 4, "tournament": 4,
                                                      "perturb_continue_pct": 50, "perturb_radius": 2,
                                                      "accept_scale_pct": 100})
        self.assertEqual(config["budget"], {"iterations": 100, "restarts": 2, "max_cost_usd": 0.0})
        self.assertEqual(validate_config(json.loads(json.dumps(config))), config)

    def test_rejects_invalid_fields(self):
        base = make_config(2, 0, 10)
        mutations = {
            "unknown key": lambda c: c.update(extra=1),
            "missing key": lambda c: c.pop("seed"),
            "format": lambda c: c.update(format="v0"),
            "problem": lambda c: c.update(problem="c5"),
            "record class": lambda c: c.update(run_class="record"),
            "bool dimension": lambda c: c.update(dimension=True),
            "float seed": lambda c: c.update(seed=1.0),
            "engine": lambda c: c["engine"].update(name="nonexistent"),
            "engine param": lambda c: c["engine"]["params"].update(tournament=0),
            "unknown param": lambda c: c["engine"]["params"].update(temperature=1),
            "restarts": lambda c: c["budget"].update(restarts=0),
            "total budget": lambda c: c["budget"].update(iterations=10_000_000, restarts=6),
            "paid": lambda c: c["budget"].update(max_cost_usd=1),
            "initial path": lambda c: c.update(
                initial_construction={"path": "../x.json", "sha256": "0" * 64}),
            "initial digest": lambda c: c.update(
                initial_construction={"path": "input.json", "sha256": "XYZ"}),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                config = json.loads(json.dumps(base))
                mutate(config)
                with self.assertRaises(ConfigError):
                    validate_config(config)


if __name__ == "__main__":
    unittest.main()
