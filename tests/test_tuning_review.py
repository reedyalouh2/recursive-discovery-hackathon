"""Offline regression checks for tuning verification and replay evidence."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from search import tuning
from search.errors import EngineError
from verification.c7 import check_file


def small_plan():
    return tuning.protocol(dimension=2, seed=11, trials=5, tuning_seeds=1,
                           heldout_seeds=2, work_budget=5000)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def evaluation_key(params, seed, work_budget):
    return json.dumps(params, sort_keys=True), seed, work_budget


class EvaluationVerificationTests(unittest.TestCase):
    def test_invalid_certificate_never_returns_a_score(self):
        # 0 and 1 encode adjacent vertices (0, 0) and (0, 1).
        result = {"best": [0, 1], "best_size": 2, "work": 5000, "iterations": 1}
        for keep in (False, True):
            with self.subTest(keep=keep), patch.object(
                    tuning.IteratedLocalSearch, "run", return_value=result):
                with self.assertRaises(EngineError):
                    tuning._evaluate((2, tuning.DEFAULT, 1, 5000, keep))

    def test_valid_certificate_with_wrong_reported_size_never_returns_a_score(self):
        # The certificate is independent, but the strategy claims an extra vertex.
        result = {"best": [0, 2], "best_size": 3, "work": 5000, "iterations": 1}
        for keep in (False, True):
            with self.subTest(keep=keep), patch.object(
                    tuning.IteratedLocalSearch, "run", return_value=result):
                with self.assertRaises(EngineError):
                    tuning._evaluate((2, tuning.DEFAULT, 1, 5000, keep))

    def test_source_change_prevents_a_successful_experiment(self):
        original = tuning._source_hashes
        calls = []

        def changed_at_end(snapshot_dir=None):
            calls.append(snapshot_dir)
            hashes = original(snapshot_dir) if snapshot_dir is not None else original()
            if len(calls) > 1:
                hashes = dict(hashes)
                first = next(iter(hashes))
                hashes[first] = "0" * 64 if hashes[first] != "0" * 64 else "1" * 64
            return hashes

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "changed"
            with patch.object(tuning, "_source_hashes", side_effect=changed_at_end):
                with self.assertRaises(EngineError):
                    tuning.run_experiment(small_plan(), out, workers=1)
            self.assertGreaterEqual(len(calls), 2)
            self.assertIsNotNone(calls[0])
            self.assertIsNone(calls[-1])
            if (out / "summary.json").exists():
                self.assertFalse(read_json(out / "summary.json")["sources_unchanged_during_run"])


class TuningEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.plan = small_plan()
        cls.outputs = {}
        cls.summaries = {}
        for workers in (1, 2):
            out = Path(temp.name) / f"workers-{workers}"
            cls.outputs[workers] = out
            cls.summaries[workers] = tuning.run_experiment(cls.plan, out, workers=workers)

    def trials(self, workers=1):
        return [json.loads(line) for line in
                (self.outputs[workers] / "trials.jsonl").read_text(encoding="utf-8").splitlines()]

    def manifests(self, workers=1):
        return {path.parent.name: read_json(path)
                for path in (self.outputs[workers] / "evaluations").glob("*/result.json")}

    def test_every_unique_evaluation_has_a_verified_certificate_and_manifest(self):
        out, summary = self.outputs[1], self.summaries[1]
        trials, manifests = self.trials(), self.manifests()
        self.assertTrue(manifests)
        actual = {}
        required = {"params", "seed", "work_budget", "evaluation_id", "certificate_path",
                    "valid", "size", "work", "iterations"}
        for evaluation_id, result in manifests.items():
            with self.subTest(evaluation_id=evaluation_id):
                self.assertTrue(required <= result.keys())
                self.assertEqual(result["evaluation_id"], evaluation_id)
                expected_path = f"evaluations/{evaluation_id}/certificate.json"
                self.assertEqual(result["certificate_path"], expected_path)
                verdict = check_file(out / expected_path)
                self.assertTrue(verdict["valid"], verdict["reason"])
                self.assertTrue(result["valid"])
                self.assertEqual(verdict["dimension"], self.plan["dimension"])
                self.assertEqual(verdict["size"], result["size"])
                self.assertGreater(result["work"], 0)
                self.assertGreaterEqual(result["iterations"], 0)
                key = evaluation_key(result["params"], result["seed"], result["work_budget"])
                self.assertNotIn(key, actual, "A cached evaluation was executed and saved twice")
                actual[key] = result
                self.assertEqual({p.name for p in (out / "evaluations" / evaluation_id).iterdir()},
                                 {"result.json", "certificate.json"})

        expected = {evaluation_key(trial["params"], seed, self.plan["work_budget_per_run"])
                    for trial in trials for seed in self.plan["tuning_seeds"]}
        for arm in tuning.ARMS:
            budget = (summary["compute"]["end_to_end_budget_per_run"]
                      if arm == "fixed_end_to_end" else self.plan["work_budget_per_run"])
            for seed in self.plan["heldout_seeds"]:
                expected.add(evaluation_key(summary["selected"][arm]["params"], seed, budget))
        self.assertEqual(set(actual), expected)
        self.assertEqual({p.name for p in (out / "evaluations").iterdir()}, set(manifests))

        nominal = (2 * self.plan["trials_per_tuned_method"] * len(self.plan["tuning_seeds"])
                   + len(tuning.ARMS) * len(self.plan["heldout_seeds"]))
        self.assertEqual(summary["verification"]["executed_evaluations"], len(expected))
        self.assertEqual(summary["verification"]["logical_results"], nominal)
        self.assertLess(len(expected), nominal, "The shared default trial should be reused")
        self.assertTrue(summary["verification"]["all_valid"])

    def test_trial_scores_reference_the_saved_evaluations(self):
        manifests, trials = self.manifests(), self.trials()
        for trial in trials:
            self.assertEqual(trial["total_score"], sum(row["size"] for row in trial["tuning"]))
            for row in trial["tuning"]:
                result = manifests[row["evaluation_id"]]
                self.assertEqual(result["params"], trial["params"])
                self.assertEqual(result["work_budget"], self.plan["work_budget_per_run"])
                for key in ("seed", "size", "work", "iterations", "valid", "certificate_path"):
                    self.assertEqual(row[key], result[key], key)
        defaults = [trial for trial in trials if trial["trial"] == 0]
        self.assertEqual(len(defaults), 2)
        self.assertEqual(defaults[0]["params"], tuning.DEFAULT)
        self.assertEqual(defaults[1]["params"], tuning.DEFAULT)
        self.assertEqual(defaults[0]["tuning"], defaults[1]["tuning"])

    def test_compatibility_certificate_directory_contains_only_heldout_copies(self):
        out, summary = self.outputs[1], self.summaries[1]
        manifests = {evaluation_key(r["params"], r["seed"], r["work_budget"]): r
                     for r in self.manifests().values()}
        expected_names = set()
        heldout = read_json(out / "heldout.json")
        rows = {row["seed"]: row for row in heldout["rows"]}
        for arm in tuning.ARMS:
            budget = (summary["compute"]["end_to_end_budget_per_run"]
                      if arm == "fixed_end_to_end" else self.plan["work_budget_per_run"])
            for seed in self.plan["heldout_seeds"]:
                name = f"heldout-{arm}-seed{seed}.json"
                expected_names.add(name)
                result = manifests[evaluation_key(summary["selected"][arm]["params"], seed, budget)]
                self.assertEqual((out / "certificates" / name).read_bytes(),
                                 (out / result["certificate_path"]).read_bytes())
                self.assertEqual(rows[seed][arm], result["size"])
        self.assertEqual({p.name for p in (out / "certificates").iterdir()}, expected_names)

    def test_source_snapshot_matches_every_recorded_hash(self):
        out, summary = self.outputs[1], self.summaries[1]
        self.assertEqual(set(summary["source_sha256"]), set(tuning.SOURCES))
        for relative, expected_hash in summary["source_sha256"].items():
            with self.subTest(source=relative):
                content = (out / "source_snapshot" / relative).read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(), expected_hash)
        self.assertTrue(summary["sources_unchanged_during_run"])

    def test_scores_evaluation_ids_and_counts_are_independent_of_worker_count(self):
        first, second = self.summaries[1], self.summaries[2]
        for key in ("selected", "heldout_mean", "heldout_best", "comparisons", "compute",
                    "verification", "source_sha256"):
            self.assertEqual(first[key], second[key], key)
        self.assertEqual(self.trials(1), self.trials(2))
        self.assertEqual(self.manifests(1), self.manifests(2))
        self.assertEqual(read_json(self.outputs[1] / "heldout.json"),
                         read_json(self.outputs[2] / "heldout.json"))
        for result in self.manifests(1).values():
            relative = result["certificate_path"]
            self.assertEqual((self.outputs[1] / relative).read_bytes(),
                             (self.outputs[2] / relative).read_bytes())


if __name__ == "__main__":
    unittest.main()
