import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from search import tuning
from search.domains.c7 import C7StrongPower
from search.rng import Rng
from strategies.mis_ils import IteratedLocalSearch
from verification.c7 import check_file

KEYS = ("best", "best_size", "final_size", "iterations", "accepted", "rejected")


class StrategySettingsTests(unittest.TestCase):
    def test_default_settings_reproduce_the_original_strategy(self):
        # Digests recorded from the strategy before the tunable settings were added.
        expected = {(3, 9, 150): "e1007349feb75acc2c9b70bde330b37f47475384c1abd55db858b9435471fa7d",
                    (4, 3, 300): "d1bffcdd6a3a3532984eaf30fe06a377614552223e2799266556c71a5e55ad45"}
        for (dimension, seed, iterations), digest in expected.items():
            with self.subTest(dimension=dimension):
                marks = []
                result = IteratedLocalSearch(C7StrongPower(dimension), Rng(seed)).run(
                    iterations, lambda i, s: marks.append((i, s)))
                payload = json.dumps([{key: result[key] for key in KEYS}, marks], sort_keys=True)
                self.assertEqual(hashlib.sha256(payload.encode()).hexdigest(), digest)

    def test_work_budget_bounds_a_run_deterministically(self):
        task = C7StrongPower(3)
        first = IteratedLocalSearch(task, Rng(4)).run(None, work_budget=50_000)
        second = IteratedLocalSearch(task, Rng(4)).run(None, work_budget=50_000)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first["work"], 50_000)
        self.assertGreater(first["iterations"], 0)
        with self.assertRaises(ValueError):
            IteratedLocalSearch(task, Rng(4)).run(None)

    def test_every_setting_in_the_space_yields_valid_sets(self):
        task = C7StrongPower(2)
        for key, (low, high) in tuning.SPACE.items():
            for value in (low, high):
                with self.subTest(key=key, value=value):
                    params = dict(tuning.DEFAULT, **{key: value})
                    result = IteratedLocalSearch(task, Rng(1), **params).run(None, work_budget=20_000)
                    self.assertTrue(task.is_independent(result["best"]))


class TuningProtocolTests(unittest.TestCase):
    def test_tuning_and_heldout_seeds_are_disjoint(self):
        plan = tuning.protocol(dimension=3, seed=7, trials=8, tuning_seeds=50, heldout_seeds=60)
        self.assertFalse(set(plan["tuning_seeds"]) & set(plan["heldout_seeds"]))
        self.assertEqual((plan["run_class"], plan["record_claim"]), ("baseline", False))
        for bad in ({"trials": 2}, {"work_budget": 10}, {"dimension": 6}, {"heldout_seeds": 0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                tuning.protocol(**bad)

    def test_mutation_stays_in_bounds_and_changes_something(self):
        rng = Rng(3)
        for _ in range(200):
            child, changed = tuning._mutate(tuning.DEFAULT, rng)
            self.assertTrue(changed)
            for key, (low, high) in tuning.SPACE.items():
                self.assertTrue(low <= child[key] <= high)
            self.assertEqual(sorted(k for k in child if child[k] != tuning.DEFAULT[k]), changed)

    def test_small_experiment_end_to_end(self):
        plan = tuning.protocol(dimension=3, seed=1, trials=6, tuning_seeds=2, heldout_seeds=3,
                               work_budget=60_000)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "tune"
            summary = tuning.run_experiment(plan, out, workers=2)
            trials = [json.loads(line) for line in (out / "trials.jsonl").read_text().splitlines()]
            heldout = json.loads((out / "heldout.json").read_text())
            saved_plan = json.loads((out / "protocol.json").read_text())
            certificates = sorted((out / "certificates").iterdir())
            self.assertEqual(saved_plan, plan)
            self.assertEqual(sum(t["method"] == "evolution" for t in trials), 6)
            self.assertEqual(sum(t["method"] == "random_search" for t in trials), 6)
            for method in ("evolution", "random_search"):
                tried = [t for t in trials if t["method"] == method]
                self.assertEqual(tried[0]["params"], tuning.DEFAULT)
                best = max(tried, key=lambda t: (t["total_score"], -t["trial"]))
                self.assertEqual(summary["selected"][method]["params"], best["params"])  # tuning data only
            self.assertTrue(all(t["parent"] is not None for t in trials
                                if t["method"] == "evolution" and t["trial"] >= tuning.MU))
            for name, comparison in summary["comparisons"].items():
                self.assertEqual(comparison["wins"] + comparison["ties"] + comparison["losses"], 3, name)
            self.assertEqual(len(heldout["rows"]), 3)
            self.assertEqual(len(certificates), 12)
            self.assertTrue(summary["sources_unchanged_during_run"])
            compute = summary["compute"]
            self.assertEqual(compute["end_to_end_budget_per_run"],
                             60_000 + -(-max(compute["tuning_work"].values()) // 3))
            self.assertTrue(all(check_file(path)["valid"] for path in certificates))
            self.assertTrue(summary["verification"]["all_valid"])
            self.assertEqual(summary["compute"]["tuning_runs"], {"evolution": 12, "random_search": 12})
            self.assertFalse(summary["record_claim"])
            again = tuning.run_experiment(plan, Path(directory) / "again", workers=1)
            for key in ("selected", "heldout_mean", "comparisons", "compute"):
                self.assertEqual(again[key], summary[key], key)  # independent of worker count

    def test_output_directory_must_be_new(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "keep.txt").write_text("x")
            with self.assertRaises(ValueError):
                tuning.run_experiment(tuning.protocol(dimension=2, trials=5, tuning_seeds=1,
                                                      heldout_seeds=1, work_budget=5000), directory)


if __name__ == "__main__":
    unittest.main()
