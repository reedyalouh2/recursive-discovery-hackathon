import unittest

from search.domains.c7 import C7StrongPower
from search.rng import Rng
from strategies.mis_ils import IteratedLocalSearch


def state(search):
    return (sorted(search.solution), list(search.tight), list(search.owner_sum),
            sorted(search.free), bytes(search.in_solution))


class IteratedLocalSearchTests(unittest.TestCase):
    def test_reaches_known_small_optima(self):
        # alpha(C7) = 3 and alpha(C7 x C7) = 10 (strong product).
        for dimension, expected in ((1, 3), (2, 10)):
            with self.subTest(dimension=dimension):
                task = C7StrongPower(dimension)
                result = IteratedLocalSearch(task, Rng(1)).run(200)
                self.assertEqual(result["best_size"], expected)
                self.assertTrue(task.is_independent(result["best"]))

    def test_result_is_a_valid_independent_set(self):
        task = C7StrongPower(3)
        result = IteratedLocalSearch(task, Rng(5)).run(100)
        self.assertTrue(task.is_independent(result["best"]))
        self.assertEqual(result["best"], sorted(result["best"]))
        self.assertEqual(result["accepted"] + result["rejected"], 100)

    def test_deterministic_for_a_seed(self):
        task = C7StrongPower(3)
        runs = []
        for _ in range(2):
            improvements = []
            result = IteratedLocalSearch(task, Rng(9)).run(150, lambda i, s: improvements.append((i, s)))
            runs.append((result, improvements))
        self.assertEqual(runs[0], runs[1])

    def test_invariants_hold_after_search(self):
        task = C7StrongPower(2)
        search = IteratedLocalSearch(task, Rng(4))
        search.run(50)
        members = set(search.solution)
        for v in range(task.num_vertices):
            neighbors_in = [w for w in task.neighbors(v) if w in members]
            self.assertEqual(search.tight[v], len(neighbors_in))
            self.assertEqual(search.owner_sum[v], sum(neighbors_in))
            self.assertEqual(bool(search.in_solution[v]), v in members)
            self.assertEqual(search.free_pos[v] >= 0, v not in members and not neighbors_in)
        self.assertEqual(search.free, [])  # local optima are maximal

    def test_undo_restores_exact_state(self):
        task = C7StrongPower(2)
        search = IteratedLocalSearch(task, Rng(3))
        search._local_search()
        for _ in range(25):
            before = state(search)
            search.log = []
            search._perturb()
            search._local_search()
            search._undo()
            self.assertEqual(state(search), before)
            self.assertEqual(search.stack, [])

    def test_initial_construction_is_kept_or_improved(self):
        task = C7StrongPower(2)
        initial = [task.index(row) for row in ((0, 0), (0, 2), (2, 0), (2, 2))]
        search = IteratedLocalSearch(task, Rng(2), initial=initial)
        self.assertEqual(sorted(search.solution), sorted(initial))
        self.assertGreaterEqual(search.run(0)["best_size"], len(initial))

    def test_rejects_invalid_initial_construction(self):
        task = C7StrongPower(1)
        for initial in ([0, 1], [0, 0], [7]):
            with self.subTest(initial=initial), self.assertRaises(ValueError):
                IteratedLocalSearch(task, Rng(0), initial=initial)


if __name__ == "__main__":
    unittest.main()
