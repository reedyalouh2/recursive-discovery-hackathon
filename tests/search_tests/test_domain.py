import itertools
import json
import unittest

from search.domains.c7 import C7StrongPower, certificate_text
from verification.c7 import validate


def adjacent_by_definition(a, b):
    return a != b and all((x - y) % 7 in (0, 1, 6) for x, y in zip(a, b))


class C7DomainTests(unittest.TestCase):
    def test_index_is_lexicographic(self):
        task = C7StrongPower(3)
        self.assertEqual([task.coords(v) for v in range(task.num_vertices)],
                         list(itertools.product(range(7), repeat=3)))
        self.assertTrue(all(task.index(task.coords(v)) == v for v in range(task.num_vertices)))

    def test_neighbors_match_definition(self):
        for dimension in (1, 2, 3):
            task = C7StrongPower(dimension)
            coords = [task.coords(v) for v in range(task.num_vertices)]
            for v in range(task.num_vertices):
                with self.subTest(dimension=dimension, v=v):
                    expected = [w for w in range(task.num_vertices)
                                if adjacent_by_definition(coords[v], coords[w])]
                    self.assertEqual(sorted(task.neighbors(v)), expected)
                    self.assertEqual(len(expected), 3 ** dimension - 1)

    def test_adjacent_matches_definition(self):
        task = C7StrongPower(2)
        for u in range(task.num_vertices):
            for w in range(task.num_vertices):
                self.assertEqual(task.adjacent(u, w),
                                 adjacent_by_definition(task.coords(u), task.coords(w)))

    def test_is_independent(self):
        task = C7StrongPower(1)
        self.assertTrue(task.is_independent([0, 2, 4]))
        self.assertTrue(task.is_independent([]))
        for bad in ([0, 1], [0, 6], [2, 2], [7], [-1], [True]):
            with self.subTest(bad=bad):
                self.assertFalse(task.is_independent(bad))

    def test_certificate_is_format_v1_and_checker_accepts_it(self):
        task = C7StrongPower(3)
        vertices = sorted(task.index(row) for row in itertools.product([0, 2, 4], repeat=3))
        certificate = json.loads(certificate_text(task.certificate(list(reversed(vertices)))))
        self.assertEqual(set(certificate), {"format_version", "problem", "dimension", "vertices"})
        self.assertEqual(certificate["vertices"], sorted(certificate["vertices"]))
        self.assertEqual(task.vertices_from_certificate(certificate), vertices)
        result = validate(certificate)
        self.assertTrue(result["valid"])
        self.assertEqual(result["size"], 27)

    def test_empty_certificate_text(self):
        task = C7StrongPower(2)
        self.assertEqual(json.loads(certificate_text(task.certificate([])))["vertices"], [])

    def test_certificate_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            C7StrongPower(2).vertices_from_certificate({"dimension": 1, "vertices": [[0]]})

    def test_dimension_limits(self):
        for dimension in (0, 6, -1, True, 2.0, "3"):
            with self.subTest(dimension=dimension), self.assertRaises(ValueError):
                C7StrongPower(dimension)


if __name__ == "__main__":
    unittest.main()
