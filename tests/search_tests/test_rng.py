import unittest

from search.rng import Rng


class RngTests(unittest.TestCase):
    def test_splitmix64_reference_values(self):
        rng = Rng(0)
        self.assertEqual([rng.next_u64() for _ in range(3)],
                         [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F])

    def test_same_seed_same_stream(self):
        first, second = Rng(42), Rng(42)
        self.assertEqual([first.randbelow(1000) for _ in range(100)],
                         [second.randbelow(1000) for _ in range(100)])
        self.assertEqual(Rng(42).child(3).next_u64(), Rng(42).child(3).next_u64())
        self.assertNotEqual(Rng(42).child(0).next_u64(), Rng(42).child(1).next_u64())

    def test_child_ignores_parent_consumption(self):
        used = Rng(7)
        used.next_u64()
        self.assertEqual(used.child(2).next_u64(), Rng(7).child(2).next_u64())

    def test_ranges(self):
        rng = Rng(1)
        self.assertEqual({rng.randbelow(1) for _ in range(20)}, {0})
        self.assertTrue(all(0 <= rng.randbelow(7) < 7 for _ in range(1000)))
        self.assertTrue(all(0.0 <= rng.random() < 1.0 for _ in range(1000)))

    def test_rejects_invalid_arguments(self):
        for seed in (-1, 2 ** 64, 1.0, True, "1", None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                Rng(seed)
        for n in (0, -3, 2.0, True):
            with self.subTest(n=n), self.assertRaises(ValueError):
                Rng(0).randbelow(n)


if __name__ == "__main__":
    unittest.main()
