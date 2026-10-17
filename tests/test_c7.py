import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from verification.c7 import MAX_FILE_BYTES, check_file, validate

ROOT = Path(__file__).resolve().parents[1]


def certificate(vertices, dimension=1):
    return {"format_version": 1, "problem": "c7_strong_power",
            "dimension": dimension, "vertices": vertices}


class CertificateTests(unittest.TestCase):
    def test_calibration(self):
        result = check_file(ROOT / "examples/c7_calibration.json")
        self.assertTrue(result["valid"])
        self.assertEqual(result["size"], 3)
        self.assertEqual(result["capacity_lower_bound_exact"], "3^(1/1)")

    def test_tensor_product(self):
        vertices = [list(row) for row in itertools.product([0, 2, 4], repeat=3)]
        result = validate(certificate(vertices, 3))
        self.assertTrue(result["valid"])
        self.assertEqual(result["capacity_lower_bound_exact"], "27^(1/3)")

    def test_wraparound_adjacency(self):
        for pair in ([[0], [1]], [[0], [6]], [[1], [0]], [[6], [0]]):
            with self.subTest(pair=pair):
                self.assertFalse(validate(certificate(pair))["valid"])

    def test_every_nonadjacent_modular_difference(self):
        for difference in (2, 3, 4, 5):
            with self.subTest(difference=difference):
                self.assertTrue(validate(certificate([[0], [difference]]))["valid"])

    def test_strong_product_not_cartesian_product(self):
        for pair in ([[0, 0], [1, 6]], [[0, 0], [0, 6]]):
            with self.subTest(pair=pair):
                self.assertFalse(validate(certificate(pair, 2))["valid"])
        self.assertTrue(validate(certificate([[0, 0], [1, 2]], 2))["valid"])

    def test_duplicate_rows(self):
        result = validate(certificate([[0, 2], [0, 2]], 2))
        self.assertFalse(result["valid"])
        self.assertIn("Duplicate", result["reason"])

    def test_strict_coordinate_types_and_range(self):
        for value in (True, False, 2.0, "2", None, -1, 7):
            with self.subTest(value=value):
                self.assertFalse(validate(certificate([[value]]))["valid"])

    def test_malformed_schema(self):
        bad = [None, [], {}, certificate(None), certificate([0]),
               certificate([[0, 2]]), certificate([[0], []])]
        for item in bad:
            with self.subTest(item=item):
                self.assertFalse(validate(item)["valid"])
        for field, values in {
            "format_version": (True, 1.0, 2, "1"),
            "dimension": (True, 1.0, 0, -1, "1"),
            "problem": (None, "C7"),
        }.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    item = certificate([[0]])
                    item[field] = value
                    self.assertFalse(validate(item)["valid"])

    def test_empty_and_singleton(self):
        self.assertTrue(validate(certificate([]))["valid"])
        self.assertTrue(validate(certificate([[0] * 16], 16))["valid"])

    def test_unsupported_caps(self):
        for item in (certificate([[0] * 17], 17), certificate([[0]] * 2001)):
            result = validate(item)
            self.assertFalse(result["valid"])
            self.assertIn("Unsupported", result["reason"])

    def test_file_errors_are_structured(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.json"
            malformed = [b"{", b"\xff", b'{"dimension": 1, "dimension": 2}',
                         b"NaN", b"[" * 2000 + b"]" * 2000,
                         b" " * (MAX_FILE_BYTES + 1)]
            for raw in malformed:
                with self.subTest(length=len(raw)):
                    path.write_bytes(raw)
                    result = check_file(path)
                    self.assertFalse(result["valid"])
                    self.assertIsNone(result["capacity_lower_bound_exact"])
            self.assertFalse(check_file(Path(directory) / "missing")["valid"])
            self.assertFalse(check_file(directory)["valid"])

    def test_cli_success_and_failure(self):
        for args, expected in ((["examples/c7_calibration.json"], 0),
                               (["missing.json"], 1), ([], 1)):
            with self.subTest(args=args):
                process = subprocess.run(
                    [sys.executable, "-m", "verification.c7", *args],
                    cwd=ROOT, capture_output=True, text=True, check=False,
                )
                self.assertEqual(process.returncode, expected)
                self.assertEqual(json.loads(process.stdout)["valid"], expected == 0)
                self.assertEqual(process.stderr, "")


if __name__ == "__main__":
    unittest.main()
