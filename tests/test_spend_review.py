"""Regression checks for persistent metering invariants; no provider calls."""

from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

from search.spend import LedgerError, SpendLedger


class SpendLedgerReviewTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="spend-review-")
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "ledger.jsonl"

    def ledger(self, run="owner", cap="1.00"):
        return SpendLedger(run, cap, path=self.path)

    def test_over_bound_charge_is_recorded_and_stops_every_run_after_reopening(self):
        ledger = self.ledger()
        first = ledger.reserve("0.10")
        pending = ledger.reserve("0.20")
        ledger.settle(first, "0.11")

        other = self.ledger("later")
        totals = other.totals()
        self.assertEqual(totals["lifetime_committed"], Decimal("0.31"))
        self.assertEqual(totals["reservation_violation"], {
            "reservation_id": first, "run": "owner", "reserved_usd": "0.10",
            "charged_usd": "0.11",
        })
        for instance in (ledger, other, self.ledger()):
            with self.subTest(run=instance.run_id), self.assertRaisesRegex(LedgerError, "bound"):
                instance.reserve("0.01")

        # Paid requests already in flight must still be accounted for honestly.
        self.ledger().settle(pending, "0.03")
        totals = self.ledger().totals()
        self.assertEqual(totals["lifetime_committed"], Decimal("0.14"))
        self.assertEqual(totals["run_open_reservations"], Decimal(0))
        with self.assertRaises(LedgerError):
            self.ledger("another").reserve("0.01")

    def test_charge_above_caps_is_not_hidden(self):
        ledger = self.ledger()
        ledger.settle(ledger.reserve("1.00"), "51.00")
        totals = ledger.totals()
        self.assertEqual(totals["run_committed"], Decimal("51.00"))
        self.assertEqual(totals["lifetime_committed"], Decimal("51.00"))
        self.assertEqual(totals["lifetime_remaining"], Decimal(0))
        self.assertIsNotNone(totals["reservation_violation"])

    def test_another_run_cannot_release_a_reservation(self):
        owner = self.ledger()
        reservation = owner.reserve("0.75")
        before = self.path.read_bytes()
        with self.assertRaisesRegex(LedgerError, "another run"):
            self.ledger("other").settle(reservation, "0")
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(owner.totals()["run_committed"], Decimal("0.75"))
        owner.settle(reservation, "0.10")
        self.assertEqual(owner.totals()["run_committed"], Decimal("0.10"))
        self.assertIsNone(owner.totals()["reservation_violation"])

    def test_invalid_history_rejects_reads_reservations_and_settlements(self):
        reserve = {"op": "reserve", "id": "first", "run": "owner", "usd": "0.10"}
        settled = dict(reserve, op="settle", usd="0.05")
        corruptions = {
            "duplicate reservation": json.dumps(reserve),
            "orphan settlement": json.dumps(dict(settled, id="missing")),
            "wrong run": json.dumps(dict(settled, run="other")),
            "double settlement": json.dumps(settled) + "\n" + json.dumps(settled),
            "empty run": json.dumps(dict(reserve, id="second", run="")),
            "empty id": json.dumps(dict(reserve, id="")),
            "negative amount": json.dumps(dict(reserve, id="second", usd="-1")),
            "duplicate field": '{"op":"reserve","id":"second","run":"owner",'
                               '"usd":"0.50","usd":"0.00"}',
            "partial record": '{"op":"reserve",',
        }
        baseline = json.dumps(reserve) + "\n"
        for name, suffix in corruptions.items():
            with self.subTest(corruption=name):
                self.path.write_text(baseline + suffix + "\n", encoding="utf-8")
                before = self.path.read_bytes()
                ledger = self.ledger()
                for operation in (ledger.totals, lambda: ledger.reserve("0.01"),
                                  lambda: ledger.settle("first", "0.01")):
                    with self.assertRaises(LedgerError):
                        operation()
                    self.assertEqual(self.path.read_bytes(), before)

    def test_invalid_encoding_fails_closed(self):
        self.path.write_bytes(b"\xff\n")
        with self.assertRaises(LedgerError):
            self.ledger().reserve("0.01")
        self.assertEqual(self.path.read_bytes(), b"\xff\n")

    def test_unterminated_record_cannot_be_joined_to_a_new_reservation(self):
        ledger = self.ledger()
        ledger.reserve("0.10")
        incomplete = self.path.read_bytes().rstrip(b"\n")
        self.path.write_bytes(incomplete)
        with self.assertRaisesRegex(LedgerError, "incomplete final line"):
            ledger.reserve("0.01")
        self.assertEqual(self.path.read_bytes(), incomplete)

    def test_invalid_append_does_not_poison_ledger(self):
        ledger = self.ledger()
        reservation = ledger.reserve("0.10")
        before = self.path.read_bytes()
        with self.assertRaises(LedgerError):
            ledger.settle(reservation, "0.01", unsupported_detail=object())
        for entry_id in (None, [], ""):
            with self.subTest(entry_id=entry_id), self.assertRaises(LedgerError):
                ledger.settle(entry_id, "0.01")
        self.assertEqual(self.path.read_bytes(), before)
        ledger.settle(reservation, "0.01")
        self.assertEqual(ledger.totals()["run_committed"], Decimal("0.01"))


if __name__ == "__main__":
    unittest.main()
