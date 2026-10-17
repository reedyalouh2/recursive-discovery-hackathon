from decimal import Decimal
from pathlib import Path
import tempfile
import threading
import unittest

from search.spend import LIFETIME_CAP_USD, BudgetExceeded, LedgerError, SpendLedger, usd


class SpendLedgerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "nested" / "ledger.jsonl"

    def ledger(self, run="run", cap="1.00"):
        return SpendLedger(run, cap, path=self.path)

    def test_lifetime_cap_is_fifty_dollars(self):
        self.assertEqual(LIFETIME_CAP_USD, Decimal("50.00"))

    def test_reserve_settle_and_release(self):
        ledger = self.ledger()
        first = ledger.reserve("0.30")
        second = ledger.reserve("0.20")
        totals = ledger.totals()
        self.assertEqual(totals["run_committed"], Decimal("0.50"))
        self.assertEqual(totals["run_open_reservations"], Decimal("0.50"))
        ledger.settle(first, "0.01")
        ledger.settle(second, 0)
        totals = ledger.totals()
        self.assertEqual(totals["run_committed"], Decimal("0.01"))
        self.assertEqual(totals["lifetime_remaining"], Decimal("49.99"))
        self.assertEqual(totals["run_open_reservations"], Decimal(0))

    def test_run_cap_is_enforced(self):
        ledger = self.ledger(cap="0.50")
        ledger.reserve("0.40")
        with self.assertRaises(BudgetExceeded):
            ledger.reserve("0.11")
        ledger.reserve("0.10")  # exactly at the cap is allowed

    def test_lifetime_cap_spans_runs_and_ledger_instances(self):
        earlier = self.ledger(run="earlier", cap="50.00")
        earlier.settle(earlier.reserve("49.99"), "49.99")
        later = SpendLedger("later", "1.00", path=self.path)
        with self.assertRaises(BudgetExceeded):
            later.reserve("0.02")
        later.reserve("0.01")
        self.assertEqual(later.totals()["lifetime_remaining"], Decimal(0))

    def test_unsettled_reservation_keeps_counting_after_a_crash(self):
        self.ledger().reserve("0.75")  # never settled
        self.assertEqual(self.ledger(run="next").totals()["lifetime_committed"], Decimal("0.75"))

    def test_concurrent_reservations_never_exceed_the_cap(self):
        ledgers = [SpendLedger("run", "5.00", path=self.path) for _ in range(4)]
        granted, lock = [], threading.Lock()

        def worker(ledger):
            for _ in range(10):
                try:
                    ledger.reserve("0.30")
                except BudgetExceeded:
                    continue
                with lock:
                    granted.append(1)

        threads = [threading.Thread(target=worker, args=(ledger,)) for ledger in ledgers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(granted), 16)  # floor(5.00 / 0.30)
        self.assertLessEqual(ledgers[0].totals()["lifetime_committed"], LIFETIME_CAP_USD)

    def test_damaged_ledgers_fail_closed(self):
        ledger = self.ledger()
        entry = ledger.reserve("0.10")
        ledger.settle(entry, "0.05")
        for bad in (lambda: ledger.settle(entry, "0.01"), lambda: ledger.settle("missing", "0.01")):
            with self.assertRaises(LedgerError):
                bad()
        with open(self.path, "a", encoding="utf-8") as stream:
            stream.write("{not json\n")
        with self.assertRaises(LedgerError):
            ledger.totals()
        with self.assertRaises(LedgerError):
            ledger.reserve("0.01")

    def test_invalid_amounts_and_caps(self):
        for value in ("-0.01", "NaN", "Infinity", True, None, "abc"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                usd(value)
        self.assertEqual(usd(0.1), Decimal("0.1"))
        for cap in ("0", "50.01", "-1"):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                SpendLedger("run", cap, path=self.path)


if __name__ == "__main__":
    unittest.main()
