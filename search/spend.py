"""Persistent spend ledger with a hard lifetime cap on paid inference.

LIFETIME_CAP_USD ($50.00, raised from $5.00 at the user's request) is a code
constant. Before each paid request the caller reserves the request's
worst-case cost; afterwards it settles the actual cost. Entries are appended
and fsynced under an exclusive file lock, so concurrent runs and checkouts
share one cap. A reservation that was never settled (for example after a
crash) keeps counting at its reserved amount. An unreadable ledger is an
error, never treated as empty.
If reported cost exceeds a reservation, the actual charge is preserved and
the ledger stops all future reservations until its billing assumptions are
reviewed. Already-open reservations may still be settled honestly.

The ledger lives outside the checkouts in ~/.recursive-discovery/spend-ledger.jsonl.
DISCOVERY_SPEND_LEDGER overrides the path (tests use it).
"""

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import json
import os
from pathlib import Path
import threading
import uuid

LIFETIME_CAP_USD = Decimal("50.00")
LEDGER_ENV = "DISCOVERY_SPEND_LEDGER"


class BudgetExceeded(Exception):
    """A reservation would exceed the run budget or the lifetime cap."""


class LedgerError(RuntimeError):
    """The ledger cannot be read or written, so paid requests must stop."""


def ledger_path():
    override = os.environ.get(LEDGER_ENV, "").strip()
    return Path(override) if override else Path.home() / ".recursive-discovery" / "spend-ledger.jsonl"


def usd(value):
    """Exact, finite, non-negative Decimal amount; floats are read via their repr."""
    if isinstance(value, bool):
        raise ValueError("A dollar amount cannot be a boolean.")
    try:
        amount = Decimal(str(value)) if isinstance(value, float) else Decimal(value)
    except (InvalidOperation, TypeError):
        raise ValueError(f"Invalid dollar amount: {value!r}") from None
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"Invalid dollar amount: {value!r}")
    return amount


class SpendLedger:
    def __init__(self, run_id, run_cap, path=None):
        self.path = Path(path) if path is not None else ledger_path()
        self.run_id = str(run_id)
        if not self.run_id:
            raise ValueError("The run id must not be empty.")
        self.run_cap = usd(run_cap)
        if not Decimal(0) < self.run_cap <= LIFETIME_CAP_USD:
            raise ValueError(f"The run cap must be above $0 and at most ${LIFETIME_CAP_USD}.")
        self._lock = threading.Lock()

    def totals(self):
        with self._lock:
            return self._summarize(self._transact())

    def reserve(self, amount, **details):
        """Record a worst-case reservation and return its id, or raise BudgetExceeded."""
        amount = usd(amount)
        with self._lock:
            def check(entries):
                totals = self._summarize(entries)
                violation = totals["reservation_violation"]
                if violation is not None:
                    raise LedgerError(
                        f"Reservation {violation['reservation_id']} was billed "
                        f"${violation['charged_usd']} above its ${violation['reserved_usd']} bound; "
                        "inspect the billing assumptions before any further paid request.")
                if totals["lifetime_committed"] + amount > LIFETIME_CAP_USD:
                    raise BudgetExceeded(
                        f"Lifetime cap ${LIFETIME_CAP_USD}: ${totals['lifetime_committed']} committed, "
                        f"${amount} requested.")
                if totals["run_committed"] + amount > self.run_cap:
                    raise BudgetExceeded(f"Run cap ${self.run_cap}: ${totals['run_committed']} committed, "
                                         f"${amount} requested.")
            entry = dict(details, op="reserve", id=uuid.uuid4().hex, usd=str(amount))
            self._transact(append=entry, check=check)
            return entry["id"]

    def settle(self, entry_id, amount, **details):
        """Record the actual cost, even above its bound; 0 releases a reservation.

        An over-bound settlement permanently stops new reservations in this
        ledger, but other in-flight requests can still record their charges.
        """
        amount = usd(amount)
        if not isinstance(entry_id, str) or not entry_id:
            raise LedgerError("Cannot settle an invalid reservation id.")
        with self._lock:
            def check(entries):
                reserved, settled = self._states(entries)
                if entry_id not in reserved or entry_id in settled:
                    raise LedgerError(f"Cannot settle {entry_id}: no open reservation.")
                if reserved[entry_id][0] != self.run_id:
                    raise LedgerError(f"Cannot settle {entry_id}: reservation belongs to another run.")
            self._transact(append=dict(details, op="settle", id=entry_id, usd=str(amount)), check=check)

    def _transact(self, append=None, check=None):
        """Read every entry under an exclusive lock, optionally check, then append and fsync."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a+", encoding="utf-8") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    stream.seek(0)
                    contents = stream.read()
                    if contents and not contents.endswith("\n"):
                        raise LedgerError(f"Spend ledger {self.path} has an incomplete final line; "
                                          "inspect it before any paid run.")
                    entries = [self._parse(number, line)
                               for number, line in enumerate(contents.splitlines(), 1) if line.strip()]
                    self._states(entries)
                    if check is not None:
                        check(entries)
                    if append is not None:
                        record = dict(append, run=self.run_id,
                                      time=datetime.now(timezone.utc).isoformat(timespec="seconds"))
                        try:
                            encoded = json.dumps(record, sort_keys=True)
                        except (ValueError, TypeError) as error:
                            raise LedgerError(f"Invalid spend ledger entry: {error}") from None
                        parsed = self._parse(len(entries) + 1, encoded)
                        self._states(entries + [parsed])
                        stream.seek(0, os.SEEK_END)
                        stream.write(encoded + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        entries.append(parsed)
                    return entries
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except (OSError, UnicodeError) as error:
            raise LedgerError(f"Spend ledger {self.path} is not usable: {error}") from None

    def _parse(self, number, line):
        try:
            entry = json.loads(line, object_pairs_hook=_unique_fields)
            if (not isinstance(entry, dict) or entry.get("op") not in ("reserve", "settle")
                    or not isinstance(entry.get("id"), str) or not entry["id"]
                    or not isinstance(entry.get("run"), str) or not entry["run"]):
                raise ValueError("missing fields")
            entry["amount"] = usd(entry["usd"])
            return entry
        except (ValueError, KeyError, TypeError) as error:
            raise LedgerError(f"Spend ledger {self.path} line {number} is unreadable ({error}); "
                              "inspect it before any paid run.") from None

    def _states(self, entries):
        """Validate the complete append-only history and return reservation state."""
        reserved, settled = {}, {}
        for entry in entries:
            if entry["op"] == "reserve":
                if entry["id"] in reserved:
                    raise LedgerError(f"Spend ledger {self.path} repeats reservation {entry['id']}.")
                reserved[entry["id"]] = (entry["run"], entry["amount"])
            else:
                if entry["id"] not in reserved or entry["id"] in settled:
                    raise LedgerError(f"Spend ledger {self.path} has an invalid settlement {entry['id']}.")
                if entry["run"] != reserved[entry["id"]][0]:
                    raise LedgerError(f"Spend ledger {self.path} has a settlement for the wrong run "
                                      f"({entry['id']}).")
                settled[entry["id"]] = entry["amount"]
        return reserved, settled

    def _summarize(self, entries):
        reserved, settled = self._states(entries)
        violation = next(({
            "reservation_id": key, "run": run, "reserved_usd": str(amount),
            "charged_usd": str(settled[key]),
        } for key, (run, amount) in reserved.items()
            if key in settled and settled[key] > amount), None)
        committed = {key: settled.get(key, amount) for key, (_, amount) in reserved.items()}
        run_keys = [key for key, (run, _) in reserved.items() if run == self.run_id]
        lifetime = sum(committed.values(), Decimal(0))
        return {
            "lifetime_cap": LIFETIME_CAP_USD,
            "lifetime_committed": lifetime,
            "lifetime_remaining": max(LIFETIME_CAP_USD - lifetime, Decimal(0)),
            "reservation_violation": violation,
            "run_cap": self.run_cap,
            "run_committed": sum((committed[key] for key in run_keys), Decimal(0)),
            "run_open_reservations": sum((reserved[key][1] for key in run_keys if key not in settled),
                                         Decimal(0)),
        }


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate field {key}")
        result[key] = value
    return result
