# Recursive Discovery

Crusoe Recursive Agents Hackathon prototype, October 17, 2026.

Autonomous bounded search for independent sets in strong powers of C7, with exact certificate verification, experiment history, deterministic offline replay, parameter tuning, and a budgeted Crusoe/ShinkaEvolve adapter.

## Run the offline demo

```sh
python3 -m search smoke
python3 -m unittest discover -s tests -v
```

Python 3.8 or newer for offline search. ShinkaEvolve needs its separate Python environment; see integrations/shinka/README.md. Candidate isolation requires macOS.

## Original work

Built during the hackathon: domain adapter and bounded search implementation, certificate verifier, replay/history, candidate isolation and guardian, metering proxy, and tuning experiment integration. ShinkaEvolve is an external dependency. The fixed search follows the ARW independent-set algorithm cited in strategies/mis_ils.py.

## Evidence and limitations

151 tests passed on Python 3.14. The original 20-seed held-out experiment found mean cardinality 101.95 for random-tuned settings, 100.40 for evolution, 99.10 for defaults and 100.15 for defaults with the tuning scan allowance. Budgets match neighbor-list scans, not proven CPU cost. Evolution showed no convincing advantage over the expanded defaults.

Real Crusoe attempts returned HTTP 403/429. Successful live model-driven improvement is unproven. No mathematical record or breakthrough is claimed. Never commit provider keys.

Source snapshot: integrated commit 198c17b.
