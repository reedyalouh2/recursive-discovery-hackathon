# ShinkaEvolve adapter

Engine `shinka` runs [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) 0.0.7 (PyPI `shinka-evolve==0.0.7`, tag `v0.0.7` = `c4568ad`) against an offline stub model, or against a priced Crusoe Managed Inference model with local spending reservations.

## Setup

ShinkaEvolve needs Python ≥ 3.10, so it gets its own git-ignored environment. Dependencies are pinned with hashes in `requirements.lock`.

```sh
uv venv --python 3.12 .venv/shinka
uv pip install --python .venv/shinka/bin/python --require-hashes --no-deps -r integrations/shinka/requirements.lock
```

To regenerate the lock, run `uv pip compile integrations/shinka/requirements.in --python-version 3.12 --generate-hashes --exclude-newer <date at least two weeks old> --no-header -o integrations/shinka/requirements.lock`. Set `SHINKA_PYTHON` to use an interpreter elsewhere.

On this machine, the existing installation in Claude's checkout can be reused from the integration or Codex checkout:

```sh
export SHINKA_PYTHON=/Users/mohamed/Documents/ChatGPT/linked/recursive-discovery-worktrees/claude/.venv/shinka/bin/python
```

Preflight checks Python >= 3.10, `shinka-evolve==0.0.7` and that candidate isolation works before creating run artifacts. The end-to-end test also honors `SHINKA_PYTHON`.

## Run

```sh
python3 -m search run --engine shinka --dimension 3 --seed 0 --iterations 6
```

`--iterations` is ShinkaEvolve's `num_generations`. Generation 0 is `initial_program.py`.

## How it fits

- `engine.py` (repo Python) writes a spec and runs `driver.py` in `.venv/shinka`, with model API keys stripped from the environment.
- `driver.py` starts `stub_model.py` on `127.0.0.1` and points ShinkaEvolve at it as `local/offline-stub-v1@http://127.0.0.1:PORT/v1`, the same OpenAI-compatible path a hosted endpoint would use. Embeddings, novelty judging and meta-recommendations are disabled. The stub's diffs double `TRIALS` and then step `SEED`.
- ShinkaEvolve runs `evaluate.py` for each candidate. The evaluator runs the program through `search.isolation` (below), requires the requested dimension, and scores its certificate with `verification.c7`.
- The engine re-checks the best certificate with `verification.c7`, and the run pipeline checks the exported certificate again. No accepted candidate is an engine failure, not an implicitly valid empty construction. The engine also fails on a runtime version mismatch, a reported nonzero cost, or a change to `verification/c7.py` during the run.

## Candidate isolation

`search.isolation` runs each candidate as `python -I -S -B main.py` in a fresh scratch directory under a deny-by-default macOS Seatbelt profile (`sandbox-exec`):

- **Network:** none (TCP, UDP, bind, Unix sockets and DNS all fail).
- **Processes:** no child creation, shell execution, other-process information access or system-service (mach) lookups. Re-execution within the interpreter install is allowed for framework Python startup and retains the sandbox.
- **Files:** reads limited to the interpreter install, system libraries and Rosetta; writes only to the scratch directory, which is deleted afterwards. File names and metadata remain visible.
- **Environment:** only `HOME`, `TMPDIR` (both the scratch directory), `PATH`, `LANG` and `PYTHONHASHSEED`, plus `__CF_USER_TEXT_ENCODING`, which macOS adds for framework Python builds.
- **Limits** (engine params): wall time `eval_timeout_seconds`, CPU time `eval_cpu_seconds` (rlimit), memory `eval_memory_mb` (a footprint watchdog targeting a 10 ms interval), 64 MiB per file, 64 open files, no core dumps. Memory is not a hard allocation cap: scheduling can delay enforcement beyond one interval. Unavailable memory telemetry stops the candidate.
- **Cancellation:** an independent guardian detects evaluator death, kills and reaps the candidate, and removes its scratch files. Cleanup restores candidate-restricted directory permissions without following symlinks. This does not cover forcibly killing the guardian itself or host failure.

Where isolation is unavailable (non-macOS, missing `sandbox-exec`, or a failed self-test such as a refused nested sandbox), candidates are not run and preflight stops the engine. Tests exercise network/filesystem/process denial, resource exhaustion, synthetic cross-process secret access, failed telemetry, evaluator SIGKILL and scratch cleanup. Passing these probes is not proof against every possible escape.

## Live Crusoe runs

Paid runs need four things: a priced Crusoe model from `integrations/crusoe/__init__.py`, `--max-cost-usd` above 0, `--allow-paid`, and `CRUSOE_API_KEY` in the environment.

```sh
export CRUSOE_API_KEY=...   # never commit it
python3 -m search run --engine shinka --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B \
    --dimension 4 --seed 0 --iterations 20 --max-cost-usd 1.00 --allow-paid
```

- **Prices:** pinned in `integrations/crusoe/__init__.py` from Crusoe's pricing page (read 2026-10-17). Unpriced models are rejected. Recheck the page before a run, because enforcement is only as accurate as that table.
- **Model IDs:** matched to the authenticated `/v1/models` response. Earlier documented aliases are normalized to the exact API IDs before configuration is saved or requests are sent.
- **Proxy and key:** ShinkaEvolve talks to `integrations/metering_proxy.py` on `127.0.0.1` with a per-run token. Only the proxy, which runs in the search process, holds `CRUSOE_API_KEY`. ShinkaEvolve, the evaluator and candidates never receive it.
- **Reservations:** before forwarding, the proxy reserves 1,048,576 input tokens (at least the full advertised context of every pinned model) plus the clamped `max_tokens` output ceiling. Only text messages and supported sampling parameters are accepted; tools, images, audio and extra token-bearing fields are rejected. Actual reported usage releases unused reservations. The default Nemotron model with 8192 output tokens needs about $0.335 of available reservation headroom per request.
- **Failures:** valid usage is accounted even on HTTP errors. Ambiguous post-send failures, oversized responses and missing usage retain the full reservation. Upstream error bodies are suppressed and known provider-secret occurrences are redacted from relayed content. A reservation overrun is recorded honestly and blocks future reservations across runs.
- **Caps:** `search/spend.py` limits reservations to the run budget and a **$50.00 shared ledger cap** in an fsynced, file-locked ledger at `~/.recursive-discovery/spend-ledger.jsonl`. All checkouts share it. Unsettled reservations keep counting; corrupt records, interrupted writes and cross-run settlements fail closed. Changing/deleting the ledger or making calls outside this adapter is outside its accounting.
- **Completion and failure reports:** budget exhaustion stops forwarding and preserves the best verified candidate. A run with only failed provider requests is an error, even when the initial construction is valid. `manifest.json` contains successful-run cost and usage; `engine/spend.jsonl` records request accounting. `engine/spend_summary.json` is saved after outstanding requests finish, including ordinary driver failures. A crash before finalization can leave open reservations in the ledger; a missing manifest never proves zero spend.
- **Preflight:** checks the opt-in, key, ledger violations, lifetime headroom and budget for one full-context reservation before creating the run directory. Replaying a paid config also needs `--allow-paid`. An exported key belongs to that shell and its descendants; exporting it in another terminal does not configure the shell starting the run.

## Limits

- Isolation is macOS-only, untested against deliberate kernel or Seatbelt exploits, and leaves total scratch disk use bounded only by time and per-file size.
- Reservation bounds assume Crusoe bills at the pinned prices, within the advertised context ceiling, and that `max_tokens` bounds billed output (including reasoning). The local ledger is conservative accounting, not an unconditional provider-side billing cap. Crusoe billing is authoritative.
- Runs are seeded but not byte-replayable, because ShinkaEvolve's async scheduling and sampling are not deterministic. Manifests list only `config.json` as deterministic.
- Initial constructions (`--initial`) are not supported. ShinkaEvolve always starts from its seed program.
