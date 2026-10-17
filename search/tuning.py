"""Offline strategy tuning: evolve ILS settings, then test them on held-out seeds.

Protocol (fixed before any evaluation and saved with the results):
- Tuning seeds and held-out seeds are disjoint ranges. Settings are selected
  from tuning scores only; held-out scores never feed back into selection.
- Every ordinary evaluation uses the same neighbor-list scan budget, checked
  between iterations. This deterministic proxy excludes sampling and adjacency
  checks; equal CPU cost is not established.
- Evolution ((mu + lambda) over the parameter space, with lineage) and the
  random-search control get the same number of trials on the same tuning seeds,
  both starting from the default settings. The fixed baseline uses the default
  settings with no tuning. A stricter end-to-end control also runs the default
  settings, giving each held-out run an equal share of the larger nominal tuning scan count
  on top of the per-run budget.
- Every construction is checked with verification.c7; every unique evaluation's
  certificate and metadata are retained. Reused results are counted separately.

Changing these settings is parameter tuning, not recursive algorithm
improvement, and no measured gain is a valid outcome. Results are baselines
with no record claim.
"""

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

from search.domains.c7 import C7StrongPower, certificate_text
from search.engines.offline import OfflineILSEngine
from search.errors import EngineError
from search.rng import Rng
from strategies.mis_ils import IteratedLocalSearch
from verification.c7 import validate

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT = dict(OfflineILSEngine.DEFAULT_PARAMS)
SPACE = dict(OfflineILSEngine.PARAM_LIMITS)
METHODS = ("fixed", "evolution", "random_search")
ARMS = METHODS + ("fixed_end_to_end",)
PAIRS = (("evolution", "fixed"), ("random_search", "fixed"), ("evolution", "random_search"),
         ("evolution", "fixed_end_to_end"), ("random_search", "fixed_end_to_end"))
SOURCES = ("search/tuning.py", "strategies/mis_ils.py", "search/domains/c7.py",
           "search/engines/offline.py", "search/rng.py", "verification/c7.py")
MU = 4  # evolution parents; each generation proposes MU children


def protocol(dimension=4, seed=0, trials=24, tuning_seeds=8, heldout_seeds=20, work_budget=10_000_000):
    """The frozen experiment definition; raises ValueError for unsupported settings."""
    for name, value, low, high in (("trials", trials, MU + 1, 400), ("tuning_seeds", tuning_seeds, 1, 1000),
                                   ("heldout_seeds", heldout_seeds, 1, 1000),
                                   ("work_budget", work_budget, 1000, 10 ** 10), ("seed", seed, 0, 10 ** 6)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer from {low} to {high}.")
    C7StrongPower(dimension)  # validates the dimension
    base = seed * 10000
    return {
        "format": "c7-strategy-tuning/v1",
        "problem": "c7_strong_power",
        "dimension": dimension,
        "seed": seed,
        "work_budget_per_run": work_budget,
        "work_unit": "neighbor-list entries scanned by the ILS strategy",
        "work_limitations": ("Checked between iterations, with possible overshoot. Excludes sampling "
                             "and adjacency checks; equal CPU cost is not established."),
        "trials_per_tuned_method": trials,
        "tuning_seeds": [base + 1000 + i for i in range(tuning_seeds)],
        "heldout_seeds": [base + 5000 + i for i in range(heldout_seeds)],
        "default_settings": DEFAULT,
        "parameter_space": {key: list(bounds) for key, bounds in SPACE.items()},
        "evolution": {"mu": MU, "lambda": MU, "initial": "default settings plus random settings"},
        "random_search": {"initial": "default settings, then uniform random settings"},
        "selection": "highest total tuning score; ties go to the earlier trial",
        "end_to_end_control": ("default settings with a per-run budget of work_budget_per_run plus "
                               "ceil(max tuned-method nominal tuning scans / held-out seeds)"),
        "comparison": "paired held-out scores per seed: wins, ties, losses; two-sided exact sign test",
        "run_class": "baseline",
        "record_claim": False,
    }


def run_experiment(plan, out_dir, workers=None, progress=None):
    """Run the frozen plan, write all artifacts into out_dir, and return the summary."""
    out_dir = Path(out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError(f"Output directory must be new or empty: {out_dir}")
    (out_dir / "certificates").mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "protocol.json", plan)
    sources = _source_hashes(out_dir / "source_snapshot")
    started, clock = datetime.now(timezone.utc).isoformat(timespec="seconds"), time.perf_counter()
    cache = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        def evaluate(configs, seeds, keep=False, budget=plan["work_budget_per_run"]):
            def key(config, seed):
                return json.dumps(config, sort_keys=True), seed, budget
            todo = [key(c, s) for c in configs for s in seeds if key(c, s) not in cache]
            jobs = [(plan["dimension"], json.loads(k[0]), k[1], budget, keep) for k in dict.fromkeys(todo)]
            for job, result in zip(jobs, pool.map(_evaluate, jobs)):
                evaluation_id = hashlib.sha256(json.dumps(job[:4], sort_keys=True).encode()).hexdigest()
                directory = out_dir / "evaluations" / evaluation_id
                directory.mkdir(parents=True)
                certificate_path = directory / "certificate.json"
                certificate_path.write_text(certificate_text(result["certificate"]), encoding="utf-8")
                result["evaluation_id"] = evaluation_id
                result["certificate_path"] = certificate_path.relative_to(out_dir).as_posix()
                metadata = {k: v for k, v in result.items() if k != "certificate"}
                metadata.update(dimension=job[0], params=job[1], work_budget=job[3])
                _write_json(directory / "result.json", metadata)
                cache[key(job[1], job[2])] = result
            return [[cache[key(c, s)] for s in seeds] for c in configs]

        trials = _random_search(plan, evaluate) + _evolution(plan, evaluate)
        with open(out_dir / "trials.jsonl", "w", encoding="utf-8") as stream:
            for trial in trials:
                stream.write(json.dumps(trial, sort_keys=True) + "\n")
        chosen = {"fixed": {"params": DEFAULT, "trial": None}}
        for method in ("evolution", "random_search"):
            best = max((t for t in trials if t["method"] == method),
                       key=lambda t: (t["total_score"], -t["trial"]))
            chosen[method] = {"params": best["params"], "trial": best["trial"]}
        if progress:
            progress("held-out evaluation")
        heldout = {m: evaluate([chosen[m]["params"]], plan["heldout_seeds"], keep=True)[0] for m in METHODS}
        tuning_work = {m: sum(r["work"] for t in trials if t["method"] == m for r in t["tuning"])
                       for m in ("evolution", "random_search")}
        seeds = len(plan["heldout_seeds"])
        end_to_end_budget = plan["work_budget_per_run"] + -(-max(tuning_work.values()) // seeds)
        chosen["fixed_end_to_end"] = {"params": DEFAULT, "trial": None,
                                      "work_budget_per_run": end_to_end_budget}
        heldout["fixed_end_to_end"] = evaluate([DEFAULT], plan["heldout_seeds"], keep=True,
                                               budget=end_to_end_budget)[0]

    rows = []
    for index, seed in enumerate(plan["heldout_seeds"]):
        row = {"seed": seed}
        for arm in ARMS:
            result = heldout[arm][index]
            row[arm] = result["size"]
            path = out_dir / "certificates" / f"heldout-{arm}-seed{seed}.json"
            path.write_text(certificate_text(result["certificate"]), encoding="utf-8")
        rows.append(row)
    references = {arm: [{key: r[key] for key in ("seed", "evaluation_id", "certificate_path")}
                        for r in heldout[arm]] for arm in ARMS}
    _write_json(out_dir / "heldout.json", {"selected": chosen, "rows": rows, "evaluations": references})

    all_results = [r for t in trials for r in t["tuning"]] + [r for arm in ARMS for r in heldout[arm]]
    if _source_hashes() != sources:
        raise EngineError("Tuning source files changed during the run; results are not accepted.")
    summary = {
        "format": "c7-strategy-tuning-summary/v1",
        "run_class": "baseline",
        "record_claim": False,
        "interpretation": ("Parameter tuning of a fixed ILS strategy at matched neighbor-list scan "
                           "budgets, with a separate expanded-budget default control. Equal CPU cost "
                           "is not established; recursive algorithm improvement is not demonstrated."),
        "dimension": plan["dimension"],
        "selected": chosen,
        "heldout_mean": {arm: _mean([row[arm] for row in rows]) for arm in ARMS},
        "heldout_best": {arm: max(row[arm] for row in rows) for arm in ARMS},
        "comparisons": {f"{a}_vs_{b}": _compare([row[a] for row in rows], [row[b] for row in rows])
                        for a, b in PAIRS},
        "compute": {
            "work_budget_per_run": plan["work_budget_per_run"],
            "end_to_end_budget_per_run": end_to_end_budget,
            "tuning_work": tuning_work,
            "tuning_runs": {m: sum(len(t["tuning"]) for t in trials if t["method"] == m)
                            for m in ("evolution", "random_search")},
            "heldout_work": {arm: sum(r["work"] for r in heldout[arm]) for arm in ARMS},
            "executed_work": sum(r["work"] for r in cache.values()),
            "note": ("Per-method totals are nominal scans, including reused evaluations; executed_work "
                     "counts cache misses once. fixed_end_to_end receives the larger nominal tuning "
                     "scan count in addition to its ordinary held-out budgets. These are not CPU costs."),
        },
        "verification": {"checker": "verification.c7.validate (in-process)", "certificates": len(cache),
                         "executed_evaluations": len(cache), "logical_results": len(all_results),
                         "all_valid": all(r["valid"] for r in all_results)},
        "source_sha256": sources,
        "sources_unchanged_during_run": True,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "workers": workers or os.cpu_count()},
        "nondeterministic": {"started_at_utc": started,
                             "wall_seconds": round(time.perf_counter() - clock, 1)},
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def _random_search(plan, evaluate):
    rng = Rng(plan["seed"]).child(1)
    configs = [dict(DEFAULT)] + [_random_config(rng) for _ in range(plan["trials_per_tuned_method"] - 1)]
    scores = evaluate(configs, plan["tuning_seeds"])
    return [_trial("random_search", index, config, results, None, [])
            for index, (config, results) in enumerate(zip(configs, scores))]


def _evolution(plan, evaluate):
    rng = Rng(plan["seed"]).child(2)
    total = plan["trials_per_tuned_method"]
    initial = [dict(DEFAULT)] + [_random_config(rng) for _ in range(MU - 1)]
    scores = evaluate(initial, plan["tuning_seeds"])
    trials = [_trial("evolution", index, config, results, None, [])
              for index, (config, results) in enumerate(zip(initial, scores))]
    while len(trials) < total:
        parents = sorted(trials, key=lambda t: (-t["total_score"], t["trial"]))[:MU]
        children, lineage = [], []
        for _ in range(min(MU, total - len(trials))):
            parent = parents[rng.randbelow(len(parents))]
            child, changed = _mutate(parent["params"], rng)
            children.append(child)
            lineage.append((parent["trial"], changed))
        scores = evaluate(children, plan["tuning_seeds"])
        for (parent, changed), config, results in zip(lineage, children, scores):
            trials.append(_trial("evolution", len(trials), config, results, parent, changed))
    return trials


def _random_config(rng):
    return {key: low + rng.randbelow(high - low + 1) for key, (low, high) in sorted(SPACE.items())}


def _mutate(params, rng):
    """Change one or more settings by a bounded random step; return (child, changed keys)."""
    child, keys = dict(params), sorted(SPACE)
    count = 1
    while count < len(keys) and rng.random() < 0.5:
        count += 1
    changed = []
    while len(changed) < count:
        key = keys[rng.randbelow(len(keys))]
        if key in changed:
            continue
        low, high = SPACE[key]
        step = 1 + rng.randbelow(max(1, (high - low) // 8))
        value = child[key] + (step if rng.random() < 0.5 else -step)
        if not low <= value <= high:
            value = child[key] - (value - child[key])  # reflect off the bound
        value = min(high, max(low, value))
        if value != child[key]:
            child[key] = value
            changed.append(key)
    return child, sorted(changed)


def _trial(method, index, params, results, parent, changed):
    return {"method": method, "trial": index, "params": params, "parent": parent, "changed": changed,
            "total_score": sum(r["size"] for r in results), "mean_score": _mean([r["size"] for r in results]),
            "tuning": [{key: r[key] for key in ("seed", "size", "work", "iterations", "valid",
                                               "evaluation_id", "certificate_path")}
                       for r in results]}


_TASKS = {}


def _evaluate(job):
    """One ILS run at the work budget; returns its verified size (picklable for worker processes)."""
    dimension, params, seed, work_budget, _keep = job
    if dimension not in _TASKS:
        _TASKS[dimension] = C7StrongPower(dimension)
    task = _TASKS[dimension]
    result = IteratedLocalSearch(task, Rng(seed), **params).run(None, work_budget=work_budget)
    certificate = task.certificate(result["best"])
    verdict = validate(certificate)
    if not verdict["valid"] or verdict["size"] != result["best_size"]:
        raise EngineError(f"Tuning evaluation failed verification (dimension={dimension}, seed={seed}).")
    return {"seed": seed, "size": verdict["size"], "work": result["work"],
            "iterations": result["iterations"],
            "valid": True, "certificate": certificate}


def _compare(a, b):
    wins = sum(x > y for x, y in zip(a, b))
    losses = sum(x < y for x, y in zip(a, b))
    n, k = wins + losses, min(wins, losses)
    p_value = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"wins": wins, "ties": len(a) - wins - losses, "losses": losses,
            "mean_difference": _mean([x - y for x, y in zip(a, b)]), "sign_test_p": round(p_value, 6)}


def _source_hashes(snapshot_dir=None):
    hashes = {}
    for path in SOURCES:
        source = (REPO_ROOT / path).read_bytes()
        hashes[path] = hashlib.sha256(source).hexdigest()
        if snapshot_dir is not None:
            target = Path(snapshot_dir) / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source)
    return hashes


def _mean(values):
    return round(sum(values) / len(values), 4) if values else None


def _write_json(path, value):
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
