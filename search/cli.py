"""Command line for bounded construction search.

  python3 -m search run --dimension 3 --seed 0 --iterations 500 [--restarts 2]
                        [--engine offline-ils|shinka] [--initial CERT.json]
                        [--run-class baseline] [--out DIR]
  python3 -m search run --engine shinka --model MODEL --max-cost-usd 1.00 --allow-paid ...
  python3 -m search run --config runs/RUN/config.json [--allow-paid] [--out DIR]   # replay
  python3 -m search smoke [--out DIR]
  python3 -m search tune [--dimension 4] [--trials 24] [--tuning-seeds 8] [--heldout-seeds 20]
                         [--work-budget 10000000] [--workers N] [--out DIR]

Prints one JSON line. Exit 0: verified (or smoke passed); 1: rejected, smoke
failed or engine failure; 2: invalid input or configuration.
"""

import argparse
import json
from pathlib import Path

from search.config import INPUT_NAME, RUN_CLASSES, ConfigError, load_config, make_config, sha256_hex
from search.engines import ENGINES
from search.errors import EngineError
from search import tuning
from search.run import RUNS_DIR, default_run_dir, execute, read_input, smoke, unique_dir

SEARCH_FLAGS = ("dimension", "seed", "iterations", "restarts", "engine", "model", "max_cost_usd",
                "initial", "run_class")


def _parser():
    parser = argparse.ArgumentParser(prog="python3 -m search",
                                     description="Bounded, seeded C7 strong-power construction search.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run one bounded search and verify its certificate.")
    run.add_argument("--config", type=Path, help="Saved config.json to replay; excludes search flags.")
    run.add_argument("--dimension", type=int)
    run.add_argument("--seed", type=int)
    run.add_argument("--iterations", type=int,
                     help="Per restart: ILS perturbation steps, or ShinkaEvolve generations.")
    run.add_argument("--restarts", type=int, help="Independent restarts (default 1).")
    run.add_argument("--engine", choices=sorted(ENGINES), help="Default: offline-ils.")
    run.add_argument("--model", help="shinka only: offline-stub-v1 (default) or a priced Crusoe model.")
    run.add_argument("--max-cost-usd", type=float, help="Run budget for a paid model (default 0).")
    run.add_argument("--allow-paid", action="store_true",
                     help="Required for any run that would spend money, including replays.")
    run.add_argument("--initial", type=Path, help="Certificate to start from; copied into the run.")
    run.add_argument("--run-class", choices=RUN_CLASSES, help="Default: baseline.")
    run.add_argument("--out", type=Path, help="New or empty output directory (default under runs/).")
    smoke_parser = commands.add_parser("smoke", help="Calibration run, replay and negative control.")
    smoke_parser.add_argument("--out", type=Path, help="New or empty output directory.")
    tune = commands.add_parser("tune", help="Tune ILS settings; compare held-out seeds at matched scan budgets.")
    tune.add_argument("--dimension", type=int, default=4)
    tune.add_argument("--seed", type=int, default=0)
    tune.add_argument("--trials", type=int, default=24, help="Trials per tuned method.")
    tune.add_argument("--tuning-seeds", type=int, default=8)
    tune.add_argument("--heldout-seeds", type=int, default=20)
    tune.add_argument("--work-budget", type=int, default=10_000_000, help="Neighbor-list scans per run.")
    tune.add_argument("--workers", type=int, help="Parallel worker processes (default: all cores).")
    tune.add_argument("--out", type=Path, help="New or empty output directory (default under runs/).")
    return parser


def _tune(args):
    try:
        plan = tuning.protocol(args.dimension, args.seed, args.trials, args.tuning_seeds,
                               args.heldout_seeds, args.work_budget)
    except ValueError as error:
        raise ConfigError(str(error)) from None
    out_dir = args.out
    if out_dir is None:
        out_dir = unique_dir(RUNS_DIR / f"tuning-d{args.dimension}-s{args.seed}")
    try:
        result = tuning.run_experiment(plan, out_dir, workers=args.workers)
    except ValueError as error:
        raise ConfigError(str(error)) from None
    summary = {"command": "tune", "status": "completed" if result["verification"]["all_valid"] else "invalid",
               "run_class": "baseline", "record_claim": False, "out_dir": str(Path(out_dir).resolve()),
               "heldout_mean": result["heldout_mean"], "comparisons": result["comparisons"],
               "selected": {m: result["selected"][m]["params"] for m in tuning.METHODS}}
    return summary, 0 if result["verification"]["all_valid"] else 1


def _run(args):
    if args.config is not None:
        given = ["--" + name.replace("_", "-") for name in SEARCH_FLAGS if getattr(args, name) is not None]
        if given:
            raise ConfigError(f"--config cannot be combined with {', '.join(given)}.")
        config, directory = load_config(args.config)
        initial_source = directory / INPUT_NAME if config["initial_construction"] else None
    else:
        missing = ["--" + name for name in ("dimension", "seed", "iterations") if getattr(args, name) is None]
        if missing:
            raise ConfigError(f"Missing {', '.join(missing)} (or pass --config).")
        initial_source = args.initial
        digest = None if initial_source is None else sha256_hex(read_input(initial_source))
        config = make_config(args.dimension, args.seed, args.iterations,
                             1 if args.restarts is None else args.restarts,
                             args.run_class or "baseline", engine=args.engine or "offline-ils",
                             params=None if args.model is None else {"model": args.model},
                             initial_sha256=digest,
                             max_cost_usd=0 if args.max_cost_usd is None else args.max_cost_usd)
    out_dir = args.out if args.out is not None else default_run_dir(config)
    manifest = execute(config, out_dir, initial_source, allow_paid=args.allow_paid)
    summary = {
        "command": "run",
        "status": "verified" if manifest["ok"] else "rejected",
        "run_class": manifest["run_class"],
        "record_claim": False,
        "dimension": manifest["dimension"],
        "size": manifest["official"]["size"],
        "capacity_lower_bound_exact": manifest["official"]["capacity_lower_bound_exact"],
        "search_reported_size": manifest["search_reported_size"],
        "cost_usd": manifest["cost"]["usd"],
        "run_dir": str(Path(out_dir).resolve()),
    }
    return summary, 0 if manifest["ok"] else 1


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            summary, code = _run(args)
        elif args.command == "tune":
            summary, code = _tune(args)
        else:
            summary = smoke(args.out)
            code = 0 if summary["status"] == "passed" else 1
    except ConfigError as error:
        summary, code = {"command": args.command, "status": "error", "reason": str(error)}, 2
    except EngineError as error:
        summary, code = {"command": args.command, "status": "error", "reason": str(error)}, 1
    print(json.dumps(summary, sort_keys=True))
    return code
