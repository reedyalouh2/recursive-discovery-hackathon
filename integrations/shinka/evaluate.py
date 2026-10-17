"""ShinkaEvolve evaluator: run one candidate in isolation, then score it with verification.c7.

ShinkaEvolve calls
  PYTHON evaluate.py --program_path gen_N/main.py --results_dir gen_N/results \
      --dimension D --timeout SECONDS [--cpu_seconds S] [--memory_mb M]
The candidate runs through search.isolation: no network, no process spawning,
writes only to a private scratch directory, and wall/CPU/memory/file-size
limits. Its output.json is copied to results/certificate.json (data), then
this script writes metrics.json and correct.json. combined_score is the size
reported by the official checker, or 0 when anything fails. Where isolation
is unavailable, candidates are not run at all.
"""

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from search import isolation, official  # noqa: E402

MAX_FEEDBACK_CHARS = 2000
MAX_CERTIFICATE_BYTES = 2 * 1024 * 1024  # the starter checker's file limit


def evaluate(program_path, results_dir, dimension, timeout, cpu_seconds=None, memory_mb=1024):
    """Return (metrics, correct, error) for one candidate program."""
    results = Path(results_dir).resolve()
    results.mkdir(parents=True, exist_ok=True)
    certificate = results / "certificate.json"
    if certificate.exists():
        certificate.unlink()
    cpu_seconds = timeout if cpu_seconds is None else cpu_seconds
    limits = isolation.Limits(wall_seconds=timeout, cpu_seconds=max(1, int(cpu_seconds)),
                              memory_bytes=int(memory_mb * 2 ** 20))
    try:
        program = Path(program_path).read_text(encoding="utf-8")
        run = isolation.run_candidate(program, ["--dimension", dimension, "--output", "output.json"],
                                      limits, max_output_bytes=MAX_CERTIFICATE_BYTES)
    except isolation.IsolationUnavailable as error:
        return _failure(f"Candidate not run: isolation unavailable: {error}")
    except (OSError, UnicodeDecodeError, ValueError) as error:
        return _failure(f"Candidate not run: {error}")
    usage = {key: run[key] for key in ("wall_seconds", "cpu_seconds", "peak_footprint_bytes")}
    failure = _run_failure(run, limits)
    if failure:
        return _failure(failure, usage)
    if run["output"] is None:
        return _failure("Candidate wrote no certificate (or one over the size limit).", usage)
    certificate.write_bytes(run["output"])
    verdict = official.check(certificate)
    result = verdict["result"] if isinstance(verdict["result"], dict) else {}
    if not verdict["valid"]:
        return _failure(f"verification.c7 rejected the output: {result.get('reason')}", usage)
    if result.get("dimension") != dimension:
        return _failure(f"Certificate dimension {result.get('dimension')} "
                        f"does not match requested dimension {dimension}.", usage)
    metrics = {
        "combined_score": float(result["size"]),
        "public": {"size": result["size"],
                   "capacity_lower_bound": result.get("capacity_lower_bound_exact")},
        "private": dict(usage, checker=official.CHECKER, checker_exit_code=verdict["exit_code"]),
        "text_feedback": result.get("reason", ""),
    }
    return metrics, True, None


def _run_failure(run, limits):
    status = run["status"]
    if status == "timeout":
        return f"Candidate exceeded the {limits.wall_seconds:g} s time limit."
    if status == "cpu":
        return f"Candidate exceeded the {limits.cpu_seconds} s CPU limit."
    if status == "memory":
        return f"Candidate exceeded the {limits.memory_bytes // 2 ** 20} MiB memory limit."
    if status == "monitor":
        return "Candidate stopped because memory monitoring became unavailable."
    if status == "cancelled":
        return "Candidate stopped because its evaluator exited."
    if status == "signal":
        return f"Candidate was killed by signal {run['signal']}."
    if status == "exit":
        return (f"Candidate exited with code {run['returncode']}: "
                f"{run['stderr_tail'][-MAX_FEEDBACK_CHARS:]}")
    if status == "ok":
        return None
    return f"Candidate stopped with unrecognized supervisor status: {status}."


def _failure(reason, usage=None):
    reason = reason[:MAX_FEEDBACK_CHARS]
    metrics = {"combined_score": 0.0, "public": {"size": 0}, "private": usage or {},
               "text_feedback": reason}
    return metrics, False, reason


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--cpu_seconds", type=int)
    parser.add_argument("--memory_mb", type=int, default=1024)
    args = parser.parse_args(argv)
    metrics, correct, error = evaluate(args.program_path, args.results_dir, args.dimension, args.timeout,
                                       args.cpu_seconds, args.memory_mb)
    results = Path(args.results_dir)
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (results / "correct.json").write_text(json.dumps({"correct": correct, "error": error}, indent=2),
                                          encoding="utf-8")
    print(json.dumps({"correct": correct, "combined_score": metrics["combined_score"], "error": error}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
