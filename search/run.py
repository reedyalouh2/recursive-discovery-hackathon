"""One bounded run: config -> engine -> certificate -> official checker -> manifest.

A run directory holds config.json, input.json (if seeded), history.jsonl,
certificate.json (candidate format v1), verification.json and manifest.json.
With a deterministic engine, config.json, certificate.json and history.jsonl
replay byte for byte from config.json and its input.json. Engines may keep
their own artifacts under engine/.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

from search import official
from search.config import (INPUT_NAME, ConfigError, config_sha256, load_config, make_config,
                           parse_json, sha256_hex)
from search.domains.c7 import C7StrongPower, certificate_text
from search.engines import ENGINES
from search.history import History
from search.rng import Rng

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
MAX_INPUT_BYTES = 2 * 1024 * 1024  # the starter checker's file limit
DETERMINISTIC_ARTIFACTS = ("config.json", "certificate.json", "history.jsonl")
CLAIM_NOTES = {
    "calibration": "Calibration of the search pipeline; not a research result.",
    "baseline": "Search baseline; novelty not assessed; no record claim.",
}
REPLAY_NOTES = {
    True: "python3 -m search run --config RUN_DIR/config.json --out NEW_DIR",
    False: "Re-running the config repeats the search, but this engine is not deterministic.",
}
SMOKE_SETTINGS = {"dimension": 3, "seed": 0, "iterations": 200, "restarts": 2,
                  "run_class": "calibration"}


def read_input(path):
    """Bytes of an initial construction file, within the checker's size limit."""
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(MAX_INPUT_BYTES + 1)
    except OSError as error:
        raise ConfigError(f"Cannot read initial construction: {error}") from None
    if len(raw) > MAX_INPUT_BYTES:
        raise ConfigError(f"Initial construction exceeds {MAX_INPUT_BYTES} bytes.")
    return raw


def execute(config, out_dir, initial_source=None, allow_paid=False):
    """Run a validated config into a new directory; return the manifest.

    allow_paid must be True for engines that would spend money.
    """
    task = C7StrongPower(config["dimension"])
    initial_raw, initial = None, []
    if config["initial_construction"] is not None:
        if initial_source is None:
            raise ConfigError("The config names an initial construction but no file was given.")
        initial_raw = read_input(initial_source)
        if sha256_hex(initial_raw) != config["initial_construction"]["sha256"]:
            raise ConfigError("Initial construction does not match the sha256 in the config.")
        verdict = official.check(initial_source)
        if not verdict["valid"]:
            reason = (verdict["result"] or {}).get("reason", "no structured output")
            raise ConfigError(f"Initial construction rejected by {official.CHECKER}: {reason}")
        try:
            initial = task.vertices_from_certificate(parse_json(initial_raw))
        except ValueError as error:
            raise ConfigError(str(error)) from None

    engine = ENGINES[config["engine"]["name"]](config["engine"]["params"])
    engine.allow_paid = allow_paid
    if hasattr(engine, "preflight"):
        engine.preflight(budget=config["budget"], allow_paid=allow_paid)
    out_dir = Path(out_dir)
    _create_empty_dir(out_dir)
    digest = config_sha256(config)
    _write(out_dir / "config.json", json.dumps(config, indent=2, sort_keys=True) + "\n")
    if initial_raw is not None:
        (out_dir / INPUT_NAME).write_bytes(initial_raw)

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    clock = time.perf_counter()
    with History(out_dir / "history.jsonl") as history:
        history.write("run_start", config_sha256=digest, problem=config["problem"],
                      dimension=config["dimension"], seed=config["seed"],
                      run_class=config["run_class"], engine=engine.identity(),
                      budget=config["budget"], initial_size=len(initial))
        result = engine.run(task, config["budget"], Rng(config["seed"]), history, initial,
                            workspace=out_dir / "engine")
        vertices = result["vertices"]
        internal_check = task.is_independent(vertices)
        history.write("run_end", search_reported_size=len(vertices),
                      best_restart=result["best_restart"], iterations=result["iterations"],
                      internal_check=internal_check)

    _write(out_dir / "certificate.json", certificate_text(task.certificate(vertices)))
    verdict = official.check(out_dir / "certificate.json")
    _write(out_dir / "verification.json", json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    wall_seconds = time.perf_counter() - clock

    reported = verdict["result"] if isinstance(verdict["result"], dict) else {}
    ok = verdict["valid"] and internal_check and reported.get("size") == len(vertices)
    deterministic = getattr(engine, "deterministic", True)
    names = list(DETERMINISTIC_ARTIFACTS) + ["verification.json"]
    if initial_raw is not None:
        names.append(INPUT_NAME)
    manifest = {
        "format": "c7-search-manifest/v1",
        "ok": ok,
        "run_class": config["run_class"],
        "record_claim": False,
        "claim_note": CLAIM_NOTES[config["run_class"]],
        "problem": config["problem"],
        "dimension": config["dimension"],
        "seed": config["seed"],
        "config_sha256": digest,
        "engine": engine.identity(),
        "engine_runtime": result.get("runtime"),
        "budget": config["budget"],
        "cost": result["cost"],
        "spend": result.get("spend"),
        "compute": {"iterations": result["iterations"], "restarts": config["budget"]["restarts"]},
        "initial_construction": None if initial_raw is None
        else dict(config["initial_construction"], size=len(initial)),
        "search_reported_size": len(vertices),
        "internal_check": internal_check,
        "official": {
            "checker": verdict["checker"],
            "exit_code": verdict["exit_code"],
            "valid": verdict["valid"],
            "size": reported.get("size"),
            "dimension": reported.get("dimension"),
            "capacity_lower_bound_exact": reported.get("capacity_lower_bound_exact"),
            "reason": reported.get("reason"),
        },
        "artifacts": {name: sha256_hex((out_dir / name).read_bytes()) for name in sorted(names)},
        "deterministic_artifacts": list(DETERMINISTIC_ARTIFACTS) if deterministic else ["config.json"],
        "replay": REPLAY_NOTES[deterministic],
        "environment": {"python": platform.python_version(),
                        "implementation": platform.python_implementation(),
                        "platform": platform.platform()},
        "nondeterministic": {"started_at_utc": started_at, "wall_seconds": round(wall_seconds, 3)},
    }
    _write(out_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def smoke(out_dir=None):
    """Calibration run, replay from its saved config, and a negative control."""
    root = Path(out_dir) if out_dir is not None else unique_dir(RUNS_DIR / "smoke")
    _create_empty_dir(root)
    first = execute(make_config(**SMOKE_SETTINGS), root / "run")
    config, _ = load_config(root / "run" / "config.json")
    second = execute(config, root / "replay")
    identical = all(first["artifacts"][name] == second["artifacts"][name]
                    for name in DETERMINISTIC_ARTIFACTS)
    negative = negative_control(root / "run" / "certificate.json", root / "negative_control.json")
    summary = {
        "command": "smoke",
        "status": "passed" if first["ok"] and second["ok"] and identical and negative["rejected"]
        else "failed",
        "run_class": first["run_class"],
        "record_claim": False,
        "dimension": first["dimension"],
        "size": first["official"]["size"],
        "capacity_lower_bound_exact": first["official"]["capacity_lower_bound_exact"],
        "replay_identical": identical,
        "negative_control_rejected": negative["rejected"],
        "negative_control_reason": negative["reason"],
        "run_dir": str(root.resolve() / "run"),
        "replay_dir": str(root.resolve() / "replay"),
    }
    _write(root / "smoke.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def negative_control(certificate_path, out_path):
    """Add a vertex adjacent to the first one and confirm the checker rejects the copy."""
    certificate = json.loads(Path(certificate_path).read_text(encoding="utf-8"))
    vertices = certificate["vertices"] or [[0] * certificate["dimension"]]
    twin = list(vertices[0])
    twin[-1] = (twin[-1] + 1) % 7
    certificate["vertices"] = vertices + [twin]
    _write(out_path, certificate_text(certificate))
    verdict = official.check(out_path)
    result = verdict["result"] if isinstance(verdict["result"], dict) else {}
    return {
        "rejected": (not verdict["valid"] and verdict["exit_code"] == 1
                     and result.get("valid") is False),
        "reason": result.get("reason"),
    }


def default_run_dir(config):
    name = f"c7-d{config['dimension']}-s{config['seed']}-{config_sha256(config)[:8]}"
    return unique_dir(RUNS_DIR / name)


def unique_dir(base):
    candidate, suffix = base, 2
    while candidate.exists():
        candidate = base.with_name(f"{base.name}-{suffix}")
        suffix += 1
    return candidate


def _create_empty_dir(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ConfigError(f"Output directory must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
