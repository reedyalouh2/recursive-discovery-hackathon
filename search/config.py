"""Search configuration: strict validation, canonical JSON and hashing."""

import hashlib
import json
import math
from pathlib import Path

from search.domains.c7 import MAX_SEARCH_DIMENSION, PROBLEM
from search.engines import ENGINES
from search.spend import LIFETIME_CAP_USD

CONFIG_FORMAT = "c7-search-config/v1"
RUN_CLASSES = ("calibration", "baseline")
INPUT_NAME = "input.json"
MAX_SEED = 2 ** 64 - 1
MAX_ITERATIONS = 10_000_000
MAX_RESTARTS = 1_000
MAX_TOTAL_ITERATIONS = 50_000_000
MAX_CONFIG_BYTES = 64 * 1024


class ConfigError(ValueError):
    """Invalid or unsupported search configuration or input."""


def make_config(dimension, seed, iterations, restarts=1, run_class="baseline",
                engine="offline-ils", params=None, initial_sha256=None, max_cost_usd=0):
    raw = {
        "format": CONFIG_FORMAT,
        "problem": PROBLEM,
        "dimension": dimension,
        "seed": seed,
        "run_class": run_class,
        "engine": {"name": engine, "params": {} if params is None else params},
        "budget": {"iterations": iterations, "restarts": restarts, "max_cost_usd": max_cost_usd},
        "initial_construction": None if initial_sha256 is None
        else {"path": INPUT_NAME, "sha256": initial_sha256},
    }
    return validate_config(raw)


def validate_config(raw):
    """Return a normalized copy of `raw` or raise ConfigError."""
    _require_keys(raw, "config", ("format", "problem", "dimension", "seed", "run_class",
                                  "engine", "budget", "initial_construction"))
    if raw["format"] != CONFIG_FORMAT:
        raise ConfigError(f"format must be {CONFIG_FORMAT!r}.")
    if raw["problem"] != PROBLEM:
        raise ConfigError(f"problem must be {PROBLEM!r}.")
    dimension = _integer(raw["dimension"], "dimension", 1, MAX_SEARCH_DIMENSION)
    seed = _integer(raw["seed"], "seed", 0, MAX_SEED)
    if raw["run_class"] not in RUN_CLASSES:
        raise ConfigError("run_class must be 'calibration' or 'baseline'; "
                          "search output cannot classify itself as a record.")

    engine = raw["engine"]
    _require_keys(engine, "engine", ("name", "params"))
    if not isinstance(engine["name"], str) or engine["name"] not in ENGINES:
        raise ConfigError(f"Unknown engine {engine['name']!r}; available: {', '.join(sorted(ENGINES))}.")
    engine_class = ENGINES[engine["name"]]
    try:
        params = engine_class.validate_params(engine["params"])
    except ValueError as error:
        raise ConfigError(str(error)) from None

    budget = raw["budget"]
    _require_keys(budget, "budget", ("iterations", "restarts", "max_cost_usd"))
    iterations = _integer(budget["iterations"], "budget.iterations", 0, MAX_ITERATIONS)
    restarts = _integer(budget["restarts"], "budget.restarts", 1, MAX_RESTARTS)
    if iterations * restarts > MAX_TOTAL_ITERATIONS:
        raise ConfigError(f"budget.iterations * budget.restarts must not exceed {MAX_TOTAL_ITERATIONS}.")
    cost = budget["max_cost_usd"]
    if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
        raise ConfigError("budget.max_cost_usd must be a finite, non-negative number.")
    if getattr(engine_class, "is_paid", lambda params: False)(params):
        if not 0 < cost <= LIFETIME_CAP_USD:
            raise ConfigError(f"A paid model needs budget.max_cost_usd above 0 and at most "
                              f"${LIFETIME_CAP_USD}, the lifetime cap.")
    elif cost != 0:
        raise ConfigError("budget.max_cost_usd must be 0 for an unpaid engine or model.")
    if hasattr(engine_class, "validate_budget"):
        try:
            engine_class.validate_budget({"iterations": iterations, "restarts": restarts})
        except ValueError as error:
            raise ConfigError(str(error)) from None

    initial = raw["initial_construction"]
    if initial is not None:
        _require_keys(initial, "initial_construction", ("path", "sha256"))
        if initial["path"] != INPUT_NAME:
            raise ConfigError(f"initial_construction.path must be {INPUT_NAME!r}.")
        digest = initial["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or digest.strip("0123456789abcdef"):
            raise ConfigError("initial_construction.sha256 must be 64 lowercase hex digits.")
        initial = {"path": INPUT_NAME, "sha256": digest}
        if not getattr(engine_class, "supports_initial_construction", True):
            raise ConfigError(f"Engine {engine['name']!r} does not accept an initial construction.")

    return {
        "format": CONFIG_FORMAT,
        "problem": PROBLEM,
        "dimension": dimension,
        "seed": seed,
        "run_class": raw["run_class"],
        "engine": {"name": engine["name"], "params": params},
        "budget": {"iterations": iterations, "restarts": restarts, "max_cost_usd": float(cost)},
        "initial_construction": initial,
    }


def load_config(path):
    """Read and validate a saved config.json; return (config, directory)."""
    source = Path(path)
    try:
        with source.open("rb") as stream:
            raw = stream.read(MAX_CONFIG_BYTES + 1)
    except OSError as error:
        raise ConfigError(f"Cannot read config: {error}") from None
    if len(raw) > MAX_CONFIG_BYTES:
        raise ConfigError(f"Config file exceeds {MAX_CONFIG_BYTES} bytes.")
    return validate_config(parse_json(raw)), source.resolve().parent


def parse_json(raw):
    """Strict JSON: UTF-8, no duplicate keys, no NaN or Infinity."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ConfigError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value):
        raise ConfigError(f"Non-JSON numeric constant: {value}")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=reject)
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        if isinstance(error, ConfigError):
            raise
        raise ConfigError(f"Invalid JSON: {error}") from None


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def config_sha256(config):
    return sha256_hex(canonical_json(config).encode("utf-8"))


def _require_keys(value, name, keys):
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a JSON object.")
    missing = [key for key in keys if key not in value]
    unknown = sorted(set(value) - set(keys))
    if missing:
        raise ConfigError(f"{name} is missing: {', '.join(missing)}.")
    if unknown:
        raise ConfigError(f"{name} has unknown keys: {', '.join(unknown)}.")


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ConfigError(f"{name} must be an integer from {low} to {high}.")
    return value
