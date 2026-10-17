"""ShinkaEvolve engine adapter: offline stub model, or budgeted Crusoe models.

Runs the pinned ShinkaEvolve release in its own Python environment through
driver.py, one ShinkaEvolve run per restart. budget.iterations is ShinkaEvolve's
num_generations, which counts the seed program as generation 0. Candidate
certificates are data: the best one is re-checked with verification.c7 here,
and the run pipeline checks the exported certificate again. Runs are seeded but
not byte-replayable, because ShinkaEvolve's async scheduling and sampling are not
deterministic. Candidates run under search.isolation (no network, no process
spawning, scratch-only writes, wall/CPU/memory/file limits); preflight refuses
to start where that isolation is unavailable.

Paid models (any priced Crusoe model) need budget.max_cost_usd > 0, an explicit
allow_paid opt-in and CRUSOE_API_KEY. ShinkaEvolve then talks to a local
metering proxy in this process, which alone holds the key and enforces both the
run budget and the lifetime cap in search.spend before each request.
"""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from integrations import crusoe
from integrations.metering_proxy import MeteringProxy, token_cost
from integrations.shinka import SHINKA_GIT_COMMIT, SHINKA_VERSION
from integrations.shinka.stub_model import MODEL_NAME
from search import isolation, official
from search.errors import EngineError
from search.spend import LIFETIME_CAP_USD, LedgerError, SpendLedger, usd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "shinka" / "bin" / "python"
VERIFIER_SOURCE = REPO_ROOT / "verification" / "c7.py"
SETUP_HINT = ("Create it with: uv venv --python 3.12 .venv/shinka && uv pip install "
              "--python .venv/shinka/bin/python --require-hashes --no-deps "
              "-r integrations/shinka/requirements.lock")
# Model API keys are deliberately not passed through; ShinkaEvolve only gets a proxy token.
PASSTHROUGH_ENV = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT")
PROXY_TOKEN_ENV = "DISCOVERY_PROXY_TOKEN"


class ShinkaEngine:
    name = "shinka"
    version = SHINKA_VERSION
    provider = "offline-stub"
    deterministic = False
    supports_initial_construction = False
    DEFAULT_PARAMS = {"model": MODEL_NAME, "eval_timeout_seconds": 60, "eval_cpu_seconds": 60,
                      "eval_memory_mb": 1024, "num_islands": 1, "max_output_tokens": 8192}
    PARAM_LIMITS = {"eval_timeout_seconds": (1, 600), "eval_cpu_seconds": (1, 600),
                    "eval_memory_mb": (64, 16384), "num_islands": (1, 8),
                    "max_output_tokens": (256, 32768)}
    MAX_GENERATIONS = 500
    MAX_RESTARTS = 10

    def __init__(self, params):
        self.params = self.validate_params(params)
        self.allow_paid = False

    @classmethod
    def validate_params(cls, params):
        """Return params with defaults filled in, or raise ValueError."""
        if not isinstance(params, dict):
            raise ValueError("engine.params must be a JSON object.")
        unknown = sorted(set(params) - set(cls.DEFAULT_PARAMS))
        if unknown:
            raise ValueError(f"Unknown engine.params for {cls.name}: {', '.join(unknown)}.")
        result = dict(cls.DEFAULT_PARAMS)
        for key, value in params.items():
            if key == "model":
                value = crusoe.canonical_model(value)
                if value != MODEL_NAME and value not in crusoe.MODELS:
                    raise ValueError(f"engine.params.model must be {MODEL_NAME!r} or a priced Crusoe "
                                     f"model: {', '.join(crusoe.MODELS)}.")
            else:
                low, high = cls.PARAM_LIMITS[key]
                if type(value) is not int or not low <= value <= high:
                    raise ValueError(f"engine.params.{key} must be an integer from {low} to {high}.")
            result[key] = value
        return result

    @classmethod
    def is_paid(cls, params):
        return params["model"] != MODEL_NAME

    @property
    def paid(self):
        return self.is_paid(self.params)

    @classmethod
    def validate_budget(cls, budget):
        if not 1 <= budget["iterations"] <= cls.MAX_GENERATIONS:
            raise ValueError(f"budget.iterations (ShinkaEvolve generations) must be from 1 to "
                             f"{cls.MAX_GENERATIONS}.")
        if budget["restarts"] > cls.MAX_RESTARTS:
            raise ValueError(f"budget.restarts must not exceed {cls.MAX_RESTARTS} for {cls.name}.")

    def identity(self):
        identity = {"name": self.name, "version": self.version,
                    "provider": "crusoe" if self.paid else self.provider,
                    "model": self.params["model"], "params": dict(self.params),
                    "upstream": {"package": f"shinka-evolve=={SHINKA_VERSION}",
                                 "git_commit": SHINKA_GIT_COMMIT}}
        if self.paid:
            identity["pricing"] = crusoe.price_record(self.params["model"])
        return identity

    def preflight(self, budget=None, allow_paid=None):
        """Check interpreter, package, isolation and paid-run authorization before any output."""
        # Keep the venv symlink: resolving it would select the base interpreter.
        python = Path(os.path.abspath(os.path.expanduser(
            os.environ.get("SHINKA_PYTHON", str(DEFAULT_PYTHON)))))
        if not python.is_file():
            raise EngineError(f"ShinkaEvolve environment not found at {python}. {SETUP_HINT}")
        probe = (
            "import importlib.metadata, json, sys; "
            "print(json.dumps({'version': importlib.metadata.version('shinka-evolve'), "
            "'python': list(sys.version_info[:2])}))"
        )
        try:
            process = subprocess.run(
                [str(python), "-I", "-c", probe], capture_output=True, text=True,
                env=_driver_environment(), timeout=15, check=False)
            if process.returncode != 0:
                raise ValueError("interpreter could not load the package metadata")
            metadata = json.loads(process.stdout)
            version = metadata["python"]
            if (not isinstance(version, list) or len(version) != 2
                    or any(type(part) is not int for part in version)
                    or tuple(version) < (3, 10)):
                raise ValueError("Python >= 3.10 is required")
            if metadata["version"] != SHINKA_VERSION:
                raise ValueError(f"expected shinka-evolve=={SHINKA_VERSION}, "
                                 f"found {metadata['version']!r}")
        except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError) as error:
            raise EngineError(f"Invalid ShinkaEvolve environment at {python}: {error}. "
                              f"{SETUP_HINT}") from None
        reason = isolation.unavailable_reason()
        if reason:
            raise EngineError(f"Candidate isolation is unavailable, so candidates cannot run: {reason}")
        if self.paid:
            self._authorize_paid(budget, self.allow_paid if allow_paid is None else allow_paid)
        return python

    def _authorize_paid(self, budget, allow_paid):
        model = self.params["model"]
        if not allow_paid:
            raise EngineError(f"{model} is a paid Crusoe model; pass --allow-paid to spend money.")
        if budget is None or not budget.get("max_cost_usd"):
            raise EngineError("A paid run needs budget.max_cost_usd above 0.")
        if not os.environ.get(crusoe.API_KEY_ENV, "").strip():
            raise EngineError(f"Set {crusoe.API_KEY_ENV} for paid Crusoe runs.")
        try:
            crusoe.upstream_url()
        except ValueError as error:
            raise EngineError(str(error)) from None
        run_cap = usd(budget["max_cost_usd"])
        try:
            totals = SpendLedger("preflight", run_cap).totals()
        except (LedgerError, ValueError) as error:
            raise EngineError(f"Cannot check the spend ledger: {error}") from None
        if totals["reservation_violation"] is not None:
            raise EngineError("A prior request exceeded its reserved cost; review the spend ledger "
                              "and billing assumptions before further paid runs.")
        if run_cap > totals["lifetime_remaining"]:
            raise EngineError(f"Run budget ${run_cap} exceeds the ${totals['lifetime_remaining']} left "
                              f"under the ${LIFETIME_CAP_USD} lifetime cap.")
        typical = token_cost(crusoe.prices(model), crusoe.MAX_INPUT_TOKENS,
                             self.params["max_output_tokens"])
        if typical > run_cap:
            raise EngineError(f"Run budget ${run_cap} cannot cover one worst-case request (${typical:.4f}); "
                              "raise max_cost_usd or lower max_output_tokens.")

    def run(self, task, budget, rng, history, initial=(), workspace=None):
        if initial:
            raise EngineError("ShinkaEvolve starts from its seed program; initial constructions "
                              "are not supported.")
        if workspace is None:
            raise EngineError("The ShinkaEvolve engine needs a workspace directory.")
        python = self.preflight(budget=budget)
        workspace = Path(workspace).resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        verifier_before = _sha256(VERIFIER_SOURCE)
        env = _driver_environment()

        best, best_restart, generations, requests, runtime = [], None, 0, 0, None
        with self._model_endpoint(budget, workspace, history) as (model_spec, proxy):
            if proxy is not None:
                env[PROXY_TOKEN_ENV] = proxy.token
            for restart in range(budget["restarts"]):
                seed = rng.child(restart).randbelow(2 ** 32)
                results_dir = workspace / f"restart_{restart}"
                spec_path = workspace / f"restart_{restart}_spec.json"
                spec_path.write_text(json.dumps({
                    "results_dir": str(results_dir),
                    "generations": budget["iterations"],
                    "seed": seed,
                    "dimension": task.dimension,
                    "evaluator_python": sys.executable,
                    "eval_timeout_seconds": self.params["eval_timeout_seconds"],
                    "eval_cpu_seconds": self.params["eval_cpu_seconds"],
                    "eval_memory_mb": self.params["eval_memory_mb"],
                    "num_islands": self.params["num_islands"],
                    "model": model_spec,
                }, indent=2, sort_keys=True), encoding="utf-8")
                history.write("restart_start", restart=restart, seed=seed, initial_size=0)

                log_path = workspace / f"restart_{restart}_driver.log"
                timeout = budget["iterations"] * (self.params["eval_timeout_seconds"] + 600) + 300
                with open(log_path, "w", encoding="utf-8") as log:
                    try:
                        process = subprocess.run(
                            [str(python), "-I", str(HERE / "driver.py"), str(spec_path)],
                            cwd=str(workspace), env=env, stdout=log, stderr=subprocess.STDOUT,
                            timeout=timeout, check=False)
                    except subprocess.TimeoutExpired:
                        raise EngineError(f"ShinkaEvolve restart {restart} exceeded {timeout} s; "
                                          f"see {log_path}.") from None
                if process.returncode != 0:
                    raise EngineError(f"ShinkaEvolve restart {restart} exited with code "
                                      f"{process.returncode}; see {log_path}.")

                summary = _read_summary(results_dir / "driver_summary.json")
                if proxy is None and summary["api_cost"] != 0:
                    raise EngineError("ShinkaEvolve reported a nonzero API cost; the stub must be free.")
                runtime = {"python": summary["python"], "shinka_version": summary["shinka_version"]}
                rows = summary["generations"]
                for row in rows:
                    history.write("generation", restart=restart, generation=row["generation"],
                                  correct=row["correct"], score=row["combined_score"],
                                  error=(row["error"] or "")[:200] or None)
                candidate = self._best_verified(task, results_dir, rows, history, restart)
                generations += len(rows)
                requests += summary["model_requests"]
                history.write("restart_end", restart=restart, generations=len(rows),
                              correct=sum(row["correct"] for row in rows),
                              model_requests=summary["model_requests"], best_size=len(candidate))
                if best_restart is None or len(candidate) > len(best):
                    best, best_restart = candidate, restart
        spend = self._spend_report(proxy)
        if spend is not None:
            if spend["reservation_violation"] is not None:
                raise EngineError(f"A provider charge exceeded its reservation; paid requests stopped. "
                                  f"See {workspace / 'spend_summary.json'}.")
            if spend["successful"] == 0 and any(spend[key] for key in (
                    "forwarded", "refused", "transport_errors", "upstream_errors")):
                raise EngineError(f"No provider request succeeded; the initial construction alone "
                                  f"does not confirm live inference. See {workspace / 'spend_summary.json'}.")

        if _sha256(VERIFIER_SOURCE) != verifier_before:
            raise EngineError("verification/c7.py changed during the ShinkaEvolve run.")
        cost = {"usd": 0.0, "llm_calls": requests, "input_tokens": 0, "output_tokens": 0}
        if spend is not None:
            cost = {"usd": float(spend["charged_usd"]), "usd_exact": spend["charged_usd"],
                    "llm_calls": spend["forwarded"], "input_tokens": spend["prompt_tokens"],
                    "output_tokens": spend["completion_tokens"], "cached_tokens": spend["cached_tokens"]}
        return {"vertices": best, "best_restart": best_restart, "iterations": generations,
                "cost": cost, "runtime": runtime, "spend": spend}

    @contextmanager
    def _model_endpoint(self, budget, workspace, history):
        """Yield (driver model spec, metering proxy or None) for this run."""
        if not self.paid:
            yield {"kind": "stub", "name": MODEL_NAME}, None
            return
        model = self.params["model"]
        ledger = SpendLedger(f"{workspace.parent.name}-{uuid.uuid4().hex[:8]}", budget["max_cost_usd"])
        before = ledger.totals()
        history.write("spend_start", model=model, run_cap=str(ledger.run_cap),
                      lifetime_cap=str(LIFETIME_CAP_USD),
                      lifetime_committed=str(before["lifetime_committed"]))
        proxy = MeteringProxy(crusoe.upstream_url(), os.environ[crusoe.API_KEY_ENV].strip(), model,
                              crusoe.prices(model), ledger, self.params["max_output_tokens"],
                              log_path=workspace / "spend.jsonl",
                              max_input_tokens=crusoe.MAX_INPUT_TOKENS)
        try:
            with proxy:
                spec = {"kind": "openai", "name": model, "base_url": proxy.base_url,
                        "api_key_env": PROXY_TOKEN_ENV,
                        "max_output_tokens": self.params["max_output_tokens"]}
                yield spec, proxy
        finally:
            # Shutdown drains in-flight accounting before writing, including on
            # driver failure. A missing final manifest never implies zero spend.
            try:
                report = self._spend_report(proxy, history)
            except LedgerError as error:
                report = dict(proxy.summary(), ledger_error=str(error),
                              ledger=str(ledger.path), accounting_complete=False)
            (workspace / "spend_summary.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _spend_report(proxy, history=None):
        if proxy is None:
            return None
        report = proxy.summary()
        totals = proxy.ledger.totals()
        report.update(run_cap=str(totals["run_cap"]), run_committed=str(totals["run_committed"]),
                      lifetime_cap=str(totals["lifetime_cap"]),
                      lifetime_committed=str(totals["lifetime_committed"]),
                      lifetime_remaining=str(totals["lifetime_remaining"]),
                      reservation_violation=totals["reservation_violation"],
                      run_open_reservations=str(totals["run_open_reservations"]),
                      ledger=str(proxy.ledger.path))
        if history is not None:
            history.write("spend_end", **{key: report[key] for key in (
                "charged_usd", "forwarded", "refused", "stopped", "prompt_tokens", "completion_tokens",
                "cached_tokens", "run_committed", "lifetime_committed")})
        return report

    @staticmethod
    def _best_verified(task, results_dir, rows, history, restart):
        """The highest-scoring candidate that verification.c7 accepts on re-check here."""
        root = results_dir.resolve()
        ranked = sorted((row for row in rows if row["correct"] and row["certificate"]),
                        key=lambda row: (-(row["combined_score"] or 0), row["generation"]))
        for row in ranked:
            path = (results_dir / row["certificate"]).resolve()
            reason = "certificate path outside the results directory"
            if root in path.parents:
                verdict = official.check(path)
                result = verdict["result"] if isinstance(verdict["result"], dict) else {}
                reason = result.get("reason") or "checker gave no structured output"
                if verdict["valid"] and result.get("size") == row["combined_score"]:
                    try:
                        return task.vertices_from_certificate(json.loads(path.read_text(encoding="utf-8")))
                    except (ValueError, KeyError) as error:
                        reason = str(error)
                elif verdict["valid"]:
                    reason = "checker size differs from the recorded score"
            history.write("candidate_rejected", restart=restart, generation=row["generation"],
                          reason=reason[:200])
        raise EngineError(f"ShinkaEvolve restart {restart} produced no verified candidate; "
                          f"see {results_dir}.")


def _read_summary(path):
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary["shinka_version"] != SHINKA_VERSION:
            raise ValueError(f"expected ShinkaEvolve {SHINKA_VERSION}, "
                             f"found {summary['shinka_version']!r}")
        if not isinstance(summary["python"], str):
            raise ValueError("missing Python runtime version")
        rows = summary["generations"]
        if not (isinstance(rows, list) and isinstance(summary["model_requests"], int)
                and isinstance(summary["api_cost"], (int, float))):
            raise ValueError("unexpected field types")
        for row in rows:
            score = row["combined_score"]
            if not (type(row["generation"]) is int and type(row["correct"]) is bool
                    and (score is None or type(score) in (int, float))
                    and (row["certificate"] is None or isinstance(row["certificate"], str))
                    and (row["error"] is None or isinstance(row["error"], str))):
                raise ValueError("unexpected generation row")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise EngineError(f"Unreadable ShinkaEvolve summary at {path}: {error}") from None
    return summary


def _driver_environment():
    env = {key: os.environ[key] for key in PASSTHROUGH_ENV if key in os.environ}
    env.update(NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost", PYTHONHASHSEED="0")
    return env


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
