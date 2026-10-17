"""Run one ShinkaEvolve search inside the pinned ShinkaEvolve environment (Python >= 3.10).

Started by integrations.shinka.engine as
  .venv/shinka/bin/python -I integrations/shinka/driver.py SPEC.json
SPEC holds results_dir, generations, seed, dimension, evaluator_python,
eval_timeout_seconds, eval_cpu_seconds, eval_memory_mb, num_islands and model.
model is {"kind": "stub"} (the localhost stub, started here) or {"kind":
"openai", ...}: the engine's metering proxy, authenticated by the token in
the named environment variable. This process never sees a provider API key.
Writes RESULTS_DIR/driver_summary.json; candidate scores come from
evaluate.py, which asks verification.c7.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import random
import re
import sys
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stub_model import MODEL_NAME, StubServer  # noqa: E402

TASK_SYS_MSG = (
    "You improve a Python program that constructs a large independent set in the strong product "
    "of {dimension} copies of the 7-cycle. Vertices are tuples of {dimension} integers from 0 to 6; two "
    "distinct vertices conflict when every coordinate differs by 0 or +-1 modulo 7. The score is the "
    "number of vertices, counted by an exact checker, and any conflict scores 0. Published lower "
    "bounds: 108 vertices in the 4th power (Vesel and Zerovnik, 2002) and 367 in the 5th power "
    "(Polak and Schrijver, 2018). The program runs with the Python standard library only, no network "
    "or subprocesses, and limits of {timeout} s wall time, {cpu} s CPU and {memory} MiB memory. "
    "construct(dimension) must keep returning a list of such tuples. Only edit code between "
    "EVOLVE-BLOCK-START and EVOLVE-BLOCK-END."
)


def main(argv):
    spec = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    model = spec["model"]
    if model["kind"] not in ("stub", "openai"):
        raise SystemExit(f"Unknown model kind {model['kind']!r}.")
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
    random.seed(spec["seed"])

    import numpy
    import shinka
    from shinka.core import EvolutionConfig, ShinkaEvolveRunner
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig

    numpy.random.seed(spec["seed"])
    results_dir = Path(spec["results_dir"])
    timeout = spec["eval_timeout_seconds"]
    task_sys_msg = TASK_SYS_MSG.format(dimension=spec["dimension"], timeout=timeout,
                                       cpu=spec["eval_cpu_seconds"], memory=spec["eval_memory_mb"])
    with _model(model) as settings:
        evo_config = EvolutionConfig(
            task_sys_msg=task_sys_msg,
            patch_types=settings["patch_types"],
            patch_type_probs=settings["patch_type_probs"],
            num_generations=spec["generations"],
            max_patch_resamples=1,
            max_patch_attempts=1,
            max_novelty_attempts=1,
            job_type="local",
            language="python",
            llm_models=[settings["model"]],
            llm_dynamic_selection=None,
            llm_kwargs=settings["llm_kwargs"],
            meta_rec_interval=None,
            embedding_model=None,
            novelty_llm_models=None,
            init_program_path=str(HERE / "initial_program.py"),
            results_dir=str(results_dir),
            max_api_costs=None,
        )
        job_config = LocalJobConfig(
            eval_program_path=str(HERE / "evaluate.py"),
            time=_hms(timeout + 60),
            python_executable=spec["evaluator_python"],
            extra_cmd_args={"dimension": spec["dimension"], "timeout": timeout,
                            "cpu_seconds": spec["eval_cpu_seconds"],
                            "memory_mb": spec["eval_memory_mb"]},
        )
        db_config = DatabaseConfig(num_islands=spec["num_islands"], archive_size=20)
        runner = ShinkaEvolveRunner(
            evo_config=evo_config, job_config=job_config, db_config=db_config,
            banner_style="minimal", verbose=False,
            max_evaluation_jobs=1, max_proposal_jobs=1, max_db_workers=1,
        )
        runner.run()
        model_requests = settings["requests"]()
    summary = {
        "shinka_version": shinka.__version__,
        "python": platform.python_version(),
        "model": model["name"],
        "model_requests": model_requests,
        "api_cost": float(runner.total_api_cost),
        "generations": _scan(results_dir),
    }
    (results_dir / "driver_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return 0


@contextmanager
def _model(model):
    """Yield ShinkaEvolve settings for the stub or for the engine's metering proxy."""
    if model["kind"] == "stub":
        with StubServer() as stub:
            yield {"model": f"local/{MODEL_NAME}@{stub.base_url}", "patch_types": ["diff"],
                   "patch_type_probs": [1.0], "llm_kwargs": {"temperatures": [0.0], "max_tokens": 2048},
                   "requests": lambda: stub.requests}
        return
    url = urlsplit(model["base_url"])
    if url.scheme != "http" or url.hostname != "127.0.0.1":
        raise SystemExit("Paid models are only reachable through the local metering proxy.")
    # The proxy counts and bills requests; ShinkaEvolve's own cost is unpriced here.
    yield {"model": f"local/{model['name']}@{model['base_url']}?api_key_env={model['api_key_env']}",
           "patch_types": ["diff", "full"], "patch_type_probs": [0.7, 0.3],
           "llm_kwargs": {"temperatures": [0.0, 0.5, 1.0], "max_tokens": model["max_output_tokens"]},
           "requests": lambda: 0}


def _scan(results_dir):
    """One row per gen_N directory, from the files evaluate.py wrote."""
    rows = []
    for gen_dir in results_dir.iterdir():
        match = re.fullmatch(r"gen_(\d+)", gen_dir.name)
        if not match or not gen_dir.is_dir():
            continue
        results = gen_dir / "results"
        correct = _read_json(results / "correct.json")
        metrics = _read_json(results / "metrics.json")
        certificate = results / "certificate.json"
        rows.append({
            "generation": int(match.group(1)),
            "correct": correct.get("correct") is True,
            "combined_score": metrics.get("combined_score"),
            "error": correct.get("error"),
            "certificate": str(certificate.relative_to(results_dir)) if certificate.is_file() else None,
        })
    return sorted(rows, key=lambda row: row["generation"])


def _read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _hms(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
