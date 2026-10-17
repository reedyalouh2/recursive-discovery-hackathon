"""Offline engine: the fixed ILS strategy under a finite, seeded budget.

Makes no network or model calls. The engine sees only the domain graph, the
budget, the rng and the history; it never calls the verifier.
"""

from strategies.mis_ils import IteratedLocalSearch


class OfflineILSEngine:
    name = "offline-ils"
    version = "1"
    provider = "none"
    model = "none"
    deterministic = True
    supports_initial_construction = True
    DEFAULT_PARAMS = {"max_perturbation": 4, "tournament": 4, "perturb_continue_pct": 50,
                      "perturb_radius": 2, "accept_scale_pct": 100}
    PARAM_LIMITS = {"max_perturbation": (1, 16), "tournament": (1, 64), "perturb_continue_pct": (10, 90),
                    "perturb_radius": (1, 3), "accept_scale_pct": (0, 400)}

    def __init__(self, params):
        self.params = self.validate_params(params)

    @classmethod
    def validate_params(cls, params):
        """Return params with defaults filled in, or raise ValueError."""
        if not isinstance(params, dict):
            raise ValueError("engine.params must be a JSON object.")
        unknown = sorted(set(params) - set(cls.PARAM_LIMITS))
        if unknown:
            raise ValueError(f"Unknown engine.params for {cls.name}: {', '.join(unknown)}.")
        result = dict(cls.DEFAULT_PARAMS)
        for key, value in params.items():
            low, high = cls.PARAM_LIMITS[key]
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"engine.params.{key} must be an integer from {low} to {high}.")
            result[key] = value
        return result

    def identity(self):
        return {"name": self.name, "version": self.version, "provider": self.provider,
                "model": self.model, "params": dict(self.params)}

    def run(self, task, budget, rng, history, initial=(), workspace=None):
        best = None
        best_restart = None
        for restart in range(budget["restarts"]):
            history.write("restart_start", restart=restart, initial_size=len(initial))

            def improved(iteration, size, restart=restart):
                history.write("restart_best", restart=restart, iteration=iteration, size=size)

            search = IteratedLocalSearch(task, rng.child(restart), initial=initial, **self.params)
            result = search.run(budget["iterations"], improved)
            history.write(
                "restart_end", restart=restart, iterations=result["iterations"],
                accepted=result["accepted"], rejected=result["rejected"],
                best_size=result["best_size"], final_size=result["final_size"],
            )
            if best is None or result["best_size"] > len(best):
                best, best_restart = result["best"], restart
        return {
            "vertices": best,
            "best_restart": best_restart,
            "iterations": budget["iterations"] * budget["restarts"],
            "cost": {"usd": 0.0, "llm_calls": 0, "input_tokens": 0, "output_tokens": 0},
        }
