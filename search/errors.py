"""Errors shared by engines and the run pipeline."""


class EngineError(RuntimeError):
    """An engine could not finish a run: missing environment, crash or integrity failure."""
