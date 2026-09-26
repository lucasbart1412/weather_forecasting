"""Facade dedicated to the Optuna optimization of the pipeline."""


def optimize_short_block(manager, *args, **kwargs):
    """Delegates short block optimization to the existing ModelManager."""
    return manager.optimize_short_block(*args, **kwargs)


__all__ = ["optimize_short_block"]
