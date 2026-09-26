"""Dedicated access to drive pipeline bias correction."""


def train_bias_correctors(manager, *args, **kwargs):
    """Delegates full training to the manager who produces the biases."""
    return manager.train(*args, **kwargs)


__all__ = ["train_bias_correctors"]
