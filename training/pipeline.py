"""Public entry point for the training pipeline."""

from training.bias_corrector import train_bias_correctors
from training.memory_impl import MemoryManager
from training.model_manager_impl import ModelManager
from training.optuna_train import optimize_short_block
from training.pipeline_core import main as implementation_main
from training.preprocessing_data import optimize_memory
from training.stacking_impl import TimeSeriesStackingTrainer


def main() -> None:
    """Run the training command-line interface."""
    return implementation_main()


__all__ = [
    "MemoryManager",
    "ModelManager",
    "TimeSeriesStackingTrainer",
    "main",
    "optimize_memory",
    "optimize_short_block",
    "train_bias_correctors",
]
