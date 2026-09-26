#!/usr/bin/env python3
"""Drive Pipeline Historical CLI Facade."""

from training.memory_impl import MemoryManager
from training.model_manager_impl import ModelManager
from training.pipeline import main
from training.pipeline_core import optimize_memory
from training.stacking_impl import TimeSeriesStackingTrainer

__all__ = [
    "MemoryManager",
    "ModelManager",
    "TimeSeriesStackingTrainer",
    "main",
    "optimize_memory",
]


if __name__ == "__main__":
    main()