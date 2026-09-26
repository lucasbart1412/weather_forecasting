"""Composition of the training manager by specialized mixins."""

from training.model_factory import ModelFactoryMixin
from training.model_manager_base import ModelManagerBase
from training.optuna_manager import OptunaTrainingMixin
from training.training_loop import TrainingLoopMixin


class ModelManager(
    ModelManagerBase,
    OptunaTrainingMixin,
    ModelFactoryMixin,
    TrainingLoopMixin,
):
    """Compound manager to preserve the historical API."""


__all__ = ["ModelManager"]
