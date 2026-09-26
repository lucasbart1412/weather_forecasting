"""Canonical entry point for model evaluation."""

from feature_engineering.feature_engineering import FeatureEngineer, optimize_memory
from infrastructure.config import WeatherConfig, logger
from ml.ml_models import InplacePredictPreprocessor, TimeSeriesStackingRegressor
from model_evaluation.evaluator_impl import ModelEvaluator
from training.training_pipeline import (
    MemoryManager,
    ModelManager,
    TimeSeriesStackingTrainer,
)

__all__ = [
    "FeatureEngineer",
    "InplacePredictPreprocessor",
    "MemoryManager",
    "ModelEvaluator",
    "ModelManager",
    "TimeSeriesStackingRegressor",
    "TimeSeriesStackingTrainer",
    "WeatherConfig",
    "logger",
    "optimize_memory",
]
