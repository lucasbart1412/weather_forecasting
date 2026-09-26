"""Canonical entry point of the drive pipeline."""

from data_fetcher.data_fetcher import DataFetcher
from feature_engineering.feature_engineering import FeatureEngineer
from infrastructure.config import WeatherConfig, logger, setup_logger
from ml.ml_models import InplacePredictPreprocessor, TimeSeriesStackingRegressor
from training.bias_corrector import train_bias_correctors
from training.memory_impl import MemoryManager
from training.model_manager_impl import ModelManager
from training.optuna_train import optimize_short_block
from training.pipeline import main
from training.preprocessing_data import optimize_memory
from training.stacking_impl import TimeSeriesStackingTrainer

__all__ = [
    "DataFetcher",
    "FeatureEngineer",
    "InplacePredictPreprocessor",
    "MemoryManager",
    "ModelManager",
    "TimeSeriesStackingRegressor",
    "TimeSeriesStackingTrainer",
    "WeatherConfig",
    "logger",
    "main",
    "optimize_memory",
    "optimize_short_block",
    "setup_logger",
    "train_bias_correctors",
]
