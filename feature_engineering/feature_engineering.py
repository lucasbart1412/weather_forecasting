"""Canonical entry point for feature engineering."""

from feature_engineering.feature_engineer_impl import FeatureEngineer
from feature_engineering.memory import optimize_memory
from infrastructure.config import WeatherConfig, logger

Feature_Engineering = FeatureEngineer

__all__ = ["FeatureEngineer", "Feature_Engineering", "WeatherConfig", "logger", "optimize_memory"]