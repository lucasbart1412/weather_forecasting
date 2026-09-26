"""Historical Features Services Compatibility Facade."""

from data_fetcher.data_fetcher import DataFetcher
from feature_engineering.feature_engineering import FeatureEngineer
from feature_engineering.memory import optimize_memory
from infrastructure.geo_api import LocationManager

LegacyLocationManager = LocationManager
LegacyDataFetcher = DataFetcher

__all__ = [
    "DataFetcher",
    "FeatureEngineer",
    "LegacyDataFetcher",
    "LegacyLocationManager",
    "LocationManager",
    "optimize_memory",
]
