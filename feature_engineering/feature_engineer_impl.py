"""Composite facade preserving the historical FeatureEngineer API."""

from feature_engineering.astronomy import AstronomyFeaturesMixin
from feature_engineering.feature_pipeline import FeaturePreparationMixin
from feature_engineering.physics import PhysicsFeaturesMixin


class FeatureEngineer(
    AstronomyFeaturesMixin,
    PhysicsFeaturesMixin,    FeaturePreparationMixin,
):
    """Composite facade preserving the historical FeatureEngineer API."""


__all__ = ["FeatureEngineer"]