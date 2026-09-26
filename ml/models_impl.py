"""Composition of serializable ML components."""

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

from ml.preprocessing import PreprocessingMixin
from ml.stacking import StackingMixin


class InplacePredictPreprocessor(PreprocessingMixin, BaseEstimator, TransformerMixin):
    """ML preprocessor keeping the historical API."""


class TimeSeriesStackingRegressor(StackingMixin, BaseEstimator, RegressorMixin):
    """Stacking estimator keeping the historical API."""


__all__ = ["InplacePredictPreprocessor", "TimeSeriesStackingRegressor"]
