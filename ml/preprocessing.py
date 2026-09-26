"""
================================================================================
models.py - Deserialization Classes for Production Inference
================================================================================
This module contains ONLY the class skeletons required by joblib
to deserialize the driven .pkl models.

Classes:
- InplacePredictPreprocessor: Imputation + Scaling float32 in-place
- TimeSeriesStackingRegressor: Multi-model stacking for time series

IMPORTANT: fit() raises NotImplementedError in production because the drive
must be done via the separate drive module.
"""

from numpy import (
    asarray,
    float32,
    isnan,
    where,
)
from sklearn.base import BaseEstimator, TransformerMixin

# Conditional imports for joblib deserialization
# These classes are necessary because joblib stores the full names of the classes
try:
    from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation

    HAS_LGBM = True
except (ImportError, OSError, Exception):
    HAS_LGBM = False
    print('⚠️ LightGBM not available - deserialization of old .pkl impossible')

try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except (ImportError, OSError, Exception):
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except (ImportError, OSError, Exception):
    HAS_CATBOOST = False

# Importing the logger from config
try:
    from infrastructure.config import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# ============================================================================
# INPLACE PREDICT PREPROCESSOR
# ============================================================================


class PreprocessingMixin:
    """Compatibility mix for serializable preprocessor."""

    def __init__(self, medians=None, means=None, stds=None, feature_names=None):
        self.medians = medians
        self.means = means
        self.stds = stds
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X_mat):
        if hasattr(X_mat, "to_numpy"):
            X_np = X_mat.to_numpy(dtype=float32).copy()
        else:
            X_np = asarray(X_mat, dtype=float32).copy()

        precip_idx = -1
        if self.feature_names is not None:
            try:
                precip_idx = list(self.feature_names).index("precip")
            except ValueError:
                pass

        nan_mask = isnan(X_np)
        if nan_mask.any():
            if self.medians is None:
                raise ValueError(
                    'Training medians must be provided for imputation.'
                )

            X_np = where(nan_mask, self.medians, X_np)

            if precip_idx != -1:
                mask_precip = nan_mask[:, precip_idx]
                if mask_precip.any():
                    X_np[mask_precip, precip_idx] = 0.0

        if self.means is not None and self.stds is not None:
            X_np -= self.means
            X_np /= self.stds

        return X_np


class InplacePredictPreprocessor(PreprocessingMixin, BaseEstimator, TransformerMixin):
    """Historical compatibility: the real logic is in the mixin."""



# ============================================================================
# TIME SERIES STACKING REGRESSOR
# ============================================================================


