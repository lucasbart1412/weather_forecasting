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

from gc import collect

from numpy import (
    array,
    asarray,
    ascontiguousarray,
    column_stack,
    float32,
    isnan,
    mean,
    nan_to_num,
    where,
)
from sklearn.base import BaseEstimator, RegressorMixin

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


class StackingMixin:
    """Compatibility mix for serializable stacker."""

    def __init__(
        self,
        estimators=None,
        final_estimator=None,
        cv=None,
        early_stopping_rounds=75,
        final_estimators=None,
    ):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.early_stopping_rounds = early_stopping_rounds
        self.estimators_ = []
        self.final_estimator_ = None
        self.is_fitted_ = False
        self.best_iterations_ = {}
        self.feature_names_in_ = None
        self.meta_scaler_means_ = None
        self.meta_scaler_stds_ = None
        self.final_estimators = (
            final_estimators if final_estimators is not None else estimators
        )
        self.target_type_ = None

    def _apply_time_series_imputation(self, X_np):
        mask = isnan(X_np)
        if not mask.any():
            return X_np

        n_samples, n_features = X_np.shape
        for j in range(n_features):
            col = X_np[:, j]
            col_mask = mask[:, j]
            if not col_mask.any():
                continue

            last_valid_val = 0.0
            for i in range(n_samples):
                if col_mask[i]:
                    col[i] = last_valid_val
                else:
                    last_valid_val = col[i]

        return X_np

    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            "fit() is not available in production mode. "
            "Use the separate training module to train the models."
        )

    @property
    def feature_importances_(self):
        importances_list = []
        for pipe in self.estimators_:
            if hasattr(pipe, "named_steps") and "model" in pipe.named_steps:
                model_step = pipe.named_steps["model"]
            else:
                model_step = pipe

            if hasattr(model_step, "feature_importances_"):
                imp = model_step.feature_importances_
                total_imp = imp.sum()
                if total_imp > 0:
                    importances_list.append(imp / total_imp)
                else:
                    importances_list.append(imp)

        if not importances_list:
            raise AttributeError(
                "No base model has a 'feature_importances_' attribute"
            )

        return mean(array(importances_list), axis=0)

    def predict(self, X):
        if not self.is_fitted_:
            raise RuntimeError('This estimator is not yet trained.')

        X_local = X.copy() if hasattr(X, "copy") else X
        del X
        collect()

        if hasattr(X_local, "columns"):
            cols_to_keep = [c for c in X_local.columns if c != "time_step"]
            X_local = ascontiguousarray(X_local[cols_to_keep].to_numpy(dtype=float32))

        X_local = self._apply_time_series_imputation(X_local)
        if hasattr(X_local, "drop") and "time_step" in X_local.columns:
            X_local = X_local.drop(columns=["time_step"])

        meta_features = []
        for pipe in self.estimators_:
            pred = asarray(pipe.predict(X_local).flatten(), dtype=float32)
            meta_features.append(pred)

        X_meta = asarray(column_stack(meta_features), dtype=float32)
        if self.meta_scaler_means_ is not None and self.meta_scaler_stds_ is not None:
            X_meta -= self.meta_scaler_means_
            safe_stds = where(self.meta_scaler_stds_ == 0, 1.0, self.meta_scaler_stds_)
            X_meta /= safe_stds

        X_meta = nan_to_num(X_meta, nan=0.0, posinf=0.0, neginf=0.0)
        return self.final_estimator_.predict(X_meta)


class TimeSeriesStackingRegressor(StackingMixin, BaseEstimator, RegressorMixin):
    """Historical compatibility: the real logic is in the mixin."""

