"""Training helpers for residual-based weather forecast correction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BIAS_CORRECTOR_ESTIMATORS = 1

BIAS_CORRECTOR_FEATURES = (
    "horizon_h",
    "target_solar_elev",
    "target_hour_sin",
    "target_hour_cos",
    "target_month_sin",
    "target_month_cos",
)


def fit_bias_corrector(
    features: pd.DataFrame,
    actual: Any,
    predictions: Any,
) -> Pipeline:
    """Fit a regressor to calibration residuals (actual minus prediction)."""
    prediction_values = np.asarray(predictions, dtype=np.float32).reshape(-1)
    actual_values = np.asarray(actual, dtype=np.float32).reshape(-1)
    if len(features) != len(actual_values) or len(actual_values) != len(prediction_values):
        raise ValueError("Features, calibration targets, and predictions must align.")

    model_features = pd.DataFrame(
        {"predictions": prediction_values},
        index=features.index,
    )
    for name in BIAS_CORRECTOR_FEATURES:
        if name in features.columns:
            model_features[name] = features[name]

    residuals = actual_values - prediction_values
    valid_rows = np.isfinite(residuals)
    if not valid_rows.any():
        raise ValueError("Bias-correction training has no finite calibration residuals.")

    corrector = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=10.0)),
        ]
    )
    corrector.fit(model_features.loc[valid_rows], residuals[valid_rows])
    return corrector


def train_bias_correctors(manager, *args, **kwargs):
    """Delegate full training to the model manager."""
    return manager.train(*args, **kwargs)


__all__ = [
    "BIAS_CORRECTOR_ESTIMATORS",
    "BIAS_CORRECTOR_FEATURES",
    "fit_bias_corrector",
    "train_bias_correctors",
]
