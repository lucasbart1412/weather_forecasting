"""Bias-correction utilities used during model inference."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMRegressor


def apply_bias_correction(
    df_inference: pd.DataFrame,
    bias_model: LGBMRegressor,
    target: str,
    df_predictions: pd.Series,
) -> pd.Series:
    """Apply a fitted bias model to predictions for one target.

    ``target`` is kept in the signature for compatibility with the legacy
    helper and for callers that dispatch corrections by target name.
    """
    del target
    features = bias_model.feature_name_
    predicted_bias = bias_model.predict(df_inference[features])
    return df_predictions + predicted_bias


__all__ = ["apply_bias_correction"]
