"""Apply learned residual corrections to model predictions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _estimator_features(bias_model) -> list[str]:
    """Get the input schema from a fitted estimator."""
    features = getattr(bias_model, "feature_name_", None)
    if features is None or len(features) == 0:
        features = getattr(bias_model, "feature_names_in_", None)
    if features is None:
        raise ValueError("The fitted bias corrector does not expose its feature names.")
    return list(features)


def _legacy_bias_delta(
    bias_model: Mapping,
    df_inference: pd.DataFrame,
) -> np.ndarray:
    """Read older hour/season correction bundles while models are migrated."""
    required = {
        "horizon_h",
        "target_hour_sin",
        "target_hour_cos",
        "target_month_sin",
        "target_month_cos",
    }
    missing = required.difference(df_inference.columns)
    if missing:
        raise ValueError(f"Legacy bias correction requires features: {sorted(missing)}")

    hour_angle = np.arctan2(
        df_inference["target_hour_sin"].to_numpy(),
        df_inference["target_hour_cos"].to_numpy(),
    )
    hours = np.rint(hour_angle * 24 / (2 * np.pi)).astype(int) % 24

    month_sin = df_inference["target_month_sin"].to_numpy()
    month_cos = df_inference["target_month_cos"].to_numpy()
    month_angles = 2 * np.pi * np.arange(1, 13) / 12
    month_scores = (
        np.outer(month_sin, np.sin(month_angles))
        + np.outer(month_cos, np.cos(month_angles))
    )
    months = np.argmax(month_scores, axis=1) + 1

    deltas = []
    for horizon, month, hour in zip(
        df_inference["horizon_h"].to_numpy(), months, hours, strict=True
    ):
        horizon_bias = bias_model.get(int(round(float(horizon))), {})
        seasonal = horizon_bias.get("seasonal", {})
        hourly = horizon_bias.get("global", {})
        if month in (12, 1, 2):
            season = "winter"
        elif month in (3, 4, 5):
            season = "spring"
        elif month in (6, 7, 8):
            season = "summer"
        else:
            season = "autumn"
        deltas.append(seasonal.get((season, int(hour)), hourly.get(int(hour), 0.0)))
    return np.asarray(deltas, dtype=np.float32)


def apply_bias_correction(
    df_inference: pd.DataFrame,
    bias_model,
    target: str,
    df_predictions: pd.Series,
) -> pd.Series:
    """Return predictions adjusted by the model's estimate of their residual."""
    if len(df_inference) != len(df_predictions):
        raise ValueError("Inference features and predictions must have the same length.")

    if isinstance(bias_model, Mapping):
        predicted_bias = _legacy_bias_delta(bias_model, df_inference)
    else:
        features = _estimator_features(bias_model)
        missing = [
            name
            for name in features
            if name != "predictions" and name not in df_inference.columns
        ]
        if missing:
            raise ValueError(
                f"Missing features for {target} bias correction: {missing}"
            )
        model_features = pd.DataFrame(index=df_inference.index)
        for name in features:
            if name == "predictions":
                model_features[name] = df_predictions.to_numpy(dtype=np.float32)
            else:
                model_features[name] = df_inference[name]
        predicted_bias = np.asarray(
            bias_model.predict(model_features.loc[:, features]), dtype=np.float32
        ).reshape(-1)

    if len(predicted_bias) != len(df_predictions):
        raise ValueError("Bias corrector returned an unexpected number of predictions.")
    return pd.Series(
        df_predictions.to_numpy(dtype=np.float32) + predicted_bias,
        index=df_predictions.index,
        name=df_predictions.name,
    )


__all__ = ["apply_bias_correction"]
