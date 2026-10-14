"""Regression metrics and threshold selection for weather model validation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import f1_score


def regression_metrics(
    actual: ArrayLike,
    predicted: ArrayLike,
) -> dict[str, float]:
    """Return MAE, RMSE, and Pearson correlation for aligned finite arrays."""
    actual_values = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted_values = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if actual_values.shape != predicted_values.shape:
        raise ValueError("Actual and predicted values must have the same shape.")
    if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("Regression metrics require finite values.")
    if actual_values.size == 0:
        raise ValueError("Regression metrics require at least one sample.")

    mae = float(np.mean(np.abs(actual_values - predicted_values)))
    rmse = float(np.sqrt(np.mean(np.square(actual_values - predicted_values))))
    if actual_values.size < 2 or np.std(actual_values) == 0 or np.std(predicted_values) == 0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(actual_values, predicted_values)[0, 1])

    return {"mae": mae, "rmse": rmse, "pearson_r": correlation}


def select_f1_threshold(
    actual: ArrayLike,
    probabilities: ArrayLike,
    thresholds: ArrayLike,
) -> tuple[float, float]:
    """Choose a binary threshold by F1 on a separate tuning set."""
    actual_values = np.asarray(actual).reshape(-1)
    probabilities_values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    threshold_values = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if actual_values.shape != probabilities_values.shape:
        raise ValueError("Labels and probabilities must have the same shape.")
    if actual_values.size == 0 or threshold_values.size == 0:
        raise ValueError("Threshold selection requires labels and candidate thresholds.")
    if not np.isin(actual_values, [0, 1]).all():
        raise ValueError("Threshold selection labels must be binary.")
    if not np.isfinite(probabilities_values).all() or not np.isfinite(threshold_values).all():
        raise ValueError("Probabilities and thresholds must be finite.")
    if ((probabilities_values < 0) | (probabilities_values > 1)).any():
        raise ValueError("Probabilities must be between zero and one.")
    if ((threshold_values < 0) | (threshold_values > 1)).any():
        raise ValueError("Thresholds must be between zero and one.")

    if np.unique(actual_values).size < 2:
        return 0.5, 0.0

    best_threshold = float(threshold_values[0])
    best_f1 = -1.0
    for threshold in threshold_values:
        predictions = probabilities_values >= threshold
        score = f1_score(actual_values, predictions, zero_division=0)
        if score > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(score)
    return best_threshold, best_f1


def has_multiple_classes(labels: ArrayLike) -> bool:
    """Return whether labels contain at least two distinct classes."""
    return np.unique(np.asarray(labels).reshape(-1)).size > 1


__all__ = ["has_multiple_classes", "regression_metrics", "select_f1_threshold"]
