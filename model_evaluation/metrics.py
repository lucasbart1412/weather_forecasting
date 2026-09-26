"""Weather assessment metrics."""

import numpy as np
from sklearn.metrics import mean_absolute_error


def compute_forecast_metrics(y_true, y_pred, y_climat, y_persistence):
    """Calculate the existing metrics of the weather assessment."""
    mae_modele = mean_absolute_error(y_true, y_pred)
    mae_climat = mean_absolute_error(y_true, y_climat)
    mae_persistance = mean_absolute_error(y_true, y_persistence)
    maess = 1 - (mae_modele / mae_climat)
    diff_persistance = mae_persistance - mae_modele
    return {
        "mae_modele": mae_modele,
        "mae_climat": mae_climat,
        "mae_persistance": mae_persistance,
        "maess": maess,
        "diff_persistance": diff_persistance,
    }


def apply_physical_bounds(target_variable, predictions):
    """Applies existing physical bounds to predictions."""
    if target_variable == "cloud":
        predictions = np.clip(predictions, 0, 100)
    if target_variable in ["precip", "wind", "gusts"]:
        predictions = np.maximum(0, predictions)
    elif target_variable in ["hum", "cloud"]:
        predictions = np.clip(predictions, 0, 100)
    return predictions


__all__ = ["apply_physical_bounds", "compute_forecast_metrics"]
