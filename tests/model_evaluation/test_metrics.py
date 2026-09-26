import numpy as np
import pandas as pd

from model_evaluation.metrics import apply_physical_bounds, compute_forecast_metrics
from model_evaluation.preprocessing_data import resolve_cut_off_date


def test_metrics_and_physical_bounds_are_available():
    y_true = np.array([0.0, 1.0])
    metrics = compute_forecast_metrics(
        y_true, y_true, np.array([1.0, 1.0]), y_true
    )

    assert metrics["mae_modele"] == 0.0
    assert metrics["maess"] == 1.0
    assert np.array_equal(apply_physical_bounds("cloud", np.array([-1.0, 101.0])), [0.0, 100.0])


def test_evaluation_respects_explicit_cut_off_date():
    dates = pd.date_range("2024-01-01", periods=10, freq="h")
    data = pd.DataFrame({"date": dates})

    assert resolve_cut_off_date(data, "2024-01-01 05:00") == pd.Timestamp(
        "2024-01-01 05:00"
    )
