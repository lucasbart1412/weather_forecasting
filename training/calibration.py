"""Time-ordered calibration data splitting."""

from __future__ import annotations

import pandas as pd


def split_calibration_data(
    features: pd.DataFrame,
    targets: pd.Series,
    gap_steps: int,
    fit_fraction: float = 0.6,
    threshold_fraction: float = 0.78,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """Split data for residual fitting, threshold tuning, and holdout evaluation."""
    if "time_step" not in features.columns:
        raise ValueError("Calibration features must include the time_step column.")
    if not features.index.equals(targets.index):
        raise ValueError("Calibration features and targets must have matching indices.")
    if gap_steps < 0:
        raise ValueError("The calibration gap cannot be negative.")
    if not 0 < fit_fraction < 1:
        raise ValueError("The calibration fit fraction must be between zero and one.")
    if not fit_fraction < threshold_fraction < 1:
        raise ValueError(
            "The threshold fraction must be greater than fit fraction and less than one."
        )
    if features.empty:
        raise ValueError("Calibration data cannot be empty.")

    steps = features["time_step"]
    first_step = int(steps.min())
    last_step = int(steps.max())
    span = last_step - first_step
    fit_cutoff = first_step + int(span * fit_fraction)
    evaluation_cutoff = first_step + int(span * threshold_fraction)
    fit_mask = steps < fit_cutoff - gap_steps
    threshold_mask = (steps >= fit_cutoff) & (
        steps < evaluation_cutoff - gap_steps
    )
    evaluation_mask = steps >= evaluation_cutoff
    if not fit_mask.any() or not threshold_mask.any() or not evaluation_mask.any():
        raise ValueError(
            "Calibration data is too short for disjoint fit and evaluation periods."
        )

    return (
        features.loc[fit_mask].copy(),
        targets.loc[fit_mask].copy(),
        features.loc[threshold_mask].copy(),
        targets.loc[threshold_mask].copy(),
        features.loc[evaluation_mask].copy(),
        targets.loc[evaluation_mask].copy(),
    )


__all__ = ["split_calibration_data"]