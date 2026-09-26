from pathlib import Path
from typing import Dict, List

from numpy import arange, int32, sort, where
from pandas import DataFrame

from training._runtime import *


class ModelManagerBase:
    def __init__(self, path: str, lat: float, lon: float):
        self.path = Path(path)
        self.lat, self.lon = lat, lon
        self.bias_correctors = {}
        self.path.mkdir(parents=True, exist_ok=True)

        self.BLOCKS = {
            "short": [1, 2, 3, 4, 6, 12, 18],
            "medium": [30, 36, 42, 48, 54, 60, 66, 72],
            "long": [84, 96, 108, 120, 132, 144, 156, 168],
        }

    def add_feature_penalty(self, param: Dict, X_tr: DataFrame) -> Dict:
        """Adds a penalty for macro/synoptic features."""
        penalties = []
        patterns_macro = ["macro", "syn", "hemi"]

        for col in [c for c in X_tr.columns if c != "time_step"]:
            if any(p in col.lower() for p in patterns_macro):
                penalties.append(0.3)
            else:
                penalties.append(1.0)

        param["feature_penalty"] = penalties
        return param

    def custom_ridgecv_tscv(
        self, X: DataFrame, n_splits: int = 5, gap: int = 168
    ) -> List:
        """Generates leak-free temporal splits."""
        if hasattr(X, "columns") and "time_step" in X.columns:
            time_steps_series = X["time_step"].reset_index(drop=True)
            unique_steps = sort(time_steps_series.unique())
            n_samples = len(unique_steps)
            test_size = (n_samples - gap) // (n_splits + 1)

            if test_size <= 0:
                raise ValueError(f"Dataset too short for gap={gap}")

            custom_splits = []

            for i in range(1, n_splits + 1):
                train_end = i * test_size
                val_start = train_end + gap
                val_end = val_start + test_size

                if val_start < n_samples:
                    train_steps = unique_steps[0:train_end]
                    val_steps = unique_steps[val_start:val_end]

                    train_idx = where(
                        time_steps_series.isin(train_steps).to_numpy()
                    )[0].astype(int32)
                    val_idx = where(
                        time_steps_series.isin(val_steps).to_numpy()
                    )[0].astype(int32)

                    if len(val_idx) > 0:
                        custom_splits.append((train_idx, val_idx))
            return custom_splits
        else:
            n_samples = len(X)
            test_size = (n_samples - gap) // (n_splits + 1)
            custom_splits = []

            for i in range(1, n_splits + 1):
                train_end = i * test_size
                val_start = train_end + gap
                val_end = val_start + test_size

                val_end = min(val_end, n_samples)
                if val_start < n_samples:
                    train_idx = arange(0, train_end, dtype=int32)
                    val_idx = arange(val_start, val_end, dtype=int32)

                    if len(val_idx) > 0:
                        custom_splits.append((train_idx, val_idx))
            return custom_splits

