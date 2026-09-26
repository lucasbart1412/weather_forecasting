"""
eval_model.py
Weather forecast model evaluation script (WeatherMaster).
This module is self-contained and imports its dependencies from config.py,
utils_features.py and models.py.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load as job_load

from feature_engineering.feature_engineering import FeatureEngineer, optimize_memory
from model_evaluation.metrics import compute_forecast_metrics
from model_evaluation.preprocessing_data import prepare_test_data

# Imports from new refactored modules


class ModelEvaluator:
    def __init__(self, base_path, geo):
        self.base_path = Path(base_path)
        self.geo = geo
        # We load the metadata generated during training
        self.meta = job_load(self.base_path / "global_meta.pkl")
        self.blocks = {
            "short": [1, 2, 3, 4, 6, 12, 18],
            "medium": [30, 36, 42, 48, 54, 60, 66, 72],
            "long": [84, 96, 108, 120, 132, 144, 156, 168],
        }

    def prepare_test_data(self, df_raw_path, cut_off_date):
        print('📥 Loading and preparing test data...')
        df_raw = pd.read_parquet(df_raw_path)
        df_raw["date"] = pd.to_datetime(df_raw["date"])
        df_raw = df_raw.sort_values("date").reset_index(drop=True)
        df_raw = optimize_memory(df_raw)

        cut_off_date = pd.Timestamp(cut_off_date)

        # 1. Strict calculation of the climate average on the TRAIN only (as in the fit)
        clima = {}
        df_train_only = df_raw[df_raw["date"] < cut_off_date]
        for t in ["temp", "press", "hum"]:
            if t in df_raw.columns:
                clima[t] = (
                    df_train_only.groupby(
                        [df_train_only["date"].dt.month, df_train_only["date"].dt.hour]
                    )[t]
                    .mean()
                    .to_dict()
                )

        # 2. Call for FeatureEngineer
        df_processed, _ = FeatureEngineer.prepare(
            df_raw,
            tz_name=self.geo["tz"],
            climatology_ref=clima,
            lon=self.geo["lon"],
            lat=self.geo["lat"],
            elevation=self.geo.get("elevation", 0),
        )

        if "tide" in df_processed.columns:
            df_processed = FeatureEngineer.add_tide_physics(df_processed)

        df_processed = df_processed.ffill(limit=3)

        # 3. Strict isolation of the test set (Final Validation)
        gap_hours = 168
        val_start_date = cut_off_date + pd.Timedelta(hours=gap_hours)
        df_test = df_processed[df_processed["date"] >= val_start_date].copy()

        return df_test, clima

    def evaluate(self, df_test, clima, target_variable="cloud", specific_horizon=24):
        print('\n' + "=" * 50)
        print(f"📊 EVALUATION : {target_variable.upper()} at H+{specific_horizon}")
        print("=" * 50)

        # Determination of the time block
        block_name = (
            "short"
            if specific_horizon <= 18
            else "medium"
            if specific_horizon <= 72
            else "long"
        )

        # Template upload
        model_path = self.base_path / f"model_{target_variable}_{block_name}.pkl"
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return

        bundle = job_load(model_path)
        model = bundle["model"]
        features = bundle.get("features", self.meta["features"])

        # Preparation of actual targets (Offset H+x)
        y_true_raw = df_test[target_variable].shift(-specific_horizon)
        y_now = df_test[target_variable]  # Persistence (value at H+0)

        # Valid Values Filter (NaN)
        mask = y_true_raw.notna()
        df_h = df_test.loc[mask].copy()
        y_true_raw = y_true_raw.loc[mask]
        y_now = y_now.loc[mask]

        if len(df_h) == 0:
            print('❌ Not enough valid test data for this horizon.')
            return

        # Construction of specific features on the horizon
        future_dates = df_h["date"] + pd.Timedelta(hours=specific_horizon)
        df_h["time_step"] = df_h.loc[mask].index.astype("int32")
        df_h["horizon_h"] = specific_horizon
        df_h["target_solar_elev"] = FeatureEngineer.get_solar_elev_vectorized(
            self.geo["lat"], self.geo["lon"], future_dates
        ).astype("float32")
        df_h["target_hour_sin"] = np.sin(2 * np.pi * future_dates.dt.hour / 24).astype(
            "float32"
        )
        df_h["target_hour_cos"] = np.cos(2 * np.pi * future_dates.dt.hour / 24).astype(
            "float32"
        )
        df_h["target_month_sin"] = np.sin(
            2 * np.pi * future_dates.dt.month / 12
        ).astype("float32")
        df_h["target_month_cos"] = np.cos(
            2 * np.pi * future_dates.dt.month / 12
        ).astype("float32")

        X_test = df_h[features].astype("float32")

        # Security for stacking
        if "time_step" in X_test.columns:
            X_test = X_test.drop(columns=["time_step"])

        # Knowing
        raw_preds = model.predict(X_test)

        # Management of specificities (scaling and anomalies)
        if target_variable == "cloud":
            y_pred = np.clip(raw_preds, 0, 100)
        else:
            y_pred = raw_preds

        # Addition of the standard if the target has been driven in anomaly (temp, press, hum)
        if target_variable in ["temp", "press", "hum"]:
            norm_futures = [
                clima[target_variable].get((m, h), df_test[target_variable].mean())
                for m, h in zip(future_dates.dt.month, future_dates.dt.hour)
            ]
            y_pred = y_pred + np.array(norm_futures)
            y_climat = np.array(norm_futures)
        else:
            y_climat = np.full_like(y_true_raw, df_test[target_variable].mean())

        # --- CALCULATION OF METRICS ---
        if target_variable in ["precip", "wind", "gusts"]:
            y_pred = np.maximum(0, y_pred)
        elif target_variable in ["hum", "cloud"]:
            y_pred = np.clip(y_pred, 0, 100)

        # --- CALCULATION OF METRICS ---
        metrics = compute_forecast_metrics(y_true_raw, y_pred, y_climat, y_now)
        mae_modele = metrics["mae_modele"]
        mae_climat = metrics["mae_climat"]
        mae_persistance = metrics["mae_persistance"]
        maess = metrics["maess"]
        diff_persistance = metrics["diff_persistance"]

        # View
        print(f"🔹 Model MAE           : {mae_modele:.3f}")
        print(f"🔹 MAE of Persistence   : {mae_persistance:.3f}")
        print(f"🔹 Climate MAE           : {mae_climat:.3f}\n")
        print('--- RESULTS ---')
        print(
            f"✅ 1) MAESS (Skill Score)  : {maess:.4f}"
            + (
                ' (The model beats the climate!)'
                if maess > 0
                else ' (The model is worse than average...)'
            )
        )
        print(
            f"✅ 2) Gain vs Persistence  : {diff_persistance:.3f}"
            + (
                '(The model anticipates the change!)'
                if diff_persistance > 0
                else "(The model does worse than saying 'identical weather')"
            )
        )

        # --- CHART FOR CLOUDINESS ---
        if target_variable == "cloud":
            self.plot_cloud_distribution(y_pred)

    def plot_cloud_distribution(self, y_pred):
        plt.figure(figsize=(10, 6))
        bins = np.arange(0, 105, 5)
        plt.hist(
            y_pred,
            bins=bins,
            rwidth=0.85,
            color="#4a90e2",
            edgecolor="black",
            alpha=0.8,
        )
        plt.title(
            'Distribution of Cloud Cover Forecasts (Cloud Cover)',
            fontsize=14,
            fontweight='bold',
            pad=15,
        )
        plt.xlabel("Predicted cloud cover (%)", fontsize=12)
        plt.ylabel("Number of forecasts", fontsize=12)
        plt.xticks(np.arange(0, 101, 10))
        plt.axvline(
            x=50, color="red", linestyle="--", linewidth=2, label="Moyenne (50%)"
        )
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

ModelEvaluator.prepare_test_data = prepare_test_data
