"""Loading all models into memory (Fast because we only have 21 compressed files)"""
from pathlib import Path

from joblib import load as job_load
from numpy import array, cos, greater, less, pi, sin
from pandas import DataFrame, Series, Timedelta, to_datetime

from feature_engineering.feature_engineering import FeatureEngineer
from infrastructure.config import WeatherConfig
from ml.bias_correction import apply_bias_correction


class WeatherPredictor:
    def __init__(self, base_path):
        self.base_path = base_path
        self.meta = job_load(Path(base_path) / "global_meta.pkl")
        self.models = {}
        self.blocks = ["short", "medium", "long"]

        # Loading all models into memory (Fast because we only have 21 compressed files)
        for t in WeatherConfig.TARGETS:
            self.models[t] = {}
            for b in self.blocks:
                p = Path(base_path) / f"model_{t}_{b}.pkl"
                if p.exists():
                    self.models[t][b] = job_load(p)

    def _get_block_for_horizon(self, h):
        if h <= 18:
            return "short"
        elif h <= 72:
            return "medium"
        else:
            return "long"

    def get_tide_extremes(self, predictions):
        """Detects peaks (Highs) and troughs (Lows) in the prediction series."""
        from scipy.signal import argrelextrema

        heights = array([p["tide"] for p in predictions])
        times = [p["time"] for p in predictions]

        # Find indices of local maxima and minima
        max_idx = argrelextrema(heights, greater)[0]
        min_idx = argrelextrema(heights, less)[0]

        extremes = []
        for i in max_idx:
            extremes.append({"type": "HAUTE", "time": times[i], "height": heights[i]})
        for i in min_idx:
            extremes.append({"type": "BASSE", "time": times[i], "height": heights[i]})

        return sorted(extremes, key=lambda x: x["time"])

    def predict(
        self, target_hours: float, df_live: DataFrame, obs: dict, geo: dict
    ) -> dict:
        # X_dict contains the CURRENT state of the atmosphere (T0) and its recent history, as well as the temporal information of the targeted future (T+H)
        last_row = df_live.iloc[-1]

        # We prepare the dictionary with the current data
        X_dict = {f: last_row.get(f, 0) for f in self.meta["features"]}

        target_dt = obs["time"] + Timedelta(hours=target_hours)

        # Explicitly overwrites/fills "future" features
        X_dict["horizon_h"] = target_hours
        X_dict["target_solar_elev"] = FeatureEngineer.get_solar_elev_vectorized(
            geo["lat"], geo["lon"], Series([target_dt])
        )[0]
        X_dict["target_hour_sin"] = sin(2 * pi * target_dt.hour / 24)
        X_dict["target_hour_cos"] = cos(2 * pi * target_dt.hour / 24)
        X_dict["target_month_sin"] = sin(2 * pi * target_dt.month / 12)
        X_dict["target_month_cos"] = cos(2 * pi * target_dt.month / 12)

        # Creating the DataFrame by forcing the exact order of the meta features
        X = DataFrame([X_dict])[self.meta["features"]].astype("float32")

        block = self._get_block_for_horizon(target_hours)
        res = {}

        for t in WeatherConfig.TARGETS:
            if block not in self.models[t]:
                res[t] = obs.get(t, 0)
                continue
            bundle = self.models[t][block]
            model_features = bundle.get("features", self.meta["features"])
            X["time_step"] = (
                target_hours  # Added the feature time_step for stacking
            )
            # We only keep the columns that the model knows
            X_model = X[model_features].copy()
            if "time_step" in X_model.columns:
                X_model = X_model.drop(columns=["time_step"])
            if t == "precip":
                # 1. Load and predict probability
                prob_path = Path(self.base_path) / f"model_precip_prob_{block}.pkl"
                if prob_path.exists():
                    prob_model = job_load(prob_path)
                    # predict_proba returns [P(no), P(yes)]
                    res["precip_prob"] = prob_model.predict_proba(X_model)[0][1] * 100

                # 2. Predict quantity only if proba > 5% (to avoid noise)
                bundle = self.models[t][block]
                if res.get("precip_prob", 0) > 5:
                    # Inference with the right columns
                    raw_prob = res["precip_prob"]
                    res["raw_precip_prob"] = raw_prob
                    res["precip_prob"] = max(10, round(raw_prob / 10) * 10)
                else:
                    res["precip_prob"] = 0
                    res["raw_precip_prob"] = 0
                    res[t] = 0
                    continue

            # Inference with the right columns
            raw_pred = bundle["model"].predict(X_model)[0]
            if t == "cloud":
                # Security to avoid slight digital overflows from the Stacking (e.g.: 100.1% or -0.2%)
                raw_pred = max(0.0, min(100.0, raw_pred))
            # INITIALIZATION BEFORE CORRECTION
            prediction_finale = raw_pred

            # Applying time bias correction
            m_h = target_dt.month
            t_h = target_dt.hour
            try:
                bias = apply_bias_correction(
                    bias_model=bundle["bias_corrector"],
                    df_inference=X_model,
                    target=t,
                    df_predictions=Series([raw_pred]),
                )
            except Exception:
                bias = 0.0
            prediction_finale = raw_pred + bias

            if t in ["temp", "press", "hum"]:
                norm = self.meta["clima"].get(t, {}).get((m_h, t_h), obs.get(t, 0))
                res[t] = norm + prediction_finale
                if t == "hum":
                    res[t] = min(
                        max(0, res[t]), 100
                    )  # Realistic humidity between 0 and 100%
                if t == "press":
                    res[t] = max(
                        300, min(1200, res[t])
                    )  # Realistic pressure between 300 and 1200 hPa
                if t == "temp" and bundle["m_low"] is not None:
                    res["t_low"] = norm + bundle["m_low"].predict(X_model)[0]
                    res["t_high"] = norm + bundle["m_high"].predict(X_model)[0]
                    res[t] = min(
                        max(-30, res[t]), 50
                    )  # Realistic temperature between -30 and 50°C
            else:
                res[t] = max(
                    0, prediction_finale
                )  # Rain, wind, cloud cannot be negative
        if "t_low" not in res:
            res["t_low"], res["t_high"] = res["temp"] - 1.5, res["temp"] + 1.5
        target_dt = to_datetime(obs["time"]) + Timedelta(hours=int(target_hours))
        res["time"] = target_dt
        return res


