"""Loading all models into memory (Fast because we only have 21 compressed files)"""
from pathlib import Path

from joblib import load as job_load
from numpy import array, asarray, cos, greater, less, pi, sin
from pandas import DataFrame, Series, Timedelta, isna, to_datetime

from feature_engineering.feature_engineering import FeatureEngineer
from feature_engineering.target_transforms import decode_model_target
from infrastructure.config import WeatherConfig, logger
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

    @staticmethod
    def _get_precip_threshold(bundle, fallback):
        return float(bundle.get("optimal_threshold", fallback))

    def _feature_names_for(self, bundle, estimator):
        """Return the model's ordered feature schema without stacking metadata."""
        model_features = getattr(estimator, "feature_name_", None)
        if model_features is None or len(model_features) == 0:
            model_features = getattr(estimator, "feature_names_in_", None)
        if model_features is not None and len(model_features) > 0:
            return [name for name in model_features if name != "time_step"]
        return [
            name
            for name in bundle.get("features", self.meta["features"])
            if name != "time_step"
        ]

    @staticmethod
    def _is_neutral_synoptic_delta(feature):
        return feature.endswith("_delta") and any(
            scale in feature for scale in ("_syn_", "_macro_", "_hemi_")
        )

    def _select_model_features(self, df, bundle, estimator, target, block):
        """Align inference data to a bundle and fill only absent synoptic deltas."""
        model_features = self._feature_names_for(bundle, estimator)
        missing_features = [name for name in model_features if name not in df.columns]
        neutral_deltas = [
            name
            for name in missing_features
            if self._is_neutral_synoptic_delta(name)
        ]
        if neutral_deltas:
            warning_key = tuple(neutral_deltas)
            warned_schemas = getattr(self, "_warned_neutral_delta_schemas", set())
            if warning_key not in warned_schemas:
                logger.warning(
                    "Missing synoptic delta features for %s/%s; using neutral zero: %s",
                    target,
                    block,
                    neutral_deltas,
                )
                self._warned_neutral_delta_schemas = warned_schemas | {warning_key}
            df = df.copy()
            for name in neutral_deltas:
                df[name] = 0.0

        remaining_missing = [
            name for name in model_features if name not in df.columns
        ]
        if remaining_missing:
            raise ValueError(
                f"Missing features for {target}/{block}: {remaining_missing}"
            )
        return df.loc[:, model_features].astype("float32")

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
        last_row = df_live.iloc[-1]
        block = self._get_block_for_horizon(target_hours)

        model_features = set(self.meta["features"])
        for bundles in self.models.values():
            bundle = bundles.get(block)
            if bundle is not None:
                model_features.update(
                    self._feature_names_for(bundle, bundle["model"])
                )

        X_dict = {}
        for feature in model_features:
            value = last_row.get(feature, 0)
            if (
                self._is_neutral_synoptic_delta(feature)
                and (feature not in last_row.index or isna(value))
            ):
                continue
            X_dict[feature] = value

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

        X = DataFrame([X_dict]).astype("float32")

        res = {}

        for t in WeatherConfig.TARGETS:
            if block not in self.models[t]:
                res[t] = obs.get(t, 0)
                continue
            bundle = self.models[t][block]
            X["time_step"] = (
                target_hours  # Added the feature time_step for stacking
            )
            X_model = self._select_model_features(
                X, bundle, bundle["model"], t, block
            )
            if t == "precip":
                # 1. Load and predict probability
                prob_path = Path(self.base_path) / f"model_precip_prob_{block}.pkl"
                if prob_path.exists():
                    prob_model = job_load(prob_path)
                    # predict_proba returns [P(no), P(yes)]
                    X_prob = self._select_model_features(
                        X,
                        {"features": self.meta["features"]},
                        prob_model,
                        "precip_prob",
                        block,
                    )
                    res["precip_prob"] = prob_model.predict_proba(X_prob)[0][1] * 100

                threshold = self._get_precip_threshold(bundle, 0.05) * 100
                # 2. Predict quantity only above the validation-optimized threshold.
                bundle = self.models[t][block]
                if res.get("precip_prob", 0) >= threshold:
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
            raw_pred = float(bundle["model"].predict(X_model)[0])
            if t == "cloud":
                # Security to avoid slight digital overflows from the Stacking (e.g.: 100.1% or -0.2%)
                raw_pred = max(0.0, min(100.0, raw_pred))
            # Corrector returns the prediction plus its learned residual estimate.
            m_h = target_dt.month
            t_h = target_dt.hour
            bias_corrector = bundle.get("bias_corrector")
            if bias_corrector is not None:
                corrected_prediction = apply_bias_correction(
                    bias_model=bias_corrector,
                    df_inference=X,
                    target=t,
                    df_predictions=Series([raw_pred]),
                )
                prediction_finale = float(corrected_prediction.iloc[0])
            else:
                prediction_finale = raw_pred
            decoded_prediction = decode_model_target(
                t,
                [prediction_finale],
                bundle.get("target_transform"),
            )
            prediction_finale = float(asarray(decoded_prediction).reshape(-1)[0])

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
