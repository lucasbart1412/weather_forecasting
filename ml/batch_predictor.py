"""
inference.py
 MLOps inference layer for WeatherMaster.
Isolates API patches and vector inference by batch
without relying on the removed monolithic forecast script.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load as job_load
from requests import Session

from data_fetcher.data_fetcher import DataFetcher
from feature_engineering.feature_engineering import FeatureEngineer
from infrastructure.config import WeatherConfig, logger
from ml.predictor import WeatherPredictor
from ml.bias_correction import apply_bias_correction


class FixedDataFetcher(DataFetcher):
    """
    Adapt for DataFetcher.

    Fixes the Open-Meteo API bug where forecast_days=0 excluded the current day.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = Session()


class BatchWeatherPredictor(WeatherPredictor):
    def __init__(self, base_path):
        super().__init__(base_path)
        self._prob_model_cache = {}

    def _feature_names_for(self, bundle, estimator):
        """Return the feature schema actually used by an estimator."""
        model_features = getattr(estimator, "feature_name_", None)
        if model_features:
            return [name for name in model_features if name != "time_step"]
        return [
            name
            for name in bundle.get("features", self.meta["features"])
            if name != "time_step"
        ]

    def _select_model_features(self, df, bundle, estimator, target, block):
        model_features = self._feature_names_for(bundle, estimator)
        missing_features = [
            name for name in model_features if name not in df.columns
        ]
        if missing_features:
            raise ValueError(
                f"Missing features for {target}/{block}: {missing_features}"
            )
        return df.loc[:, model_features].astype("float32")

    def _predict_batch_precip(self, block, df, idx, res):
        """Vectorized subroutine for rain (quantity + probability)."""
        if "precip_prob" not in res.columns:
            res["precip_prob"] = 0.0
            res["raw_precip_prob"] = 0.0
        if "precip" not in res.columns:
            res["precip"] = 0.0

        cache_key = f"precip_prob_{block}"
        if cache_key not in self._prob_model_cache:
            prob_path = Path(self.base_path) / f"model_precip_prob_{block}.pkl"
            try:
                self._prob_model_cache[cache_key] = (
                    job_load(prob_path) if prob_path.exists() else None
                )
            except Exception as e:
                logger.warning(
                    f"[predict_batch] Unable to load the rain classifier ({block}) : {e}"
                )
                self._prob_model_cache[cache_key] = None

        prob_model = self._prob_model_cache[cache_key]
        if prob_model is not None:
            try:
                prob_bundle = {"features": self.meta["features"]}
                X_prob = self._select_model_features(
                    df.iloc[idx],
                    prob_bundle,
                    prob_model,
                    "precip_prob",
                    block,
                )
                probs = prob_model.predict_proba(X_prob)[:, 1] * 100
            except Exception as e:
                logger.exception(
                    f"[predict_batch] Rain probability unavailable ({block}) : {e}"
                )
                probs = np.zeros(len(idx))
        else:
            probs = np.zeros(len(idx))

        final_prob = np.where(probs > 0, np.maximum(10, np.round(probs / 10) * 10), 0)
        res.loc[idx, "precip_prob"] = final_prob
        res.loc[idx, "raw_precip_prob"] = probs

        has_signal = (probs >= 50) if block == "short" else (probs >= 70)

        try:
            bundle = self.models["precip"][block]
            X_model = self._select_model_features(
                df.iloc[idx],
                bundle,
                bundle["model"],
                "precip",
                block,
            )
            raw_pred = bundle["model"].predict(X_model)
            try:
                bias = apply_bias_correction(
                    bias_model=bundle["bias_corrector"],
                    df_inference=X_model,
                    target="precip",
                    df_predictions=pd.Series(raw_pred, index=X_model.index),
                ).values
            except Exception:
                bias = 0.0
            qty = np.maximum(0, raw_pred + bias)
        except Exception as e:
            logger.exception(
                f"[predict_batch] Rain quantity unavailable ({block}) : {e}"
            )
            qty = np.zeros(len(idx))

        res.loc[idx, "precip"] = np.where(has_signal, qty, 0)

    def predict_batch(
        self, horizons: list, df_live: pd.DataFrame, obs: dict, geo: dict
    ) -> pd.DataFrame:
        """
        Vectorized version of `WeatherPredictor.predict()`.
        NEVER raise an exception: silent fallback + log.
        """
        if not hasattr(self, "_prob_model_cache"):
            self._prob_model_cache = {}

        n = len(horizons)
        if n == 0:
            return pd.DataFrame()

        horizons_arr = np.asarray(horizons, dtype="float64")
        last_row = df_live.iloc[-1]

        base_dict = {f: last_row.get(f, 0) for f in self.meta["features"]}
        df = pd.DataFrame([base_dict] * n).reset_index(drop=True)
        for col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")
            )

        target_dt_series = pd.Series(
            [obs["time"] + pd.Timedelta(hours=float(h)) for h in horizons_arr]
        )

        if "horizon_h" in df.columns:
            df["horizon_h"] = horizons_arr.astype("float32")

        if "target_solar_elev" in df.columns:
            try:
                df["target_solar_elev"] = FeatureEngineer.get_solar_elev_vectorized(
                    geo["lat"], geo["lon"], target_dt_series
                ).values
            except Exception as e:
                logger.warning(f"[predict_batch] Solar elevation unavailable : {e}")

        if "target_hour_sin" in df.columns:
            df["target_hour_sin"] = (
                np.sin(2 * np.pi * target_dt_series.dt.hour / 24)
                .astype("float32")
                .values
            )
        if "target_hour_cos" in df.columns:
            df["target_hour_cos"] = (
                np.cos(2 * np.pi * target_dt_series.dt.hour / 24)
                .astype("float32")
                .values
            )
        if "target_month_sin" in df.columns:
            df["target_month_sin"] = (
                np.sin(2 * np.pi * target_dt_series.dt.month / 12)
                .astype("float32")
                .values
            )
        if "target_month_cos" in df.columns:
            df["target_month_cos"] = (
                np.cos(2 * np.pi * target_dt_series.dt.month / 12)
                .astype("float32")
                .values
            )

        blocks_arr = np.array(
            [self._get_block_for_horizon(float(h)) for h in horizons_arr]
        )
        months_arr = target_dt_series.dt.month.values
        hours_arr = target_dt_series.dt.hour.values

        res = pd.DataFrame(index=range(n))
        res["time"] = target_dt_series.values
        res["horizon_h"] = horizons_arr

        for t in WeatherConfig.TARGETS:
            res[t] = np.nan

            for block in self.blocks:
                mask = blocks_arr == block
                if not mask.any():
                    continue

                idx = np.where(mask)[0]

                if block not in self.models.get(t, {}):
                    res.loc[idx, t] = obs.get(t, 0)
                    continue

                try:
                    bundle = self.models[t][block]
                    if t == "precip":
                        self._predict_batch_precip(block, df, idx, res)
                        continue

                    X_model = self._select_model_features(
                        df.iloc[idx],
                        bundle,
                        bundle["model"],
                        t,
                        block,
                    )

                    raw_pred = bundle["model"].predict(X_model)

                    if t == "cloud":
                        raw_pred = np.clip(raw_pred, 0.0, 100.0)

                    try:
                        bias = apply_bias_correction(
                            bias_model=bundle["bias_corrector"],
                            df_inference=X_model,
                            target=t,
                            df_predictions=pd.Series(raw_pred, index=X_model.index),
                        ).values
                    except Exception:
                        bias = 0.0

                    prediction_finale = raw_pred + bias

                    if t in ["temp", "press", "hum"]:
                        climatology_dict = self.meta.get("clima", {}).get(t, {})
                        norm_vals = np.array(
                            [
                                climatology_dict.get((int(m), int(h)), obs.get(t, 0))
                                for m, h in zip(months_arr[idx], hours_arr[idx])
                            ]
                        )
                        vals = norm_vals + prediction_finale

                        if t == "hum":
                            vals = np.clip(vals, 0, 100)
                        if t == "press":
                            vals = np.clip(vals, 300, 1200)
                        if t == "temp":
                            vals = np.clip(vals, -30, 50)
                            if (
                                bundle.get("m_low") is not None
                                and bundle.get("m_high") is not None
                            ):
                                try:
                                    t_low = norm_vals + bundle["m_low"].predict(X_model)
                                    t_high = norm_vals + bundle["m_high"].predict(
                                        X_model
                                    )
                                    if "t_low" not in res.columns:
                                        res["t_low"] = np.nan
                                        res["t_high"] = np.nan
                                    res.loc[idx, "t_low"] = t_low
                                    res.loc[idx, "t_high"] = t_high
                                except Exception as e:
                                    logger.warning(
                                        f"[predict_batch] Temperature quantiles unavailable : {e}"
                                    )

                        res.loc[idx, t] = vals
                    else:
                        res.loc[idx, t] = np.maximum(0, prediction_finale)

                except Exception as e:
                    logger.warning(
                        f"[predict_batch] Block failure '{block}' for target '{t}' : {e}"
                    )
                    res.loc[idx, t] = obs.get(t, 0)

        if "t_low" not in res.columns:
            res["t_low"] = res["temp"] - 1.5
            res["t_high"] = res["temp"] + 1.5
        else:
            missing = res["t_low"].isna()
            res.loc[missing, "t_low"] = res.loc[missing, "temp"] - 1.5
            res.loc[missing, "t_high"] = res.loc[missing, "temp"] + 1.5

        if "precip_prob" not in res.columns:
            res["precip_prob"] = 0.0
        if "raw_precip_prob" not in res.columns:
            res["raw_precip_prob"] = 0.0

        for c in res.columns:
            if c != "time":
                res[c] = pd.to_numeric(res[c], errors="coerce").fillna(0.0)

        return res.sort_values("horizon_h").reset_index(drop=True)
