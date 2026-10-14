import joblib
import numpy as np
import pandas as pd
import pytest

from ml.ml_models import (
    InplacePredictPreprocessor,
    TimeSeriesStackingRegressor,
)
from ml.bias_correction import apply_bias_correction
from ml.predictor import WeatherPredictor
from training.bias_corrector import fit_bias_corrector


class ConstantEstimator:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=np.float32)


class SumEstimator:
    def predict(self, X):
        return np.asarray(X, dtype=np.float32).sum(axis=1)


class BiasEstimator:
    feature_name_ = ["temp"]

    def predict(self, X):
        return np.full(len(X), 0.5, dtype=np.float32)


def test_bias_correction_applies_fitted_delta():
    predictions = pd.Series([1.0, 2.0])
    inference = pd.DataFrame({"temp": [10.0, 11.0]})

    result = apply_bias_correction(
        inference,
        BiasEstimator(),
        "temp",
        predictions,
    )

    assert np.allclose(result, [1.5, 2.5])


def test_predictor_uses_neutral_values_for_unavailable_synoptic_deltas(caplog):
    predictor = WeatherPredictor.__new__(WeatherPredictor)
    predictor.meta = {"features": ["temperature"]}

    class Model:
        feature_name_ = ["temperature", "w_macro_temp_delta"]

    result = predictor._select_model_features(
        pd.DataFrame({"temperature": [12.0]}),
        {"features": ["temperature", "w_macro_temp_delta"]},
        Model(),
        "temp",
        "short",
    )

    assert list(result.columns) == ["temperature", "w_macro_temp_delta"]
    assert np.allclose(result["w_macro_temp_delta"], [0.0])
    assert "using neutral zero" in caplog.text


def test_predictor_still_rejects_missing_non_delta_features():
    predictor = WeatherPredictor.__new__(WeatherPredictor)
    predictor.meta = {"features": ["temperature"]}

    class Model:
        feature_name_ = ["temperature", "pressure"]

    with pytest.raises(ValueError, match="Missing features for temp/short"):
        predictor._select_model_features(
            pd.DataFrame({"temperature": [12.0]}),
            {"features": ["temperature", "pressure"]},
            Model(),
            "temp",
            "short",
        )


def test_bias_corrector_learns_residuals_without_time_step():
    feature = np.tile([0.0, 1.0], 400)
    features = pd.DataFrame(
        {
            "horizon_h": np.ones(len(feature), dtype=np.float32),
            "target_solar_elev": feature,
            "target_hour_sin": np.zeros(len(feature), dtype=np.float32),
            "target_hour_cos": np.ones(len(feature), dtype=np.float32),
            "target_month_sin": np.zeros(len(feature), dtype=np.float32),
            "target_month_cos": np.ones(len(feature), dtype=np.float32),
            "irrelevant_feature": np.arange(len(feature), dtype=np.float32),
            "time_step": np.arange(len(feature), dtype=np.int32),
        }
    )
    residuals = np.where(feature > 0.5, 4.0, -2.0)

    corrector = fit_bias_corrector(
        features,
        pd.Series(residuals, dtype=np.float32),
        np.zeros(len(feature), dtype=np.float32),
    )

    assert corrector.feature_names_in_.tolist() == [
        "predictions",
        "horizon_h",
        "target_solar_elev",
        "target_hour_sin",
        "target_hour_cos",
        "target_month_sin",
        "target_month_cos",
    ]
    inference = features.iloc[[0, 1]].drop(columns=["irrelevant_feature", "time_step"])
    correction = apply_bias_correction(
        inference,
        corrector,
        "temp",
        pd.Series([0.0, 0.0]),
    )
    assert correction[1] > correction[0] + 3


def test_legacy_bias_correction_uses_horizon_hour_and_season():
    features = pd.DataFrame(
        {
            "horizon_h": [1.0, 1.0],
            "target_hour_sin": [0.0, 0.0],
            "target_hour_cos": [1.0, 1.0],
            "target_month_sin": [
                np.sin(2 * np.pi * 1 / 12),
                np.sin(2 * np.pi * 12 / 12),
            ],
            "target_month_cos": [
                np.cos(2 * np.pi * 1 / 12),
                np.cos(2 * np.pi * 12 / 12),
            ],
        }
    )
    legacy_corrector = {
        1: {
            "global": {0: 0.25},
            "seasonal": {("winter", 0): 0.5},
        }
    }

    result = apply_bias_correction(
        features,
        legacy_corrector,
        "temp",
        pd.Series([10.0, 10.0]),
    )

    assert np.allclose(result, [10.5, 10.5])


def test_inplace_preprocessor_preserves_shape_and_scales_values():
    preprocessor = InplacePredictPreprocessor(
        medians=np.array([2.0, 5.0], dtype=np.float32),
        means=np.array([1.0, 3.0], dtype=np.float32),
        stds=np.array([1.0, 2.0], dtype=np.float32),
        feature_names=["temp", "precip"],
    )
    values = np.array([[np.nan, np.nan], [3.0, 7.0]], dtype=np.float32)

    result = preprocessor.transform(values)

    assert result.shape == values.shape
    assert result.dtype == np.float32
    assert np.allclose(result[0], [1.0, -1.5])
    assert np.allclose(result[1], [2.0, 2.0])


def test_time_series_imputation_keeps_array_shape():
    regressor = TimeSeriesStackingRegressor()
    values = np.array([[np.nan, 1.0], [2.0, np.nan], [np.nan, 3.0]], dtype=np.float32)

    result = regressor._apply_time_series_imputation(values.copy())

    assert result.shape == values.shape
    assert np.array_equal(result, [[0.0, 1.0], [2.0, 1.0], [2.0, 3.0]])


def test_time_series_stacking_forward_pass_without_training():
    regressor = TimeSeriesStackingRegressor()
    regressor.estimators_ = [ConstantEstimator(2.0), SumEstimator()]
    regressor.final_estimator_ = ConstantEstimator(7.5)
    regressor.is_fitted_ = True

    values = pd.DataFrame(
        {"feature_a": [1.0, np.nan], "feature_b": [2.0, 3.0], "time_step": [0, 1]}
    )
    result = regressor.predict(values)

    assert result.shape == (2,)
    assert result.dtype == np.float32
    assert np.allclose(result, [7.5, 7.5])


def test_weather_predictor_instantiates_from_metadata(tmp_path):
    metadata = {"features": ["temp", "press"]}
    joblib.dump(metadata, tmp_path / "global_meta.pkl")

    predictor = WeatherPredictor(tmp_path)

    assert predictor.base_path == tmp_path
    assert predictor.meta == metadata
    assert predictor.blocks == ["short", "medium", "long"]
    assert set(predictor.models) >= {"temp", "press"}


def test_precip_threshold_uses_bundle_value_with_legacy_fallback():
    assert WeatherPredictor._get_precip_threshold({"optimal_threshold": 0.35}, 0.5) == 0.35
    assert WeatherPredictor._get_precip_threshold({}, 0.7) == 0.7


def test_stacking_fit_is_disabled_in_production():
    with pytest.raises(NotImplementedError):
        TimeSeriesStackingRegressor().fit(np.zeros((2, 2)), np.zeros(2))
