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


def test_stacking_fit_is_disabled_in_production():
    with pytest.raises(NotImplementedError):
        TimeSeriesStackingRegressor().fit(np.zeros((2, 2)), np.zeros(2))
