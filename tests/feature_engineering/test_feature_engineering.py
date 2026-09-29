import numpy as np
import pandas as pd

from feature_engineering.feature_engineering import Feature_Engineering
from feature_engineering.target_transforms import (
    decode_model_target,
    decode_rain_target,
    encode_rain_target,
)


def test_rain_target_log_transform_round_trips_and_bounds_negative_values():
    amounts = np.array([0.0, 0.1, 1.0, 10.0, -1.0], dtype=np.float32)

    encoded = encode_rain_target(amounts)
    decoded = decode_rain_target(encoded)

    assert np.allclose(decoded, [0.0, 0.1, 1.0, 10.0, 0.0], atol=1e-6)
    assert np.all(np.isfinite(encoded))


def test_model_target_decode_only_applies_to_tagged_precipitation():
    encoded = np.log1p(np.array([0.0, 1.0, 10.0], dtype=np.float32))

    decoded = decode_model_target("precip", encoded, "log1p")

    assert np.allclose(decoded, [0.0, 1.0, 10.0], atol=1e-6)
    assert np.array_equal(decode_model_target("precip", encoded, None), encoded)
    assert np.array_equal(decode_model_target("temp", encoded, "log1p"), encoded)


def test_add_physics_adds_features_without_changing_row_count(raw_weather_dataframe):
    raw = raw_weather_dataframe.copy()
    result = Feature_Engineering.add_physics(raw, lat=50.8)

    assert len(result) == len(raw_weather_dataframe)
    assert result.index.equals(raw_weather_dataframe.index)
    assert result.shape[1] > raw_weather_dataframe.shape[1]
    assert {"vpd", "wind_u", "wind_v", "coriolis_param"}.issubset(result.columns)
    assert result["vpd"].notna().all()
    assert np.isfinite(result["wind_u"]).all()
    assert np.isfinite(result["coriolis_param"]).all()


def test_prepare_preserves_dimensions_and_adds_temporal_features(raw_weather_dataframe):
    raw = raw_weather_dataframe.copy()
    result, clima = Feature_Engineering.prepare(
        raw,
        tz_name="Europe/Brussels",
        lon=4.3,
        lat=50.8,
        elevation=62.0,
    )

    assert isinstance(result, pd.DataFrame)
    assert isinstance(clima, dict)
    assert len(result) == len(raw_weather_dataframe)
    assert result.shape[1] > raw_weather_dataframe.shape[1]
    assert {"solar_elev", "month_sin", "month_cos", "hour_sin", "hour_cos"}.issubset(
        result.columns
    )
    assert result["date"].is_monotonic_increasing
