import numpy as np
import pandas as pd

from feature_engineering.feature_engineering import Feature_Engineering


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
