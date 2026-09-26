from unittest.mock import Mock

import pandas as pd
import pytest


@pytest.fixture
def weather_json_response():
    """Minimum JSON response corresponding to an Open-Meteo hourly response."""
    return {
        "hourly": {
            "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
            "temperature_2m": [4.5, 4.2],
            "relative_humidity_2m": [82, 84],
            "pressure_msl": [1012.0, 1011.5],
        }
    }


@pytest.fixture
def mock_json_response(weather_json_response):
    """Configurable response object for JSON API testing."""
    response = Mock()
    response.status_code = 200
    response.json.return_value = weather_json_response
    response.text = ""
    return response


@pytest.fixture
def raw_weather_dataframe():
    """Little deterministic weather game for feature engineering tests."""
    dates = pd.date_range("2024-01-01", periods=8, freq="h")
    return pd.DataFrame(
        {
            "date": dates,
            "temp": [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5],
            "hum": [82.0, 81.0, 80.0, 79.0, 78.0, 77.0, 76.0, 75.0],
            "press": [1012.0, 1011.5, 1011.0, 1010.5, 1010.0, 1009.5, 1009.0, 1008.5],
            "wind": [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
            "wind_dir": [180.0] * 8,
            "dew": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4],
            "t850": [-1.0] * 8,
            "t700": [-8.0] * 8,
            "t500": [-24.0] * 8,
            "h850": [75.0] * 8,
            "h700": [65.0] * 8,
            "t1000": [5.0] * 8,
        }
    )
