from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from data_fetcher.data_fetcher import DataFetcher

CSV_RESPONSE = """# latitude: 50.8
# longitude: 4.3
time,temperature_2m (°C),relative_humidity_2m (%),pressure_msl (hPa)
2024-01-01T00:00,4.5,82,1012
2024-01-01T01:00,4.2,84,1011.5
"""


@pytest.fixture
def geo_point():
    return {"tz": "Europe/Brussels", "points": {"center": (50.8, 4.3)}}


def test_fetch_single_point_parses_csv_successfully(geo_point):
    response = Mock(status_code=200, text=CSV_RESPONSE)
    fetcher = DataFetcher()
    fetcher.ua = None

    with patch.object(fetcher.session, "get", return_value=response) as mock_get:
        result = fetcher._fetch_single_point(
            "center", 50.8, 4.3, geo_point, is_archive=False
        )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["date", "temp", "hum", "press"]
    assert result.loc[0, "temp"] == 4.5
    assert result.loc[1, "hum"] == 84
    mock_get.assert_called_once()


def test_fetch_single_point_returns_none_for_http_404(geo_point):
    response = Mock(status_code=404, text="not found")
    fetcher = DataFetcher()
    fetcher.ua = None

    with patch.object(fetcher.session, "get", return_value=response) as mock_get, patch(
        "data_fetcher.data_fetcher.trigger_optional_vpn"
    ) as mock_vpn:
        result = fetcher._fetch_single_point(
            "center", 50.8, 4.3, geo_point, is_archive=False
        )

    assert result is None
    assert mock_get.call_count == 5
    mock_vpn.assert_called()


def test_fetch_single_point_returns_none_for_http_500(geo_point):
    response = Mock(status_code=500, text="server error")
    fetcher = DataFetcher()
    fetcher.ua = None

    with patch.object(fetcher.session, "get", return_value=response) as mock_get, patch(
        "data_fetcher.data_fetcher.trigger_optional_vpn"
    ):
        result = fetcher._fetch_single_point(
            "center", 50.8, 4.3, geo_point, is_archive=False
        )

    assert result is None
    assert mock_get.call_count == 5


def test_fetch_single_point_handles_timeout(geo_point):
    fetcher = DataFetcher()
    fetcher.ua = None

    with patch.object(
        fetcher.session, "get", side_effect=requests.Timeout("API timeout")
    ) as mock_get, patch("data_fetcher.data_fetcher.trigger_optional_vpn"):
        result = fetcher._fetch_single_point(
            "center", 50.8, 4.3, geo_point, is_archive=False
        )

    assert result is None
    assert mock_get.call_count == 5
