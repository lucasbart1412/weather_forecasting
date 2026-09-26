from pathlib import Path

from infrastructure.config import WeatherConfig


def test_weather_config_exposes_portable_paths_and_api_contracts():
    assert isinstance(WeatherConfig.BASE_DIR, Path)
    assert WeatherConfig.GEO_CACHE_FILE.parent == WeatherConfig.BASE_DIR
    assert WeatherConfig.API_VARS
    assert WeatherConfig.MAPPING_VARS
    assert WeatherConfig.MARINE_API_URL.startswith("https://")
