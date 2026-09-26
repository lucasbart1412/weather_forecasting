"""Weather icon creators used by the interface."""

import pandas as pd


def get_weather_icon(
    cloud_val: float,
    precip_val: float,
    temp_val: float = None,
    wind_val: float = None,
    hum_val: float = None,
) -> str:
    """Extended weather icon: rain, thunderstorm, snow, fog, strong wind, cloudiness, thinning."""
    try:
        cloud_val = 0 if cloud_val is None or pd.isna(cloud_val) else float(cloud_val)
        precip_val = (
            0 if precip_val is None or pd.isna(precip_val) else float(precip_val)
        )
        temp_val = None if temp_val is None or pd.isna(temp_val) else float(temp_val)
        wind_val = 0 if wind_val is None or pd.isna(wind_val) else float(wind_val)
        hum_val = 0 if hum_val is None or pd.isna(hum_val) else float(hum_val)

        if temp_val is not None and temp_val <= 0.5 and precip_val > 0.1:
            return "🌨️"  # Snow
        if precip_val > 4.0:
            return "⛈️"  # thunderstorm with heavy rain
        if precip_val > 1.0:
            return "🌧️"  # Rain
        if precip_val > 0.1:
            return "🌦️"  # Flurries
        if cloud_val > 85 and wind_val < 10 and hum_val > 90:
            return "🌫️"  # Fog
        if cloud_val > 75:
            return "☁️"  # CLOUDY
        if cloud_val > 35:
            return "⛅"  # Partly cloudy
        if wind_val > 45:
            return "🌬️"  # Great wind, clear sky
        return "☀️"  # Sunny
    except Exception:
        return "🌤️"
