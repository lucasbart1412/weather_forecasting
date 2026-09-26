"""
================================================================================
utils_features.py - Feature Engineering and Data Services
================================================================================
Contains:
- LocationManager: Geocoding, geographic cache, elevation
- DataFetcher: Open-Meteo API calls (forecast, archive, marine)
- FeatureEngineer: Vectorized atmospheric physics (Magnus-Tetens, K-Index, etc.)

All functions are optimized in NumPy vectorization and use
docstrings in Google format.
"""

from typing import Any

from numpy import (
    arcsin,
    cos,
    degrees,
    float32,
    radians,
    sin,
)
from pandas import (
    DataFrame,
    Series,
    concat,
)
from requests import get

try:
    from fake_useragent import UserAgent

    HAS_FAKE_UA = True
except ImportError:
    HAS_FAKE_UA = False

try:
    from timezonefinder import TimezoneFinder

    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    from geopy.geocoders import Nominatim

    HAS_GEOPY = True
except ImportError:
    HAS_GEOPY = False

# Imports from the infrastructure layer.
from infrastructure.config import logger


class FeatureEngineer:
    """Historical compatibility with monolithic API."""



class AstronomyFeaturesMixin:
    def get_solar_elev_vectorized(lat: float, lon: float, dt_series: Series) -> Series:
        """
        Returns:
            Series with solar elevation in degrees (float32)

        Uses standard astronomical formulas to calculate
        the position of the sun at each moment.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            dt_series: Panda timestamp series

        Returns:
            Series with solar elevation in degrees (float32)
        """
        day_of_year = dt_series.dt.dayofyear
        hour_frac = dt_series.dt.hour + dt_series.dt.minute / 60.0

        # Solar variant (simplified Cooper formula)
        declination = 23.45 * sin(radians(360 * (day_of_year - 81) / 365.25))

        # Hour angle (15° per hour)
        hour_angle = 15 * (hour_frac - 12)

        lat_rad = radians(lat)
        dec_rad = radians(declination)
        ha_rad = radians(hour_angle)

        # Solar Elevation Formula
        elev = arcsin(
            sin(lat_rad) * sin(dec_rad) + cos(lat_rad) * cos(dec_rad) * cos(ha_rad)
        )

        return degrees(elev).astype(float32)

    @staticmethod
    def add_tide_physics(df: DataFrame) -> DataFrame:
        """
        Adds tidal features (lags).

        Args:
            df: DataFrame with 'tide' column

        Returns:
            DataFrame with tidal lags added
        """
        try:
            if df["tide"].isna().all():
                logger.warning('All tide values are missing.')
                df.drop(columns=["tide"], inplace=True)
                return df

            # Vectorized creation of lags
            lags_dict = {f"tide_lag_{i}h": df["tide"].shift(i) for i in range(1, 49)}
            df = concat([df, DataFrame(lags_dict, index=df.index)], axis=1)
            return df

        except KeyError:
            logger.warning("The column 'tide' is missing.")
            return df

    @staticmethod
    def get_sun_times(lat: float, lon: float, date_obj: Any) -> tuple[str, str]:
        """
        Recovers sunrise/sunset times via Open-Meteo API.

        Args:
            lat: Latitude
            lon: Longitude
            date_obj: Reference date

        Returns:
            Tuple (sunrise, sunset) in string format gold ("N/A", "N/A")
        """
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=sunrise,sunset&timezone=auto&forecast_days=7"
        try:
            r = get(url).json()
            return r["daily"]["sunrise"][0], r["daily"]["sunset"][0]
        except Exception:
            return "N/A", "N/A"


class FeatureEngineer(AstronomyFeaturesMixin):
    """Historical compatibility for monolithic calls."""

