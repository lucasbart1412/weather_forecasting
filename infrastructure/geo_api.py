"""
Geolocation and altitude services.
"""

import json
from time import sleep
from typing import Any

from requests import get

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

from .config import WeatherConfig, logger


class LocationManager:
    """
    Location Manager with persistent JSON cache.

    Responsible for:
    - Geocoding via Nominatim (OpenStreetMap)
    - Elevation recovery via Open-Meteo API
    - Creation of multi-scale grids for synoptic features
    - JSON cache to avoid repeated requests

    Attributes:
        tf: TimezoneFinder to detect time zones
        geolocator: Nominatim for geocoding
        cache: Cache dictionary {city: data}
    """

    def __init__(self) -> None:
        """Altitude retrieves via Open-Meteo API (free, no key required)."""
        self.tf = TimezoneFinder() if HAS_TF else None
        self.geolocator = (
            Nominatim(user_agent="WeatherMaster") if HAS_GEOPY else None
        )
        WeatherConfig.BASE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, dict[str, Any]] = self._load_cache()

    def _get_elevation(self, lat: float, lon: float) -> float:
        """
        Altitude retrieves via Open-Meteo API (free, no key required).

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            Altitude in meters (0 if error)
        """
        try:
            url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
            response = get(url, timeout=10)
            if response.status_code == 200:
                return response.json().get("elevation", [0])[0]
        except Exception as e:
            logger.warning(f"Unable to retrieve altitude: {e}")
        return 0

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        """
        Loads the cache from the JSON file.

        Returns:&#10; Dictionary cache (empty if file is missing)
        """
        if WeatherConfig.GEO_CACHE_FILE.exists():
            try:
                with open(WeatherConfig.GEO_CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        """Save the cache to the JSON file."""
        try:
            with open(WeatherConfig.GEO_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Unable to save the cache: {e}")

    def get_geo_grid(self, city: str) -> dict[str, Any] | None:
        """
        Gets or creates a multi-scale geographic grid for a city.

        The grid includes:
        - Central point
        - North/South/East/West points at 5 scales (0.2°, 2.2°, 9°, 20°, 40°)
        - Time zone
        - Altitude

        Args:
            city: City name

        Returns:
            Dictionary containing the grid or None if error
        """
        city_key = city.lower().strip()
        if city_key in self.cache:
            return self.cache[city_key]

        try:
            sleep(1.2)  # Compliance with Nominatim limits
            loc = self.geolocator.geocode(city)
            if not loc:
                return None

            elev = self._get_elevation(loc.latitude, loc.longitude)

            # Multi-scale grid for synoptic prediction
            scales = [
                ("loc", 0.2),  # Local scale (micro-climate)
                ("reg", 2.2),  # Regional
                ("syn", 9.0),  # Synoptic scale
                ("macro", 20.0),  # Macro scale (airflow)
                ("hemi", 40.0),  # Hemispheric scale (NAO, etc.)
            ]

            points = {"center": (loc.latitude, loc.longitude)}
            for scale, offset in scales:
                # Latitude: clipping between -85 and 85° (avoid polar areas)
                points[f"n_{scale}"] = (min(85, loc.latitude + offset), loc.longitude)
                points[f"s_{scale}"] = (max(-85, loc.latitude - offset), loc.longitude)
                # Longitude: wrap-around with modulo
                points[f"e_{scale}"] = (
                    loc.latitude,
                    WeatherConfig.wrap_lon(loc.longitude + offset),
                )
                points[f"w_{scale}"] = (
                    loc.latitude,
                    WeatherConfig.wrap_lon(loc.longitude - offset),
                )

            tz = (
                self.tf.timezone_at(lng=loc.longitude, lat=loc.latitude) or "UTC"
                if self.tf
                else "UTC"
            )

            result = {
                "points": points,
                "tz": tz,
                "lat": loc.latitude,
                "lon": loc.longitude,
                "elevation": elev,
            }

            self.cache[city_key] = result
            self._save_cache()
            return result

        except Exception as e:
            logger.error(f"Geocoding error: {str(e)[:100]}")
            return None
