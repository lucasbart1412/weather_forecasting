"""Weather data retrieval via Open-Meteo APIs."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import StringIO
from typing import Any

from pandas import DataFrame, Timedelta, date_range, read_csv, to_datetime
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from fake_useragent import UserAgent

    HAS_FAKE_UA = True
except ImportError:
    HAS_FAKE_UA = False

from infrastructure.config import WeatherConfig, logger, trigger_optional_vpn


class DataFetcher:
    """Supports:
    - Archive mode (ERA5, 2010 to present)
    - Forecast mode (real-time forecast)
    - Navy fashion (tides)
    - HTTP connection pool optimized with retry

    Supports:
    - Archive mode (ERA5, 2010 to present)
    - Forecast mode (real-time forecast)
    - Navy fashion (tides)
    - HTTP connection pool optimized with retry

    Attributes:
        session: HTTP session with connection pool
        ua: User-Rotating Agent (optional)"""

    def __init__(self) -> None:
        """Initializes the DataFetcher with HTTP connection pool."""
        self.session = Session()

        # Connection pool optimized for multithreading
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=Retry(
                total=5,
                backoff_factor=2,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET"],
            ),
        )
        self.session.mount("https://", adapter)

        # User-Rotating Agent (fallback if not available)
        if HAS_FAKE_UA:
            try:
                self.ua = UserAgent()
            except Exception:
                self.ua = None
        else:
            self.ua = None

    def _fetch_single_point(
        self, name: str, lat: float, lon: float, geo: dict[str, Any], is_archive: bool
    ) -> DataFrame | None:
        """Retrieves data for a single geographic point.

        Args:
            name: Point name (ex: 'center', 'n_syn')
            lat: Latitude
            lon: Longitude
            geo: Geographical Dictionary (contains tz, etc.)
            is_archive: True for historical data, False for forecast

        Returns:
            DataFrame with data or None if error"""
        base_url = (
            "https://archive-api.open-meteo.com/v1/archive"
            if is_archive
            else "https://api.open-meteo.com/v1/forecast"
        )

        def execute_query(
            hourly_vars: list[str], model_name: str
        ) -> DataFrame | None:
            """Executes an API query with retry."""
            params = {
                "latitude": str(lat),
                "longitude": str(lon),
                "hourly": ",".join(hourly_vars),
                "timezone": geo["tz"],
                "format": "csv",
            }

            if is_archive:
                params.update(
                    {
                        "models": model_name,
                        "start_date": "2010-01-01",
                        "end_date": (datetime.now() - Timedelta(days=5)).strftime(
                            "%Y-%m-%d"
                        ),
                    }
                )
            else:
                # Keep the current local day in the response. With zero
                # forecast days, Open-Meteo stops at the previous day.
                params.update({"past_days": 7, "forecast_days": 1})

            for attempt in range(5):
                try:
                    headers = {
                        "User-Agent": self.ua.random if self.ua else "Mozilla/5.0"
                    }
                    response = self.session.get(
                        base_url, params=params, headers=headers, timeout=15
                    )

                    if response.status_code == 200:
                        lines = response.text.strip().split("\n")
                        idx = next(
                            i
                            for i, line in enumerate(lines)
                            if line.lower().startswith("time")
                        )
                        df = read_csv(StringIO("\n".join(lines[idx:])))

                        # Header Cleanup
                        df.columns = [
                            c.split(" ")[0].strip().lower() for c in df.columns
                        ]
                        df = df.rename(columns={"time": "date"})

                        # Applying Global Mapping
                        rename_dict = {
                            k.lower(): v for k, v in WeatherConfig.MAPPING_VARS.items()
                        }
                        df = df.rename(columns=rename_dict)

                        # Elimination of duplicate columns
                        df = df.loc[:, ~df.columns.duplicated()]
                        return df

                    elif response.status_code == 400:
                        trigger_optional_vpn()
                        logger.error(
                            f"API error 400 for {name} ({model_name}) : {response.text[:200]}"
                        )
                        break
                    else:
                        trigger_optional_vpn()
                        logger.warning(
                            f"Response {response.status_code} for {name}. Attempt {attempt + 1}/5."
                        )

                except Exception as e:
                    trigger_optional_vpn()
                    logger.error(
                        f"Request exception {name}: {str(e)[:100]}. Attempt {attempt + 1}/5."
                    )

            return None

        # Weather model selection
        if is_archive and (name == "center" or "loc" in name):
            return execute_query(WeatherConfig.API_VARS, "era5")
        else:
            model_default = "era5" if is_archive else "best_match"
            return execute_query(WeatherConfig.API_VARS, model_default)

    def fetch_marine_data(
        self, lat: float, lon: float, is_archive: bool = True
    ) -> DataFrame:
        """Recovers the height of the water (tides) via the Marine API.

        Args:
            lat: Latitude
            lon: Longitude
            is_archive: True for historical data

        Returns:
            DataFrame with columns ['date', 'tide']"""
        if is_archive:
            logger.info(
                'Archive mode: Tide data ignored (not supported for 15 years).'
            )
            return DataFrame(columns=["date", "tide"])

        marine_params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "ocean_tide",
            "timezone": "auto",
        }

        try:
            resp = self.session.get(
                WeatherConfig.MARINE_API_URL, params=marine_params, timeout=15
            )
            data = resp.json()

            if "hourly" in data:
                df = DataFrame(
                    {
                        "date": to_datetime(data["hourly"]["time"]),
                        "tide": data["hourly"]["ocean_tide"],
                    }
                )
                df["date"] = df["date"].dt.tz_localize(None)
                return df
            else:
                logger.warning('Marine data not available for these coordinates.')
                return DataFrame(columns=["date", "tide"])

        except Exception as e:
            logger.error(f"Marine API Error: {e}")
            return DataFrame(columns=["date", "tide"])

    def fetch_data(self, geo: dict[str, Any], is_archive: bool = True) -> DataFrame:
        """Retrieves and merges data from all points in the grid.

        Args:
            geo: Geographical Dictionary (output of LocationManager.get_geo_grid)
            is_archive: True for historical data

        Returns:
            DataFrame merged with all variables from all points

        Raises:
            ValueError: If the central point could not be retrieved"""
        data_points: dict[str, DataFrame] = {}
        max_w = 2 if is_archive else 6

        with ThreadPoolExecutor(max_workers=max_w) as executor:
            futures = {
                executor.submit(
                    self._fetch_single_point, n, p[0], p[1], geo, is_archive
                ): n
                for n, p in geo["points"].items()
            }

            for f in as_completed(futures):
                name = futures[f]
                df = f.result()
                if df is not None:
                    if name != "center":
                        df.columns = [
                            f"{name}_{c}" if c != "date" else "date" for c in df.columns
                        ]
                    data_points[name] = df
                    logger.info(f"✅ {name} : Retrieval completed")
                else:
                    logger.error(f"❌ {name} : Retrieval failed")

        if "center" not in data_points:
            raise ValueError(
                'Critical error: The central point data could not be retrieved.'
            )

        # Merge on 'date' column
        final_df = data_points["center"].set_index("date")
        for name, df_point in data_points.items():
            if name != "center":
                df_to_join = df_point.set_index("date")
                final_df = final_df.join(df_to_join, how="left")

        final_df = final_df.reset_index()
        final_df["date"] = to_datetime(final_df["date"])

        # Conversion timezone
        if final_df["date"].dt.tz is None:
            final_df["date"] = (
                final_df["date"].dt.tz_localize("UTC").dt.tz_convert(geo["tz"])
            )
        else:
            final_df["date"] = final_df["date"].dt.tz_convert(geo["tz"])

        final_df["date"] = (
            to_datetime(final_df["date"]).dt.tz_convert("UTC").dt.tz_localize(None)
        )
        final_df = (
            final_df.drop_duplicates(subset=["date"]).set_index("date").sort_index()
        )

        # Full hourly re-indexing
        full_range = date_range(
            start=final_df.index.min(), end=final_df.index.max(), freq="h"
        )
        final_df = final_df.reindex(full_range)

        # Interpolation and Filling
        final_df = final_df.interpolate(method="linear", limit=2)
        final_df = final_df.ffill(limit=1)

        # Removing 100% empty columns
        empty_cols = final_df.columns[final_df.isna().all()]
        if len(empty_cols) > 0:
            logger.info(
                f"Removal of unsupported API columns: {empty_cols.tolist()}"
            )
            final_df = final_df.drop(columns=empty_cols)

        final_df = final_df.ffill(limit=3)
        final_df = final_df.reset_index().rename(columns={"index": "date"})

        return final_df
