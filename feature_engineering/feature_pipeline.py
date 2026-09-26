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


from numpy import (
    cos,
    float32,
    pi,
    sin,
)
from pandas import (
    DataFrame,
    date_range,
    to_datetime,
)

from feature_engineering.astronomy import AstronomyFeaturesMixin
from feature_engineering.physics import PhysicsFeaturesMixin

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


class FeatureEngineer:
    """Historical compatibility with monolithic API."""



class FeaturePreparationMixin:
    def prepare(
        df_raw: DataFrame,
        tz_name: str,
        is_live: bool = False,
        climatology_ref: dict[str, dict[tuple[int, int], float]] | None = None,
        lon: float | None = None,
        lat: float | None = None,
        elevation: float = 0,
    ) -> DataFrame | tuple[DataFrame, dict[str, dict[tuple[int, int], float]]]:
        """
        Steps:
        1. Full hourly re-indexing
        2. Astronomical calculations (sun)
        3. Cyclic Encodings (Month, Hour)
        4. Climate anomalies
        5. Atmospheric Physics
        6. Lags and differences
        7. Rolling statistics
        8. Synoptic interactions

        Steps:
        1. Full hourly re-indexing
        2. Astronomical calculations (sun)
        3. Cyclic Encodings (Month, Hour)
        4. Climate anomalies
        5. Atmospheric Physics
        6. Lags and differences
        7. Rolling statistics
        8. Synoptic interactions

        Args:
            df_raw: Raw DataFrame with weather data
            tz_name: Time zone name
            is_live: True for real-time inference, False for training
            climatology_ref: Reference climatology dictionary
            lon: Longitude
            lat: Latitude
            elevation: Elevation (not currently used)

        Returns:
            If is_live=False: Tuple (Enhanced DataFrame, climatology_map)
            If is_live=True: Enhanced DataFrame
        """
        df = df_raw.copy()
        df["date"] = to_datetime(df["date"])
        df = df.sort_values("date")

        # Full hourly re-indexing
        full_range = date_range(start=df["date"].min(), end=df["date"].max(), freq="h")
        df = (
            df.set_index("date")
            .interpolate(method="linear", limit=1)
            .reset_index()
            .rename(columns={"index": "date"})
        )

        if not is_live:
            if "date" in df.columns:
                df = df.set_index("date").reindex(
                    date_range(df["date"].min(), df["date"].max(), freq="h")
                )
                df = df.ffill(limit=1).reset_index().rename(columns={"index": "date"})
        else:
            if "date" in df.columns:
                df = (
                    df.set_index("date")
                    .ffill(limit=3)
                    .reset_index()
                    .rename(columns={"index": "date"})
                )

        # =========================================================================
        # Cyclic Encodings
        # =========================================================================
        if lat is not None and lon is not None:
            df["solar_elev"] = FeatureEngineer.get_solar_elev_vectorized(
                lat, lon, df["date"]
            )

        # Cyclic Encodings
        df["month_sin"] = sin(2 * pi * df["date"].dt.month / 12)
        df["month_cos"] = cos(2 * pi * df["date"].dt.month / 12)
        df["hour_sin"] = sin(2 * pi * df["date"].dt.hour / 24).astype(float32)
        df["hour_cos"] = cos(2 * pi * df["date"].dt.hour / 24).astype(float32)

        # =========================================================================
        # CLIMATOLOGY AND ANOMALIES
        # =========================================================================
        targets_c = ["temp", "press", "hum"]

        if not is_live and climatology_ref is None:
            climatology_map = {}
            for t in targets_c:
                if t in df.columns:
                    climatology_map[t] = (
                        df.groupby([df["date"].dt.month, df["date"].dt.hour])[t]
                        .mean()
                        .to_dict()
                    )

            for t in targets_c:
                if t in df.columns and t in climatology_map:
                    df[f"{t}_norm"] = [
                        climatology_map[t].get((m, h), df[t].mean())
                        for m, h in zip(df["date"].dt.month, df["date"].dt.hour)
                    ]
                    df[f"{t}_anom"] = (df[t] - df[f"{t}_norm"]).astype(float32)
        else:
            climatology_map = climatology_ref if climatology_ref else {}
            for t in targets_c:
                if t in df.columns:
                    if t in climatology_map:
                        df[f"{t}_norm"] = [
                            climatology_map[t].get((m, h), 0)
                            for m, h in zip(df["date"].dt.month, df["date"].dt.hour)
                        ]
                    else:
                        df[f"{t}_norm"] = 0
                    df[f"{t}_anom"] = (df[t] - df[f"{t}_norm"]).astype(float32)

        # =========================================================================
        # ATMOSPHERIC PHYSICS
        # =========================================================================
        if lat is not None:
            df = FeatureEngineer.add_physics(df, lat=lat)

        # =========================================================================
        # LAGS AND DIFFERENCES
        # =========================================================================
        lags = [1, 2, 3, 6, 12, 24, 48]
        for col in ["temp", "press", "hum", "wind"]:
            if col in df.columns:
                for lag in lags:
                    df[f"{col}_lag_{lag}h"] = df[col].shift(lag).astype(float32)
                    if lag <= 6:
                        df[f"{col}_diff_{lag}h"] = (
                            df[col] - df[f"{col}_lag_{lag}h"]
                        ).astype(float32)

        # =========================================================================
        # DELETION OF raw SYNOPTIC COLUMNS (after delta calculation)
        # =========================================================================
        syn_cols = [
            c for c in df.columns if any(s in c for s in ["syn_", "macro_", "hemi_"])
        ]
        for col in syn_cols:
            base_var = col.split("_")[-1]
            if base_var in df.columns:
                df[f"{col}_delta"] = (df[col] - df[base_var]).astype(float32)
            df.drop(columns=[col], inplace=True)

        # =========================================================================
        # ROLLING STATISTICS
        # =========================================================================
        windows = [6, 24, 72]
        for w in windows:
            if "temp" in df.columns:
                df[f"temp_roll_mean_{w}h"] = (
                    df["temp"].rolling(window=w, min_periods=w).mean().astype(float32)
                )

        # =========================================================================
        # SYNOPTIC INTERACTIONS (Spatial scales)
        # =========================================================================
        syn_cols = [
            c for c in df.columns if any(s in c for s in ["reg_", "syn_", "macro_"])
        ]
        for col in syn_cols:
            base_var_name = col.split("_", 2)[-1]
            if base_var_name in df.columns:
                df[f"{col}_delta_local"] = (df[col] - df[base_var_name]).astype(float32)
            else:
                df[f"{col}_delta_local"] = 0.0
                df[f"{col}_delta_local"] = df[f"{col}_delta_local"].astype(float32)

        # Final chronological sorting
        df = df.sort_values("date").reset_index(drop=True)

        return (df, climatology_map) if not is_live else df


class FeatureEngineer(AstronomyFeaturesMixin, PhysicsFeaturesMixin):
    """Historical compatibility for monolithic calls."""


