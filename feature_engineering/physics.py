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
    arctan,
    clip,
    cos,
    exp,
    float32,
    log,
    pi,
    radians,
    sin,
    sqrt,
)
from pandas import (
    DataFrame,
    Series,
)

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



class PhysicsFeaturesMixin:
    def add_physics(df: DataFrame, lat: float) -> DataFrame:
        """
        Includes:
        - Magnus-Tetens (e_sat, e_act, VPD, dew depression)
        - Specific humidity (q)
        - Wet thermometer temperature (Stull)
        - K-Index (George) for convective instability
        - Wind components (u, v) and shear
        - Theta-E (potential equivalent temperature)
        - PWAT (precipitable water)
        - Thermal Advection
        - Relative Vorticity
        - NAO proxy

        Includes:
        - Magnus-Tetens (e_sat, e_act, VPD, dew depression)
        - Specific humidity (q)
        - Wet thermometer temperature (Stull)
        - K-Index (George) for convective instability
        - Wind components (u, v) and shear
        - Theta-E (potential equivalent temperature)
        - PWAT (precipitable water)
        - Thermal Advection
        - Relative Vorticity
        - NAO proxy

        Args:
            df: DataFrame with basic weather variables
            lat: Latitude for geographic calculations

        Returns:
            DataFrame enriched with all physical features
        """
        # Physical constants
        a, b = 17.27, 237.3
        R = 6371000.0  # Earth radius in meters
        lat_rad = radians(lat)
        cos_lat = cos(lat_rad)

        # Conversion degrees -> meters
        m_per_deg_lat = (pi / 180) * R
        m_per_deg_lon = (pi / 180) * R * cos_lat

        # =========================================================================
        # 1. MAGNUS-TETENS: Saturated vapour pressure and VPD
        # =========================================================================
        if "temp" in df.columns and "hum" in df.columns:
            e_sat = 0.61078 * exp((17.27 * df["temp"]) / (df["temp"] + 237.3))
            e_act = e_sat * (df["hum"] / 100.0)
            df["vpd"] = (e_sat - e_act).clip(lower=0).astype(float32)
            if "dew" in df.columns:
                df["dew_depression"] = (df["temp"] - df["dew"]).astype(float32)

        # =========================================================================
        # 3. instability INDEX (difference T_Sol - T_850hPa)
        # =========================================================================
        if "press" in df.columns and "hum" in df.columns and "temp" in df.columns:
            e_sat = 0.61078 * exp((17.27 * df["temp"]) / (df["temp"] + 237.3))
            e_act = e_sat * (df["hum"] / 100.0)
            df["specific_humidity"] = (
                0.622 * e_act / (df["press"] - 0.378 * e_act)
            ).astype(float32)

        # =========================================================================
        # 3. instability INDEX (difference T_Sol - T_850hPa)
        # =========================================================================
        if "t850" in df.columns and "temp" in df.columns:
            df["instability_index"] = (df["temp"] - df["t850"]).astype(float32)

        # =========================================================================
        # 4. WET BULB temperature (Tw) - Stull formula
        # =========================================================================
        if "temp" in df.columns and "hum" in df.columns:
            T = df["temp"]
            RH = df["hum"]
            df["tw"] = (
                T * arctan(0.151977 * sqrt(clip(RH + 8.313659, 0, None)))
                + arctan(T + RH)
                - arctan(RH - 1.676331)
                + 0.00391838 * (clip(RH, 0, None) ** 1.5) * arctan(0.023101 * RH)
                - 4.686035
            ).astype(float32)

        # =========================================================================
        # 5. PRESSURE TRENDS (3h and 6h)
        # =========================================================================
        if "press" in df.columns:
            df["press_trend_3h"] = df["press"].diff(3).ffill(limit=3).astype(float32)
            df["press_trend_6h"] = df["press"].diff(6).ffill(limit=3).astype(float32)

        # =========================================================================
        # 6. K-INDEX (George's Index) - Convective instability
        # =========================================================================
        if all(c in df.columns for c in ["t850", "t500", "t700", "h850", "h700"]):
            # Dew point at 850 hPa via Magnus-Tetens reversed
            alpha_850 = ((a * df["t850"]) / (b + df["t850"])) + log(
                clip(df["h850"], 0.1, None) / 100.0
            )
            td850 = (b * alpha_850) / (a - alpha_850)

            # Dew point at 700 hPa
            alpha_700 = ((a * df["t700"]) / (b + df["t700"])) + log(
                clip(df["h700"], 0.1, None) / 100.0
            )
            td700 = (b * alpha_700) / (a - alpha_700)

            # K-Index = (T850 - T500) + Td850 - (T700 - Td700)
            df["k_index"] = (df["t850"] - df["t500"]) + td850 - (df["t700"] - td700)
            df["k_index"] = df["k_index"].astype(float32)

        # =========================================================================
        # 7. WIND: U/V components and shear
        # =========================================================================
        if "wind" in df.columns and "wind_dir" in df.columns:
            rad = radians(df["wind_dir"])

            # Cyclic Encoding of Direction
            df["wind_dir_sin"] = sin(rad).astype(float32)
            df["wind_dir_cos"] = cos(rad).astype(float32)

            # Vector decomposition (weather convention)
            df["wind_u"] = (-df["wind"] * sin(rad)).astype(float32)
            df["wind_v"] = (-df["wind"] * cos(rad)).astype(float32)

        # Vertical wind shear (jet stream - surface)
        if "jet_wind" in df.columns and "wind" in df.columns:
            df["wind_shear"] = clip(df["jet_wind"] - df["wind"], 0, None).astype(
                float32
            )

        # =========================================================================
        # 8. THETA-E (Equivalent Potential Temperature)
        # =========================================================================
        if all(c in df.columns for c in ["temp", "dew", "press", "hum"]):
            e = 6.11 * 10 ** (7.5 * df["dew"] / (237.3 + df["dew"]))
            w = 622 * e / (df["press"] - e)
            tlcl = (
                1.0
                / (
                    1.0 / (df["temp"] + 273.15 - 55)
                    - log(clip(df["hum"], 0.01, None) / 100.0) / 2840
                )
                + 55
            )
            df["theta_e"] = (
                (df["temp"] + 273.15)
                * (1000 / df["press"]) ** 0.2854
                * exp((3.376 / tlcl - 0.00254) * w * (1 + 0.81e-3 * w))
            ).astype(float32)

        # =========================================================================
        # 9. APPROXIMATE PWAT (Precipitable Water)
        # =========================================================================
        levels = [1000, 850, 700, 500]
        # Initialized as a Series (not a Python scalar) so that
        # .astype(float32) works even if no level pair is
        # available in df (q_sum then remains a series of zeros).
        q_sum = Series(0.0, index=df.index)

        for i in range(len(levels) - 1):
            p1, p2 = levels[i], levels[i + 1]
            h_col1, h_col2 = f"h{p1}", f"h{p2}"
            t_col1, t_col2 = f"t{p1}", f"t{p2}"

            if all(c in df.columns for c in [h_col1, h_col2, t_col1, t_col2]):
                for t_c, h_c, p_val in [(t_col1, h_col1, p1), (t_col2, h_col2, p2)]:
                    es = 0.6112 * exp((17.67 * df[t_c]) / (df[t_c] + 243.5))
                    e = es * (clip(df[h_c], 0.1, None) / 100.0)
                    q = (0.622 * e) / (p_val / 10 - 0.378 * e)
                    q_sum = q_sum + q * (p1 - p2) * 100

        df["pwat_approx"] = (q_sum / 9.81).astype(float32)

        # =========================================================================
        # 10. THERMAL ADVECTION
        # =========================================================================
        if "n_syn_temp" in df.columns and "wind_u" in df.columns:
            offset_syn = 9.0
            dist_x = 2 * offset_syn * m_per_deg_lon
            dist_y = 2 * offset_syn * m_per_deg_lat

            dT_dx = (df["e_syn_temp"] - df["w_syn_temp"]) / dist_x
            dT_dy = (df["n_syn_temp"] - df["s_syn_temp"]) / dist_y

            u_ms = df["wind_u"] / 3.6
            v_ms = df["wind_v"] / 3.6
            df["thermal_advection"] = -3600 * (u_ms * dT_dx + v_ms * dT_dy).astype(
                float32
            )

        # coriolis parameter
        df["coriolis_param"] = 2 * 7.2921e-5 * sin(lat_rad)

        # =========================================================================
        # 11. GEOPOTENTIAL THICKNESS 500-1000HPA
        # =========================================================================
        if "geo_500" in df.columns and "geo_1000" in df.columns:
            df["thickness_500_1000"] = (df["geo_500"] - df["geo_1000"]).astype(float32)

        # =========================================================================
        # 12. PWAT VERSION 2 (trapezoidal integration)
        # =========================================================================
        q_sum2 = Series(0.0, index=df.index)
        for i in range(len(levels) - 1):
            p1, p2 = levels[i], levels[i + 1]
            if all(f"t{p}" in df.columns for p in [p1, p2]) and all(
                f"h{p}" in df.columns for p in [p1, p2]
            ):
                q_p = []
                for p in [p1, p2]:
                    es_p = 6.11 * 10 ** (7.5 * df[f"t{p}"] / (237.3 + df[f"t{p}"]))
                    e_p = es_p * (clip(df[f"h{p}"], 1, None) / 100.0)
                    q_p.append(0.622 * e_p / p)
                q_sum2 = q_sum2 + 0.5 * (q_p[0] + q_p[1]) * (p1 - p2) * 100

        df["precipitable_water"] = (q_sum2 / 9.81).astype(float32)

        # =========================================================================
        # 13. VERTICAL THERMAL GRADIENTS (lapse rate)
        # =========================================================================
        if "t1000" in df.columns and "t500" in df.columns and "t850" in df.columns:
            df["lapse_rate_low"] = (df["t1000"] - df["t850"]) / 1.5
            df["mid_troposphere_stability"] = (df["t850"] - df["t500"]) / 3.5

        # =========================================================================
        # 14. RELATIVE STRENGTH (regional scale)
        # =========================================================================
        if all(
            f"{d}_v" in df.columns for d in ["e_reg", "w_reg", "n_reg", "s_reg"]
        ) and all(f"{d}_u" in df.columns for d in ["e_reg", "w_reg", "n_reg", "s_reg"]):
            offset_reg = 2.2
            dx_reg = 2 * offset_reg * m_per_deg_lon
            dy_reg = 2 * offset_reg * m_per_deg_lat

            dv_dx = (df["e_reg_v"] - df["w_reg_v"]) / dx_reg
            du_dy = (df["n_reg_u"] - df["s_reg_u"]) / dy_reg
            vort = (dv_dx - du_dy).astype(float32)
            df["relative_vorticity"] = vort * 1e5

        # =========================================================================
        # 15. HORIZONTAL PRESSURE GRADIENTS
        # =========================================================================
        if all(
            f"{d}_press" in df.columns for d in ["n_reg", "s_reg", "e_reg", "w_reg"]
        ):
            df["pres_grad_x"] = df["e_reg_press"] - df["w_reg_press"]
            df["pres_grad_y"] = df["n_reg_press"] - df["s_reg_press"]
            df["wind_forcing"] = sqrt(df["pres_grad_x"] ** 2 + df["pres_grad_y"] ** 2)
            df["pressure_tendency_magnitude"] = sqrt(
                df["pres_grad_x"] ** 2 + df["pres_grad_y"] ** 2
            )

        # =========================================================================
        # 16. PROXY NAO (North Atlantic Oscillation)
        # =========================================================================
        if "n_hemi_press" in df.columns and "s_hemi_press" in df.columns:
            df["nao_proxy"] = (df["s_hemi_press"] - df["n_hemi_press"]).astype(float32)
            df["nao_trend"] = df["nao_proxy"].diff(6).ffill(limit=3).astype(float32)

        return df


class FeatureEngineer(PhysicsFeaturesMixin):
    """Historical compatibility for monolithic calls."""

