"""Shared helpers for the Streamlit weather dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from infrastructure.config import logger


__all__ = [
    "safe_float",
    "sunrise_sunset",
    "get_sun_times_cached",
    "sunrise_sunset_",
    "build_daily_aggregate",
    "plot_hourly",
]


def safe_float(val, default=0.0):
    try:
        f = float(val)
        return default if pd.isna(f) else f
    except Exception:
        return default


def sunrise_sunset(lat: float, lon: float, date_obj, tz_name: str = "UTC"):
    """Calculates sunrise/sunset for a location and date, respecting the local time zone."""
    if isinstance(date_obj, str) or not isinstance(date_obj, pd.Timestamp):
        date_obj = pd.Timestamp(date_obj)

    if date_obj.tzinfo is not None:
        date_part = date_obj.tz_convert("UTC").tz_localize(None)
    else:
        date_part = date_obj.tz_localize(None)

    anchor = pd.Timestamp(date_part.strftime("%Y-%m-%d"), tz="UTC")

    day_of_year = date_part.dayofyear
    declination = np.radians(
        23.45 * np.sin(2 * np.pi * (284 + day_of_year) / 365)
    )
    equation_of_time_angle = 2 * np.pi * (day_of_year - 81) / 364
    equation_of_time = (
        9.87 * np.sin(2 * equation_of_time_angle)
        - 7.53 * np.cos(equation_of_time_angle)
        - 1.5 * np.sin(equation_of_time_angle)
    )
    latitude_rad = np.radians(lat)
    sunset_altitude_rad = np.radians(-0.833)
    cos_hour_angle = (
        np.sin(sunset_altitude_rad) - np.sin(latitude_rad) * np.sin(declination)
    ) / (np.cos(latitude_rad) * np.cos(declination))
    hour_angle = np.arccos(np.clip(cos_hour_angle, -1.0, 1.0))
    hour_angle_hours = np.degrees(hour_angle) / 15.0
    solar_noon = 12.0 - lon / 15.0 - equation_of_time / 60.0
    sunrise_hour = solar_noon - hour_angle_hours
    sunset_hour = solar_noon + hour_angle_hours

    sunrise_utc = anchor + pd.to_timedelta(float(sunrise_hour) % 24, unit="h")
    sunset_utc = anchor + pd.to_timedelta(float(sunset_hour) % 24, unit="h")

    if tz_name:
        try:
            return sunrise_utc.tz_convert(tz_name), sunset_utc.tz_convert(tz_name)
        except Exception:
            return sunrise_utc, sunset_utc
    return sunrise_utc, sunset_utc


def get_sun_times_cached(lat: float, lon: float, date_obj, tz_name: str = "UTC"):
    """Formats a tuple (sunrise, sunset) into HH:MM strings suitable for display."""
    sunrise, sunset = sunrise_sunset(lat, lon, date_obj, tz_name=tz_name)
    return sunrise.strftime("%H:%M"), sunset.strftime("%H:%M")


def sunrise_sunset_(df_pred: pd.DataFrame, geo: dict) -> pd.DataFrame:
    """Adds sunrise/sunset features to the Forecast DataFrame."""
    df = df_pred.copy()
    times = pd.to_datetime(df["time"])
    tz_name = geo.get("tz", "UTC")

    if times.dt.tz is None:
        times = times.dt.tz_localize(tz_name)
    else:
        times = times.dt.tz_convert(tz_name)

    dates_uniques = times.dt.date.unique()

    sunrise_map = {}
    sunset_map = {}
    for d in dates_uniques:
        sunrise_dt, sunset_dt = sunrise_sunset(
            geo["lat"],
            geo["lon"],
            pd.Timestamp(d).tz_localize(tz_name),
            tz_name=tz_name,
        )
        sunrise_map[d] = sunrise_dt
        sunset_map[d] = sunset_dt

    df_dates = times.dt.date
    df["sunrise"] = df_dates.map(sunrise_map)
    df["sunset"] = df_dates.map(sunset_map)
    df["is_day"] = (times >= df["sunrise"]) & (times <= df["sunset"])
    return df


def build_daily_aggregate(df_pred: pd.DataFrame) -> pd.DataFrame:
    """Aggregates hourly forecasts into daily warnings/totals."""
    if df_pred is None or df_pred.empty:
        return pd.DataFrame()

    df = df_pred.copy()
    if "temp" in df.columns and "hum" in df.columns:
        t = df["temp"].astype(float)
        rh = df["hum"].astype(float).clip(1, 100)
        alpha = (17.27 * t) / (237.7 + t) + np.log(rh / 100.0)
        df["dew_point"] = (237.7 * alpha) / (17.27 - alpha)

    df["date_only"] = pd.to_datetime(df["time"]).dt.date

    agg_kwargs = {}
    if "temp" in df.columns:
        agg_kwargs["temp_max"] = ("temp", "max")
        agg_kwargs["temp_min"] = ("temp", "min")
        agg_kwargs["temp_mean"] = ("temp", "mean")
    if "dew_point" in df.columns:
        agg_kwargs["dew_point_mean"] = ("dew_point", "mean")
    if "precip" in df.columns:
        agg_kwargs["precip_sum"] = ("precip", "sum")
    if "precip_prob" in df.columns:
        agg_kwargs["precip_prob_mean"] = ("precip_prob", "mean")
        agg_kwargs["precip_prob_max"] = ("precip_prob", "max")
    if "wind" in df.columns:
        agg_kwargs["wind_max"] = ("wind", "max")
        agg_kwargs["wind_mean"] = ("wind", "mean")
    if "cloud" in df.columns:
        agg_kwargs["cloud_mean"] = ("cloud", "mean")
    if "hum" in df.columns:
        agg_kwargs["hum_mean"] = ("hum", "mean")
    if "tide" in df.columns and df["tide"].notna().any():
        agg_kwargs["tide_max"] = ("tide", "max")
        agg_kwargs["tide_min"] = ("tide", "min")

    if not agg_kwargs:
        return pd.DataFrame()

    try:
        daily = df.groupby("date_only").agg(**agg_kwargs).reset_index()
        return daily
    except Exception as e:
        logger.warning(f"[build_daily_aggregate] {e}")
        return pd.DataFrame()


def plot_hourly(df_pred: pd.DataFrame, variable: str):
    fig = go.Figure()

    def _find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    try:
        if (
            variable == "temp"
            and "temp" in df_pred.columns
            and df_pred["temp"].var() != 0
        ):
            fig.add_trace(
                go.Scatter(
                    x=df_pred["time"],
                    y=df_pred["t_high"],
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_pred["time"],
                    y=df_pred["t_low"],
                    fill="tonexty",
                    fillcolor="rgba(255,138,101,0.20)",
                    line=dict(width=0),
                    name="Confidence interval",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_pred["time"],
                    y=df_pred["temp"],
                    mode="lines",
                    line=dict(color="#ff6b4a", width=3),
                    name="Temperature (°C)",
                )
            )
        elif (
            variable == "precip"
            and "precip" in df_pred.columns
            and (
                df_pred["precip"].var() != 0
                or (
                    "precip_prob" in df_pred.columns
                    and df_pred["precip_prob"].var() != 0
                )
            )
        ):
            if df_pred["precip"].var() != 0:
                fig.add_trace(
                    go.Bar(
                        x=df_pred["time"],
                        y=df_pred["precip"],
                        name="Precipitation (mm)",
                        marker_color="#4a90d9",
                        yaxis="y1",
                    )
                )
            if (
                "precip_prob" in df_pred.columns
                and df_pred["precip_prob"].var() != 0
            ):
                fig.add_trace(
                    go.Scatter(
                        x=df_pred["time"],
                        y=df_pred["precip_prob"],
                        name="Probability (%)",
                        line=dict(color="#c77dff", width=2),
                        yaxis="y2",
                    )
                )
            fig.update_layout(
                yaxis=dict(title="mm"),
                yaxis2=dict(
                    title="%",
                    overlaying="y",
                    side="right",
                    range=[0, 100],
                    showgrid=False,
                ),
            )
        elif (
            variable == "wind"
            and "wind" in df_pred.columns
            and df_pred["wind"].var() != 0
        ):
            fig.add_trace(
                go.Scatter(
                    x=df_pred["time"],
                    y=df_pred["wind"],
                    mode="lines",
                    line=dict(color="#2ec4b6", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(46,196,182,0.15)",
                    name="Wind (km/h)",
                )
            )
            if "gusts" in df_pred.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_pred["time"],
                        y=df_pred["gusts"],
                        mode="lines",
                        line=dict(color="#95a5a6", width=1, dash="dot"),
                        name="gusts (km/h)",
                    )
                )
        elif variable == "hum":
            col = _find_col(
                df_pred,
                [
                    "hum",
                    "hum_mean",
                    "relative_humidity_2m",
                    "relative_humidity_2m_era5_land",
                    "relative_humidity_2m_era5",
                ],
            )
            if col and df_pred[col].notna().any() and df_pred[col].var() != 0:
                fig.add_trace(
                    go.Scatter(
                        x=df_pred["time"],
                        y=df_pred[col].astype(float),
                        mode="lines",
                        line=dict(color="#3a7bd5", width=2),
                        fill="tozeroy",
                        fillcolor="rgba(58,123,213,0.12)",
                        name="Humidity (%)",
                    )
                )
                fig.update_layout(yaxis=dict(range=[0, 100], title="%"))
            else:
                return None
        elif (
            variable == "cloud"
            and "cloud" in df_pred.columns
            and df_pred["cloud"].var() != 0
        ):
            fig.add_trace(
                go.Scatter(
                    x=df_pred["time"],
                    y=df_pred["cloud"],
                    mode="lines",
                    line=dict(color="#8d99ae", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(141,153,174,0.20)",
                    name="Cloud cover (%)",
                )
            )
            fig.update_layout(yaxis=dict(range=[0, 100]))
        elif (
            variable == "tide"
            and "tide" in df_pred.columns
            and df_pred["tide"].notna().any()
            and df_pred["tide"].var() != 0
        ):
            fig.add_trace(
                go.Scatter(
                    x=df_pred["time"],
                    y=df_pred["tide"],
                    mode="lines",
                    line=dict(color="#457b9d", width=2),
                    name="Tide (m)",
                )
            )
        else:
            return None
    except Exception as e:
        logger.warning(f"[plot_hourly] {e}")
        return None

    if not fig.data:
        return None

    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#16324f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
