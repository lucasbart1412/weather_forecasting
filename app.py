"""Imports from the new inference layer"""

from datetime import date, datetime

import pandas as pd
import streamlit as st

from feature_engineering.feature_engineering import FeatureEngineer

# Imports from the new inference layer
from inference import BatchWeatherPredictor, FixedDataFetcher

# PART 1 — LOADING & CACHE
from infrastructure.config import WeatherConfig, logger
from infrastructure.geo_api import LocationManager
from presentation.dashboard_helpers import (
    build_daily_aggregate,
    get_sun_times_cached,
    plot_hourly,
    safe_float,
    sunrise_sunset_,
)
from presentation.weather_icons import get_weather_icon

MODEL_DIR = WeatherConfig.BASE_DIR

if "lang" not in st.session_state:
    st.session_state.lang = "en"  # Default language

# ============================================================================
# PART 1 — LOADING & CACHE
# ============================================================================


@st.cache_resource(show_spinner="🔧 Initializing geolocation services...")
def get_location_manager() -> LocationManager:
    return LocationManager()


@st.cache_resource(
    show_spinner="🌐 Initializing the corrected Open-Meteo data client..."
)
def get_data_fetcher() -> FixedDataFetcher:
    return FixedDataFetcher()


@st.cache_resource(
    show_spinner="🧠 Loading the 9 AI models (LightGBM + stacking + MOS)..."
)
def get_predictor() -> BatchWeatherPredictor:
    predictor = BatchWeatherPredictor(MODEL_DIR)
    loaded_models = sum(len(models) for models in predictor.models.values())
    if loaded_models == 0:
        raise RuntimeError(
            f"No weather model found in the expected directory: {MODEL_DIR}"
        )
    return predictor


@st.cache_data(ttl=3600, show_spinner="📡 Retrieving current conditions...")
def load_live_conditions(
    city: str, _predictor: BatchWeatherPredictor, hour_bucket: str
):
    """Retrieves and prepares live data for a city. Never raise an exception."""
    lm = get_location_manager()
    fetcher = get_data_fetcher()
    try:
        geo = lm.get_geo_grid(city)
        if not geo:
            return {"error": f"City not found: {city}."}

        live_grid = fetcher.fetch_data(geo, is_archive=False)
        if live_grid is None or live_grid.empty:
            return {
                "error": "No live weather data is available for this location."
            }

        climatology_ref = _predictor.meta.get("clima", {}) if _predictor is not None else {}
        df_now_result = FeatureEngineer.prepare(
            live_grid,
            geo["tz"],
            is_live=True,
            climatology_ref=climatology_ref,
            lat=geo["lat"],
            lon=geo["lon"],
            elevation=geo.get("elevation", 0),
        )
        df_now = df_now_result[0] if isinstance(df_now_result, tuple) else df_now_result

        # DataFetcher normalizes dates in naive UTC before returning the DataFrame.
        now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
        df_now_obs = df_now[df_now["date"] <= now_utc]
        if df_now_obs.empty:
            df_now_obs = df_now.iloc[(df_now["date"] - now_utc).abs().argsort()[:1]]

        last_row = df_now_obs.iloc[-1]
        obs = {t: last_row.get(t, 0) for t in WeatherConfig.TARGETS}
        obs["time"] = last_row["date"]

        return {
            "geo": geo,
            "df_now_obs": df_now_obs,
            "obs": obs,
            "last_row": last_row.to_dict(),
        }
    except Exception as e:
        logger.warning(f"[load_live_conditions] Failure for '{city}' : {e}")
        return {
            "error": "Temporary error while retrieving data. Please try again shortly."
        }


@st.cache_data(
    ttl=1800, show_spinner="⚙️ Calculating the 7-day forecast (batch inference)..."
)
def compute_predictions(
    city: str,
    hour_bucket: str,
    _predictor: BatchWeatherPredictor,
    _df_now_obs: pd.DataFrame,
    _obs: dict,
    _geo: dict,
) -> pd.DataFrame:
    """Calculate the 168 hours of forecast in a single vectorized pass (caching)."""
    try:
        horizons = list(range(1, 169))
        df_pred = _predictor.predict_batch(horizons, _df_now_obs, _obs, _geo)
        return df_pred
    except Exception as e:
        logger.warning(f"[compute_predictions] Failure for '{city}' : {e}")
        return pd.DataFrame()


# ============================================================================
# PART 2 — HIGH FIDELITY STREAMLIT INTERFACE (MRI STYLE, CLEAR THEME)
# ============================================================================

st.set_page_config(
    page_title='WeatherMaster · AI Forecasts', page_icon="🌤️", layout='wide'
)

CUSTOM_CSS = """
<style>
:root {
    --irm-blue-dark: #0b3d6b;
    --irm-blue: #1467b5;
    --irm-blue-light: #5aa7e0;
    --irm-accent: #f7931e;
    --irm-card-bg: #ffffff;
    --irm-card-border: #e3ebf3;
    --irm-text-main: #16324f;
    --irm-text-soft: #5c7086;
    --irm-page-bg: #f4f8fc;
}
.stApp { background: linear-gradient(180deg, #eaf3fb 0%, #f7fafd 40%, #ffffff 100%); }
.block-container { padding-top: 1.5rem; max-width: 1250px; }

.irm-header {
    background: linear-gradient(120deg, var(--irm-blue-dark), var(--irm-blue) 60%, var(--irm-blue-light));
    border-radius: 18px;
    padding: 1.6rem 2rem;
    box-shadow: 0 8px 24px rgba(20,103,181,0.20);
    margin-bottom: 1.2rem;
}
.irm-header h1 { color: white; font-size: 1.7rem; margin-bottom: 0.1rem; }
.irm-header p { color: #dceaf7; margin: 0; font-size: 0.9rem; }

.metric-card {
    background: var(--irm-card-bg);
    border: 1px solid var(--irm-card-border);
    border-radius: 14px;
    padding: 0.9rem 0.6rem;
    text-align: center;
    box-shadow: 0 3px 10px rgba(20,103,181,0.06);
}
.metric-card .metric-icon { font-size: 1.5rem; }
.metric-card .metric-value { font-size: 1.35rem; font-weight: 700; color: var(--irm-text-main); margin-top: 0.15rem; }
.metric-card .metric-label { font-size: 0.72rem; color: var(--irm-text-soft); text-transform: uppercase; letter-spacing: 0.04em; }

.sun-banner {
    display: flex; justify-content: center; gap: 2.2rem; margin-top: 0.8rem;
    color: var(--irm-text-soft); font-size: 0.92rem;
}

.day-card {
    background: var(--irm-card-bg);
    border-radius: 16px;
    padding: 0.9rem 0.4rem 0.4rem 0.4rem;
    text-align: center;
    border: 1px solid var(--irm-card-border);
    box-shadow: 0 3px 10px rgba(20,103,181,0.06);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.day-card:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(20,103,181,0.14); }
.day-card.selected { border: 2px solid var(--irm-blue); background: #eaf3fc; }
.day-card .day-name { color: var(--irm-blue); font-weight: 700; font-size: 0.85rem; text-transform: uppercase; }
.day-card .day-icon { font-size: 2rem; margin: 0.35rem 0; }
.day-card .day-temps { color: var(--irm-text-main); font-size: 0.95rem; font-weight: 600; }
.day-card .day-temps .max { color: #e8590c; }
.day-card .day-temps .min { color: #1467b5; }
.day-card .day-rain { color: var(--irm-text-soft); font-size: 0.75rem; margin-top: 0.2rem; }

.section-title { color: var(--irm-text-main); font-size: 1.15rem; font-weight: 700; margin: 1.3rem 0 0.6rem 0; }
.updated-caption { color: var(--irm-text-soft); font-size: 0.78rem; text-align: right; }

div[data-testid="stButton"] > button {
    border-radius: 8px;
    border: 1px solid var(--irm-card-border);
    color: var(--irm-blue);
    font-size: 0.75rem;
    padding: 0.15rem 0.4rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

WEEKDAY_LABELS = {0: "Mon.", 1: "Tue.", 2: "Wed.", 3: "Thu.", 4: "Fri.", 5: "Sat.", 6: "Sun."}

VARIABLE_OPTIONS = {
    "🌡️ Temperature": "temp",
    "🌧️ Rain & Probability": "precip",
    "💨 Wind": "wind",
    "☁️ Cloud Cover": "cloud",
    "💦 Humidity": "hum",
    "🌊 Tides": "tide",
}


def choice_control(label: str, options: list, key: str):
    """Uses st.segmented_control if available (Streamlit >= 1.36), otherwise horizontal radio fallback."""
    try:
        if hasattr(st, "segmented_control"):
            val = st.segmented_control(
                label, options=options, key=key, default=options[0]
            )
            return val if val else options[0]
    except Exception:
        pass
    return st.radio(label, options=options, horizontal=True, key=key + "_radio")




# ----------------------------------------------------------------------------
# HEADER & CITY SEARCH
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="irm-header">\n    <h1>🌤️ WeatherMaster · 7‑day AI forecasts</h1>\n    <p>LightGBM models + Stacking + MOS bias correctors — vectorized batch inference</p>\n</div>',
    unsafe_allow_html=True,
)

col_search, col_refresh = st.columns([5, 1])
with col_search:
    city = st.text_input(
        "📍 City",
        value=st.session_state.get("city", "Uccle"),
        label_visibility="collapsed",
        placeholder="Enter a city (e.g. Uccle, Paris, Brussels)",
    )
with col_refresh:
    if st.button("🔄 Refresh", width="stretch"):
        st.cache_data.clear()
        st.session_state["city"] = city

if not city or not city.strip():
    st.info('👆 Enter the name of a city to display the forecasts.')
    st.stop()

hour_bucket = datetime.now().strftime("%Y-%m-%d-%H")
try:
    predictor = get_predictor()
except (FileNotFoundError, RuntimeError) as exc:
    logger.warning("Forecast models are unavailable: %s", exc)
    st.error("Forecast models are not available in this checkout.")
    st.info(
        "Train the models first, or place the generated artifacts in the "
        f"'{MODEL_DIR.name}/' directory."
    )
    st.stop()
live = load_live_conditions(city, predictor, hour_bucket)

if live.get("error"):
    st.error(f"⚠️ {live['error']}")
    st.stop()

geo, df_now_obs, obs, last_row = (
    live["geo"],
    live["df_now_obs"],
    live["obs"],
    live["last_row"],
)

# ----------------------------------------------------------------------------
# 1. CURRENT CONDITIONS BANNER
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="section-title">📡 Current conditions</div>', unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)
metrics = [
    (c1, "🌡️", f"{safe_float(obs.get('temp')):.1f} °C", "Temperature"),
    (c2, "💧", f"{safe_float(obs.get('hum')):.0f} %", "Humidity"),
    (c3, "💨", f"{safe_float(obs.get('wind')):.0f} km/h", "Wind"),
    (c4, "🧭", f"{safe_float(obs.get('press')):.0f} hPa", "Pressure"),
]

for col, icon, value, label in metrics:
    with col:
        st.markdown(
            f"""<div class="metric-card">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>""",
            unsafe_allow_html=True,
        )

date_bucket = date.today().isoformat()
sunrise, sunset = get_sun_times_cached(
    geo["lat"], geo["lon"], date_bucket, tz_name=geo.get("tz", "UTC")
)

updated_time = "--:--"
if obs.get("time") is not None:
    try:
        observed_at = pd.Timestamp(obs["time"])
        if observed_at.tzinfo is None:
            observed_at = observed_at.tz_localize("UTC")
        updated_time = observed_at.tz_convert(geo.get("tz", "UTC")).strftime(
            "%H:%M"
        )
    except Exception as e:
        logger.warning(f"[display] Unable to convert observation time: {e}")

st.markdown(
    f"""<div class="sun-banner">
            <span>🌅 Sunrise: {sunrise[-5:] if sunrise != "N/A" else "N/A"}</span>
            <span>🌇 Sunset: {sunset[-5:] if sunset != "N/A" else "N/A"}</span>
        </div>
        <p class="updated-caption">Last updated: {updated_time}</p>""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# BATCH FORECAST CALCULATION (168 hours — single vectorized call)
# ----------------------------------------------------------------------------
df_pred = compute_predictions(city, hour_bucket, predictor, df_now_obs, obs, geo)

if df_pred is None or df_pred.empty:
    st.warning(
        '⚠️ Unable to calculate forecasts at the moment. Please try again later.'
    )
    st.stop()

df_sun = sunrise_sunset_(df_pred, geo)
daily = build_daily_aggregate(df_pred)

# ----------------------------------------------------------------------------
# 2. 7-DAY OVERVIEW (CLICKABLE CARDS)
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="section-title">📅 Overview of the next 7 days</div>',
    unsafe_allow_html=True,
)

daily_7 = pd.DataFrame()
if daily.empty:
    st.info('No daily data available at the moment.')
else:
    daily_7 = daily.head(7).reset_index(drop=True)

    if "selected_date" not in st.session_state or st.session_state[
        "selected_date"
    ] not in list(daily_7["date_only"]):
        st.session_state["selected_date"] = daily_7.iloc[0]["date_only"]

    cols = st.columns(len(daily_7))
    for i, row in daily_7.iterrows():
        with cols[i]:
            d = row["date_only"]
            weekday_label = WEEKDAY_LABELS.get(pd.Timestamp(d).weekday(), "")
            cloud_val = row.get("cloud_mean", 0)
            precip_val = row.get("precip_sum", 0)
            hum_val = row.get("hum_mean", 0)
            wind_val = row.get("wind_mean", 0)
            temp_min_val = row.get("temp_min", None)
            icon = get_weather_icon(
                cloud_val,
                precip_val,
                temp_val=temp_min_val,
                wind_val=wind_val,
                hum_val=hum_val,
            )
            tmax = safe_float(row.get("temp_max"))
            tmin = safe_float(row.get("temp_min"))
            rain_text = (
                f"🌧️ {safe_float(precip_val):.1f} mm"
                if "precip_sum" in daily_7.columns
                else ""
            )
            rose_val = safe_float(row.get("dew_point_mean", 0))
            dew_point_text = (
                f"🌡️ Dew point: {rose_val:.1f}°C"
                if "dew_point_mean" in daily_7.columns
                else ""
            )
            selected_class = (
                " selected" if d == st.session_state["selected_date"] else ""
            )
            st.markdown(
                f"""<div class="day-card{selected_class}">
                        <div class="day-name">{weekday_label} {d.strftime("%d/%m")}</div>
                        <div class="day-icon">{icon}</div>
                        <div class="day-temps"><span class="max">{tmax:.0f}°</span> / <span class="min">{tmin:.0f}°</span></div>
                        <div class="day-rain">{rain_text}</div>
                        <div style="font-size: 0.72rem; color: var(--irm-text-soft); margin-top: 0.1rem;">{dew_point_text}</div>
                    </div>""",
                unsafe_allow_html=True,
            )
            if st.button("🔍 Details", key=f"sel_day_{i}", width="stretch"):
                st.session_state["selected_date"] = d

# ----------------------------------------------------------------------------
# 4. DETAILS OF THE SELECTED DAY (CLICK ON A MAP, EVOLUTIONARY GRAPH)
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="section-title">⏱️ Next 24 hours</div>', unsafe_allow_html=True
)

choice_today_label = choice_control(
    "Variable (24h)", list(VARIABLE_OPTIONS.keys()), key="today_var"
)
var_today = VARIABLE_OPTIONS.get(choice_today_label, "temp")
df_24h = df_pred[df_pred["horizon_h"] <= 24].reset_index(drop=True)

fig_24h = plot_hourly(df_24h, var_today)
if fig_24h is not None:
    st.plotly_chart(fig_24h, width="stretch", key="fig_24h")
else:
    st.info(
        'ℹ️ This data is not currently available (target not trained or missing for this location).'
    )

# ----------------------------------------------------------------------------
# 4. DETAILS OF THE SELECTED DAY (CLICK ON A MAP, EVOLUTIONARY GRAPH)
# ----------------------------------------------------------------------------
selected_date = st.session_state.get("selected_date")
if selected_date is not None:
    selected_weekday_label = WEEKDAY_LABELS.get(pd.Timestamp(selected_date).weekday(), "")
    st.markdown(
        f'<div class="section-title">🔎 Details for {selected_weekday_label} {pd.Timestamp(selected_date).strftime("%d/%m/%Y")}</div>',
        unsafe_allow_html=True,
    )

    choice_day_label = choice_control(
        "Variable (selected day)",
        list(VARIABLE_OPTIONS.keys()),
        key="day_detail_var",
    )
    var_day = VARIABLE_OPTIONS.get(choice_day_label, "temp")
    df_day = df_pred[
        pd.to_datetime(df_pred["time"]).dt.date == selected_date
    ].reset_index(drop=True)

    fig_day = plot_hourly(df_day, var_day)
    if fig_day is not None:
        st.plotly_chart(fig_day, width="stretch", key="fig_day")
    else:
        st.info(
            'ℹ️ This data is not currently available (target not trained or missing for this location).'
        )

# ----------------------------------------------------------------------------
# 5. FULL HOURLY VIEW (7 DAYS, EVOLVING CHART)
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="section-title">📊 Full hourly view (7 days)</div>',
    unsafe_allow_html=True,
)

choice_hourly_label = choice_control(
    "Variable (7-day timetable)", list(VARIABLE_OPTIONS.keys()), key="hourly_var"
)
var_hourly = VARIABLE_OPTIONS.get(choice_hourly_label, "temp")

fig_hourly = plot_hourly(df_pred, var_hourly)
if fig_hourly is not None:
    st.plotly_chart(fig_hourly, width="stretch", key="fig_hourly")
else:
    st.info(
        'ℹ️ This data is not currently available (target not trained or missing for this location).'
    )

st.caption(
    '🔧 Batch vectorized inference (predict_batch) — the 168 hours are computed in a handful \n    of model calls grouped by block (short ≤18h / medium ≤72h / long >72h), instead of 168 sequential calls.'
)
