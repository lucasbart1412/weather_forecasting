"""--- CORRECTION OF MISSING TARGET ---"""

import gc
from pathlib import Path

import pandas as pd
from numpy import cos, pi, sin
from pandas import Timedelta

from correlation_plotter.plotting import plot_top_correlations
from feature_engineering.feature_engineering import FeatureEngineer

# Imports from new refactored modules
from infrastructure.geo_api import LocationManager


def analyze_and_plot_correlations(
    path_parquet, city, target_base_name, horizon, exclude_keywords
):
    """--- CORRECTION OF MISSING TARGET ---"""
    print(f"[1/4] Retrieving coordinates for '{city}'...")
    loc_mgr = LocationManager()
    geo = loc_mgr.get_geo_grid(city)
    if not geo:
        print(f"❌ Unable to find coordinates for the city: {city}")
        return
    lat, lon = geo["lat"], geo["lon"]
    print(f"-> {city} found : Lat={lat:.4f}, Lon={lon:.4f}")

    print('[2/4] Partial loading of the original dataset (Parquet format)...')
    try:
        df_raw = pd.read_parquet(path_parquet)
    except Exception as e:
        print(f"❌ Error while loading the parquet file: {e}")
        return

    # --- CORRECTION OF MISSING TARGET ---
    target_col_generated = f"target_{target_base_name}_h{horizon}"
    if target_col_generated not in df_raw.columns:
        print(
            f"💡 The target column '{target_col_generated}' is missing. Creating by temporal shift (shift)..."
        )
        if target_base_name not in df_raw.columns:
            print(
                f"❌ Unable to continue: the base variable '{target_base_name}' does not exist in the dataset."
            )
            del df_raw
            gc.collect()
            return

        # The target is recreated exactly as in the training logic:
        # We shift up (in the future) by 'horizon' lines
        df_raw[target_col_generated] = df_raw[target_base_name].shift(-horizon)

    # Backup of cleaned target series and mask
    y_h = df_raw[target_col_generated].copy()
    essential_features = ["temp", "press", "hum"]
    essential_present = [f for f in essential_features if f in df_raw.columns]

    # Creating the standardized training mask
    if essential_present:
        mask = y_h.notna() & df_raw[essential_present].notna().all(axis=1)
    else:
        mask = y_h.notna()

    if not mask.any():
        print(
            f"❌ No valid data after filtering (NaN) for the horizon {horizon}h."
        )
        del df_raw, y_h
        gc.collect()
        return

    # Filtering and application of the 3-hour max forward fill
    base_features = [
        col for col in df_raw.columns if col not in ["date", target_col_generated]
    ]
    X_h = df_raw.loc[mask, base_features].copy()
    X_h = X_h.ffill(limit=3).astype("float32")
    y_h_clean = y_h.loc[mask].astype("float32")
    future_dates = df_raw.loc[mask, "date"] + Timedelta(hours=horizon)

    # Immediate release of the initial heavy raw DataFrame
    del df_raw, y_h, mask
    gc.collect()

    print('[3/4] Applying Feature Engineering (Temporal Logic & Solar)...')
    # Injection of requested features from the Train
    X_h["time_step"] = X_h.index.astype("int32")
    X_h["horizon_h"] = horizon

    # Call to the vectorized method of utils_features.py
    X_h["target_solar_elev"] = FeatureEngineer.get_solar_elev_vectorized(
        lat, lon, future_dates
    ).astype("float32")

    # Cleaning Intermediate Date Variables
    X_h["target_hour_sin"] = sin(2 * pi * future_dates.dt.hour / 24).astype("float32")
    X_h["target_hour_cos"] = cos(2 * pi * future_dates.dt.hour / 24).astype("float32")
    X_h["target_month_sin"] = sin(2 * pi * future_dates.dt.month / 12).astype("float32")
    X_h["target_month_cos"] = cos(2 * pi * future_dates.dt.month / 12).astype("float32")

    # Cleaning Intermediate Date Variables
    del future_dates
    gc.collect()

    # --- FILTERING OF EXCLUDED WORDS PASSED BY THE USER ---
    keywords_to_skip = [target_base_name.lower()] + [
        k.lower().strip() for k in exclude_keywords if k.strip()
    ]
    features_to_keep = [
        col
        for col in X_h.columns
        if not any(word in col.lower() for word in keywords_to_skip)
    ]
    X_filtered = X_h[features_to_keep]

    # --- ANTI-WARNING CLEANING ---
    # 1. Remove the 100% empty columns after filtering
    X_filtered = X_filtered.dropna(how="all", axis=1)
    # 2. Remove the constant columns (standard deviation less than or equal to 0)
    usable_cols = X_filtered.columns[X_filtered.std(axis=0) > 1e-6]
    X_filtered = X_filtered[usable_cols]

    print('[4/4] Calculating Pearson correlations...')
    correlations = X_filtered.corrwith(y_h_clean)

    # Extraction and sorting of the absolute Top 15
    top15_corr = correlations.reindex(
        correlations.abs().sort_values(ascending=False).index
    ).head(15)
    top15_corr = top15_corr.iloc[::-1]  # Inversion for orderly graph display

    # Cleaning of large calculation objects
    del X_h, X_filtered, y_h_clean, correlations
    gc.collect()

    # Graph display
    plot_top_correlations(
        top15_corr, city, horizon, target_col_generated, keywords_to_skip
    )


def main_loop():
    """While true interactive loop managing user input and cleaning memory."""
    # Replace this path with your Parquet global file
    path_parquet = Path.cwd() / "binary" / "full_grid_archive.parquet"

    print("=" * 60)
    print('Interactive & Optimized RAM Correlation Tracker')
    print("=" * 60)

    while True:
        try:
            print("--- NEW ANALYSIS (Press 'q' at any time to quit) ---")
            city = input(
                "👉 Reference city (e.g. Uccle, Paris, La Rochelle): "
            ).strip()
            if city.lower() == "q":
                break
            if not city:
                continue

            target = input(
                "👉 Variable cible de base (ex: temp, press, hum) : "
            ).strip()
            if target.lower() == "q":
                break
            if not target:
                continue

            horizon_input = input(
                "👉 Forecast horizon in hours (e.g. 1, 26, 48): "
            ).strip()
            if horizon_input.lower() == "q":
                break
            try:
                horizon = int(horizon_input)
            except ValueError:
                print('❌ The horizon must be a valid integer.')
                continue

            exclude_input = input(
                "👉 Keywords to exclude (comma-separated, e.g. wind, gusts): "
            ).strip()
            if exclude_input.lower() == "q":
                break

            # Parsing the Exclusion List
            exclude_keywords = (
                [k.strip() for k in exclude_input.split(",")] if exclude_input else []
            )

            # Execution of the main analysis function
            analyze_and_plot_correlations(
                path_parquet=path_parquet,
                city=city,
                target_base_name=target,
                horizon=horizon,
                exclude_keywords=exclude_keywords,
            )

            # Aggressive system cleaning at the end of each iteration
            gc.collect()

        except KeyboardInterrupt:
            print('[!] Exit requested by the terminal. Closing.')
            break
        except Exception as e:
            print(f"⚠️ An error occurred in the main loop: {e}")
            gc.collect()


if __name__ == "__main__":
    main_loop()
