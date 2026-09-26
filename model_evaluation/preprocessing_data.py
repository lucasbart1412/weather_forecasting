"""Preparation of the test set for evaluation."""

import pandas as pd

from feature_engineering.feature_engineering import FeatureEngineer, optimize_memory


def resolve_cut_off_date(df_raw: pd.DataFrame, cut_off_date) -> pd.Timestamp:
    """Use the requested temporal split, with an 80% fallback for old callers."""
    if cut_off_date is not None:
        return pd.Timestamp(cut_off_date)
    split_idx = int(len(df_raw) * 0.8)
    return pd.Timestamp(df_raw.iloc[split_idx]["date"])


def prepare_test_data(evaluator, df_raw_path, cut_off_date):
    """Prepares test data with the same logic as the current evaluator."""
    print('📥 Loading and preparing test data...')
    df_raw = pd.read_parquet(df_raw_path)
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw = df_raw.sort_values("date").reset_index(drop=True)
    df_raw = optimize_memory(df_raw)

    cut_off_date = resolve_cut_off_date(df_raw, cut_off_date)

    clima = {}
    df_train_only = df_raw[df_raw["date"] < cut_off_date]
    for target in ["temp", "press", "hum"]:
        if target in df_raw.columns:
            clima[target] = (
                df_train_only.groupby(
                    [df_train_only["date"].dt.month, df_train_only["date"].dt.hour]
                )[target]
                .mean()
                .to_dict()
            )

    df_processed, _ = FeatureEngineer.prepare(
        df_raw,
        tz_name=evaluator.geo["tz"],
        climatology_ref=clima,
        lon=evaluator.geo["lon"],
        lat=evaluator.geo["lat"],
        elevation=evaluator.geo.get("elevation", 0),
    )

    if "tide" in df_processed.columns:
        df_processed = FeatureEngineer.add_tide_physics(df_processed)

    df_processed = df_processed.ffill(limit=3)
    gap_hours = 168
    val_start_date = cut_off_date + pd.Timedelta(hours=gap_hours)
    df_test = df_processed[df_processed["date"] >= val_start_date].copy()
    return df_test, clima


__all__ = ["prepare_test_data", "resolve_cut_off_date"]
