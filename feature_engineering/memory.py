"""Optimizes data types to reduce memory footprint."""

import gc

from pandas import DataFrame, to_numeric


def optimize_memory(df: DataFrame) -> DataFrame:
    """Optimizes data types to reduce memory footprint."""
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            df[col] = to_numeric(df[col], downcast="integer")
    gc.collect()
    return df
