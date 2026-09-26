"""CLI entry point for the weather assessment."""

import argparse
from pathlib import Path

import pandas as pd

from infrastructure.config import WeatherConfig
from model_evaluation.evaluator_impl import ModelEvaluator

GEO_INFO = {
    "lat": 50.8035441,
    "lon": 4.3338445,
    "tz": "Europe/Brussels",
    "elevation": 62.0,
}


def main(argv: list[str] | None = None) -> None:
    """Executes the assessment on the configured targets and horizons."""
    parser = argparse.ArgumentParser(description="Evaluate trained weather models")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=WeatherConfig.BASE_DIR,
        help="Directory containing global_meta.pkl and model bundles",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Raw archive parquet (default: <base-dir>/full_grid_archive.parquet)",
    )
    parser.add_argument(
        "--cutoff-date",
        default="2024-01-01",
        help="End of the training period in ISO format",
    )
    args = parser.parse_args(argv)

    data_path = args.data or args.base_dir / "full_grid_archive.parquet"
    evaluateur = ModelEvaluator(base_path=args.base_dir, geo=GEO_INFO)
    df_test, climatology_map = evaluateur.prepare_test_data(
        data_path, pd.to_datetime(args.cutoff_date)
    )

    for target in WeatherConfig.TARGETS:
        for horizon in [12, 24, 48, 72, 96, 120, 144, 168]:
            evaluateur.evaluate(
                df_test=df_test,
                clima=climatology_map,
                target_variable=target,
                specific_horizon=horizon,
            )


__all__ = ["main"]
