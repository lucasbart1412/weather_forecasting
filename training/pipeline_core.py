from training._runtime import *
from training.memory_impl import MemoryManager
from training.model_manager_impl import ModelManager


def optimize_memory(df: DataFrame) -> DataFrame:
    """Optimizes data types to reduce memory footprint."""
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            df[col] = to_numeric(df[col], downcast="integer")
    gc.collect()
    return df


def main():
    """Dynamic truncation"""
    parser = argparse.ArgumentParser(description="WeatherMaster - Training Pipeline")
    parser.add_argument(
        "--city", type=str, required=True, help="City used for training"
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    city = args.city.lower().strip()
    logger.info(f"{'=' * 50}")
    logger.info('🌤️  WEATHER MASTER ULTIMATE - TRAINING')
    logger.info(f"{'=' * 50}")
    logger.info(f"📍 City: {city}")

    # Loading archive data
    lm = LocationManager()
    fetcher = DataFetcher()

    geo = lm.get_geo_grid(city)
    if not geo:
        logger.error(f"❌ Geocoding failed for {city}")
        sys.exit(1)

    logger.info(f"✅ Coordinates: {geo['lat']:.4f}, {geo['lon']:.4f}")
    path = Path(args.output) if args.output else WeatherConfig.BASE_DIR
    path.mkdir(parents=True, exist_ok=True)

    # Loading archive data
    cache_p = path / "full_grid_archive.parquet"
    cache_df_arc = path / "df_arc.pkl"
    if cache_p.exists():
        logger.info(f"📦 Loading weather archive: {cache_p}")
        df_raw = read_parquet(cache_p)
    else:
        logger.info(
            '📡 Weather archive missing: retrieving via DataFetcher (archive mode)...'
        )
        try:
            df_raw = fetcher.fetch_data(geo, is_archive=True)
            if df_raw is None or df_raw.empty:
                raise ValueError('DataFetcher returned an empty archive.')
            df_raw.to_parquet(cache_p, index=False)
            logger.info(f"✅ Weather archive saved: {cache_p}")
        except Exception as exc:
            logger.error(f"❌ Unable to retrieve the archive: {exc}")
            sys.exit(1)

    # Dynamic truncation
    idx_debut = df_raw.first_valid_index()
    idx_fin = df_raw.last_valid_index()
    df_raw = df_raw.loc[idx_debut:idx_fin]
    df_raw["date"] = to_datetime(df_raw["date"])
    df_raw = df_raw.sort_values("date").reset_index(drop=True)

    # Memory optimization
    df_raw = optimize_memory(df_raw)
    MemoryManager.trim_memory()

    # Tide:
    logger.info('Tide integration...')
    df_marine = fetcher.fetch_marine_data(geo["lat"], geo["lon"], is_archive=True)

    if not df_marine.empty and "date" in df_marine.columns:
        df_marine["date"] = to_datetime(df_marine["date"])
        df_marine = optimize_memory(df_marine)
        df_raw = merge(df_raw, df_marine, on="date", how="left")
        if "tide" not in WeatherConfig.TARGETS:
            WeatherConfig.TARGETS.append("tide")
    else:
        logger.warning('⚠️ Tides not available')
        if "tide" in WeatherConfig.TARGETS:
            WeatherConfig.TARGETS.remove("tide")

    df_raw = df_raw.ffill(limit=3)

    # Time Split
    split_idx = int(len(df_raw) * 0.8)
    cut_off_date = df_raw.iloc[split_idx]["date"]

    # Climatology
    clima = {}
    for t in ["temp", "press", "hum"]:
        if t in df_raw.columns:
            clima[t] = (
                df_raw.loc[df_raw["date"] < cut_off_date]
                .groupby([df_raw["date"].dt.month, df_raw["date"].dt.hour])[t]
                .mean()
                .to_dict()
            )

    # Feature engineering
    df_all_processed, _ = FeatureEngineer.prepare(
        df_raw, tz_name=geo["tz"], climatology_ref=clima, lon=geo["lon"], lat=geo["lat"]
    )
    df_all_processed = FeatureEngineer.add_tide_physics(df_all_processed)
    df_all_processed = df_all_processed.copy(deep=True)

    del df_raw
    MemoryManager.trim_memory()

    df_arc = optimize_memory(df_all_processed)
    del df_all_processed

    df_arc.replace([inf, -inf], nan, inplace=True)
    df_arc = df_arc.ffill(limit=3)

    # Verification of existing models
    mm = ModelManager(str(path), geo["lat"], geo["lon"])
    min_pkl = 10 if "tide" in WeatherConfig.TARGETS else 9
    min_pkl = min_pkl * 3

    existing_models = sum(1 for f in path.glob("model_*.pkl") if f.is_file())

    if existing_models < min_pkl:
        logger.info(
            f"🚀 {existing_models}/{min_pkl} models found - Training required"
        )

        job_dump(df_arc, cache_df_arc)
        del df_arc
        MemoryManager.trim_memory()

        try:
            libc = ctypes.CDLL(None)
            libc.malloc_trim(0)
        except (AttributeError, OSError):
            pass

        mm.train(filename=str(cache_df_arc), climatology_ref=clima, cut_off_date=cut_off_date)
        sleep(1e-4)

        try:
            ctypes.CDLL(None).malloc_trim(0)
        except (AttributeError, OSError):
            pass
    else:
        logger.info(f"✅ {existing_models} models found - Training skip")

    logger.info(f"🎉 Pipeline finished! Models in: {path}")


def _legacy_main():
    """Delegates CLI execution to the package training orchestrator."""
    from training.pipeline import main as pipeline_main

    return pipeline_main()


if __name__ == "__main__":
    main()