from infrastructure.config import WeatherConfig
from training._runtime import *
from training.memory_impl import MemoryManager
from numpy.random import RandomState


class TrainingLoopMixin:
    def train(
        self,
        filename: str,
        climatology_ref: Optional[Dict] = None,
        cut_off_date: Optional[Timestamp] = None,
    ):
        """Complete training pipeline by blocks and targets."""
        if not os.path.exists(path=Path(WeatherConfig.BASE_DIR)):
            os.makedirs(name=Path(WeatherConfig.BASE_DIR), exist_ok=True)
        df = job_load(filename)
        df = df.sort_values("date").reset_index(drop=True)

        # Train Separation/Validation/Calibration
        for t in WeatherConfig.TARGETS:
            y_col = f"{t}_anom" if t in ["temp", "press", "hum"] else t
            for _block_name, horizons in self.BLOCKS.items():
                for h in horizons:
                    df[f"target_{t}_h{h}"] = df[y_col].shift(-h)

        gap_hours = 168

        # Train Separation/Validation/Calibration
        df_train_full = df[df["date"] < cut_off_date].copy()
        val_start_date = cut_off_date + Timedelta(hours=gap_hours)
        df_val_base = df[df["date"] >= val_start_date].copy()

        calib_duration_days = 180
        calib_cut_off = cut_off_date - Timedelta(days=calib_duration_days)
        df_train_base = df_train_full[df_train_full["date"] < calib_cut_off].copy()
        calib_start_date = calib_cut_off + Timedelta(hours=gap_hours)
        df_calib = df_train_full[df_train_full["date"] >= calib_start_date].copy()

        job_dump(
            {
                "df_train_base": df_train_base,
                "df_val_base": df_val_base,
                "df_calib": df_calib,
                "df_train_full": df_train_full,
            },
            filename=self.path / "base_data.pkl",
        )

        del df_train_base, df_val_base, df_calib, df_train_full
        MemoryManager.trim_memory()

        # Features
        excluded = ["date"] + [f"{t}_anom" for t in ["temp", "press", "hum"]]
        base_features = [
            c
            for c in df.columns
            if c not in WeatherConfig.TARGETS
            and c not in ["month_cos", "month_sin", "hour_cos", "hour_sin"]
            and not c.startswith("target_")
            and c not in excluded
        ]
        extra_features = [
            "horizon_h",
            "target_solar_elev",
            "target_hour_sin",
            "target_hour_cos",
            "target_month_sin",
            "target_month_cos",
        ]
        all_features = base_features + extra_features

        meta_data = {"features": all_features, "clima": climatology_ref if climatology_ref else {}}
        del climatology_ref
        MemoryManager.trim_memory()

        # Meters
        steps_stacking = 15700
        steps_long_block = 2000
        steps_short_block = 10000
        steps_quantile = 6000
        total_steps = (
            2000
            + 100 * 1000
            + steps_stacking * len(WeatherConfig.TARGETS)
            + steps_long_block * len(WeatherConfig.TARGETS)
            + steps_short_block * len(WeatherConfig.TARGETS)
            + steps_quantile * 2
        )
        step = 0

        df = df.sort_values("date").reset_index(drop=True)
        start_time = time()
        logger.info(f"\n🚀 Training by Blocks ({total_steps} iterations planned)")

        missing = [f for f in base_features if f not in df.columns]
        del df
        MemoryManager.trim_memory()

        if missing:
            logger.error(f"❌ Missing columns: {missing}")

        if not hasattr(self, "bias_correctors"):
            self.bias_correctors = {}

        # Main Loop
        for block_name, horizons in self.BLOCKS.items():
            for t in WeatherConfig.TARGETS:
                start_time_model = time()

                # Skip models already trained (optimization)
                if os.path.exists(self.path / f"model_{t}_{block_name}.pkl"):
                    logger.info('Model already trained!')
                    continue

                X_tr_list, y_tr_list = [], []
                val_dates_list = []
                cal_dates_list = []
                X_val_list, y_val_list = [], []
                X_cal_list, y_cal_list = [], []
                val_results_list = []

                logger.info(
                    f"⏳ Block '{block_name}' | Target '{t}' ({step + 1}/{total_steps})"
                )

                # Loading Master Data
                base_data = job_load(self.path / "base_data.pkl")
                df_train_base = base_data["df_train_base"]
                df_val_base = base_data["df_val_base"]
                df_calib = base_data["df_calib"]
                del base_data
                MemoryManager.trim_memory()

                horizons_valides = []

                for h in horizons:
                    target_col_generated = f"target_{t}_h{h}"
                    all_sets = True

                    for d_set, l_x, l_y in [
                        (df_train_base, X_tr_list, y_tr_list),
                        (df_val_base, X_val_list, y_val_list),
                        (df_calib, X_cal_list, y_cal_list),
                    ]:
                        col_empty = d_set[base_features].isna().all(axis=0)
                        if col_empty.any():
                            bad_cols = col_empty[col_empty].index.tolist()
                            logger.warning(f"⚠️ Columns 100% empty: {bad_cols}")

                        y_h = d_set[target_col_generated]
                        essential_features = ["temp", "press", "hum"]
                        mask = y_h.notna() & d_set[essential_features].notna().all(
                            axis=1
                        )

                        if not mask.any():
                            all_sets = False
                            logger.warning(f"⚠️ No valid data horizon {h}h")
                            break

                        X_h = d_set.loc[mask, base_features].copy()
                        X_h = X_h.ffill(limit=3).astype("float32")
                        y_h_clean = y_h.loc[mask].astype("float32")

                        future_dates = d_set.loc[mask, "date"] + Timedelta(hours=h)

                        X_h["time_step"] = d_set.loc[mask].index.astype("int32")
                        X_h["horizon_h"] = h
                        X_h["target_solar_elev"] = (
                            FeatureEngineer.get_solar_elev_vectorized(
                                self.lat, self.lon, future_dates
                            ).astype("float32")
                        )
                        X_h["target_hour_sin"] = sin(
                            2 * pi * future_dates.dt.hour / 24
                        ).astype("float32")
                        X_h["target_hour_cos"] = cos(
                            2 * pi * future_dates.dt.hour / 24
                        ).astype("float32")
                        X_h["target_month_sin"] = sin(
                            2 * pi * future_dates.dt.month / 12
                        ).astype("float32")
                        X_h["target_month_cos"] = cos(
                            2 * pi * future_dates.dt.month / 12
                        ).astype("float32")

                        if d_set is df_val_base:
                            val_results_list.append(
                                DataFrame(
                                    {
                                        "hour": future_dates.dt.hour,
                                        "y_true": y_h_clean.values,
                                    }
                                )
                            )
                            val_dates_list.append(future_dates)
                        elif d_set is df_calib:
                            cal_dates_list.append(future_dates)

                        l_x.append(X_h)
                        l_y.append(y_h_clean)

                        del (
                            X_h,
                            y_h_clean,
                            future_dates,
                            mask,
                            essential_features,
                            col_empty,
                            d_set,
                            y_h,
                        )
                        MemoryManager.trim_memory()

                    if not all_sets:
                        logger.warning(f"⚠️ Incomplete data horizon {h}h")
                        raise ValueError(f"Not enough data for horizon {h}.")
                    horizons_valides.append(h)

                # Original DataFrames Cleanup
                del df_train_base, df_val_base, df_calib
                MemoryManager.trim_memory()

                if not X_tr_list or not X_val_list:
                    logger.error(f"❌ No valid data for {t} | {block_name}")
                    continue

                # Chronological weights
                weights_list = []
                for _h, df_x in zip(horizons, X_tr_list):
                    w_chrono = exp(linspace(0, 1.2, len(df_x)))
                    w_horizon = 1.0
                    weights_list.append(w_chrono * w_horizon)

                # Float32 concatenation
                X_tr = concat(X_tr_list, ignore_index=True).astype("float32")
                y_tr = concat(y_tr_list, ignore_index=True).astype("float32")
                X_val = concat(X_val_list, ignore_index=True).astype("float32")
                y_val = concat(y_val_list, ignore_index=True).astype("float32")
                X_cal = concat(X_cal_list, ignore_index=True).astype("float32")
                y_cal = concat(y_cal_list, ignore_index=True).astype("float32")

                del X_tr_list, y_tr_list, X_val_list, y_val_list, X_cal_list, y_cal_list
                MemoryManager.trim_memory()

                # Lengths per horizon
                X_tr_lengths = []
                X_val_lengths = []
                X_cal_lengths = []

                for _h in horizons_valides:
                    mask_tr = y_tr.notna()
                    X_tr_lengths.append(mask_tr.sum())

                    mask_val = y_val.notna()
                    X_val_lengths.append(mask_val.sum())

                    mask_cal = y_cal.notna()
                    X_cal_lengths.append(mask_cal.sum())

                # Selection features by variance
                selector = (
                    Series({c: X_tr[c].std() for c in X_tr.columns if c != "time_step"})
                    > 0
                )
                used_features = X_tr.columns[
                    X_tr.columns.isin(selector[selector].index.tolist())
                    | (X_tr.columns == "time_step")
                ].tolist()
                if "time_step" not in used_features:
                    used_features.append("time_step")

                X_tr = X_tr[used_features]
                X_val = X_val[used_features]
                X_cal = X_cal[used_features]

                # Calibration backup
                X_cal.to_parquet(self.path / f"X_cal_{t}_{block_name}.parquet")
                job_dump(y_cal, self.path / f"y_cal_{t}_{block_name}.pkl")

                del X_cal, y_cal
                MemoryManager.trim_memory()

                weights = concat([Series(w) for w in weights_list]).values
                del weights_list

                logger.info(f"📊 Features: {len(used_features)}/{len(base_features)}")
                logger.info(f"✅ Train: {len(X_tr)} | Val: {len(X_val)}")

                # Delete time_step for short/long
                if (
                    hasattr(X_tr, "drop")
                    and block_name in ["short", "long"]
                    and "time_step" in X_tr.columns
                    and "time_step" in X_val.columns
                ):
                    X_tr = X_tr.drop(columns=["time_step"])
                    X_val = X_val.drop(columns=["time_step"])

                # Probability model for precip
                if t == "precip":
                    y_tr_bin = (y_tr > 0.1).astype(int)
                    y_val_bin = (y_val > 0.1).astype(int)

                    prob_params = {
                        "n_estimators": 10000,
                        "learning_rate": 0.03,
                        "objective": "binary",
                        "max_depth": 4,
                        "num_leaves": 15,
                        "min_child_samples": 300,
                        "colsample_bytree": 0.65,
                        "subsample": 0.8,
                        "reg_alpha": 2.0,
                        "reg_lambda": 5.0,
                        "class_weight": "balanced",
                        "n_jobs": -1,
                        "verbose": -1,
                    }
                    
                    if WeatherConfig.HAS_GPU:
                        prob_params.update({'device_type': 'gpu', 'max_bin': 63, 'gpu_use_dp': False})

                    prob_model = LGBMClassifier(**prob_params)

                    X_tr_prob = (
                        X_tr.drop(columns=["time_step"])
                        if "time_step" in X_tr.columns
                        else X_tr
                    )
                    X_val_prob = (
                        X_val.drop(columns=["time_step"])
                        if "time_step" in X_val.columns
                        else X_val
                    )

                    total_steps -= 2000

                    # ✅ LIGHTGBM CORRECTION: eval_X/eval_y
                    prob_model.fit(
                        X_tr_prob,
                        y_tr_bin,
                        eval_set=[(X_val_prob, y_val_bin)],
                        eval_metric="auc",
                        callbacks=[early_stopping(100), log_evaluation(0)],
                    )

                    best_it_prob = (
                        prob_model.best_iteration_
                        if (
                            prob_model.best_iteration_
                            and prob_model.best_iteration_ > 0
                        )
                        else prob_model.n_estimators
                    )
                    total_steps = total_steps + best_it_prob
                    step += best_it_prob

                    try:
                        prob_preds = prob_model.predict_proba(X_val_prob)[:, 1]

                        seuils = arange(0.1, 0.9, 0.05)
                        best_seuil = 0.5
                        best_f1 = 0

                        for seuil in seuils:
                            preds_temp = (prob_preds >= seuil).astype(int)
                            score_f1 = f1_score(y_val_bin, preds_temp, pos_label=1)

                            if score_f1 > best_f1:
                                best_f1 = score_f1
                                best_seuil = seuil

                        logger.info(
                            f"💡 Optimal threshold: {best_seuil:.2f} (F1: {best_f1:.4f})"
                        )

                        binary_preds = (prob_preds >= best_seuil).astype(int)
                        error_rate = (1 - accuracy_score(y_val_bin, binary_preds)) * 100
                        logger.info(f"Classification error: {error_rate:.2f}%")

                        brier = brier_score_loss(y_val_bin, prob_preds)
                        logger.info(f"Brier Score: {brier:.4f}")

                        report = classification_report(
                            y_val_bin,
                            binary_preds,
                            target_names=["No rain", "Rain"],
                        )
                        logger.info(f"\n  📊 Report Classification:\n{report}")

                    except Exception as e:
                        logger.error(f"Evaluation error: {e}")

                    job_dump(
                        prob_model, self.path / f"model_precip_prob_{block_name}.pkl"
                    )
                    del prob_model, X_tr_prob, X_val_prob, y_tr_bin, y_val_bin
                    MemoryManager.trim_memory()

                # Optuna for short block
                if block_name == "short" and t in ["temp", "precip", "cloud"]:
                    sample_size = min(len(X_tr), 130000)
                    indices = sort(
                        RandomState().choice(
                            len(X_tr), sample_size, replace=False
                        )
                    )
                    X_tr_optuna = (
                        X_tr.drop(columns=["time_step"])
                        if "time_step" in X_tr.columns
                        else X_tr
                    )

                    X_opt_tr, X_opt_val, y_opt_tr, y_opt_val, w_opt_tr, _ = (
                        train_test_split(
                            X_tr_optuna.iloc[indices].copy(),
                            y_tr.iloc[indices].copy(),
                            weights[indices].copy(),
                            test_size=0.15,
                            shuffle=False,
                        )
                    )

                    logger.info(
                        f"🔍 Optuna for {t} (Short) on {sample_size} points..."
                    )
                    best_params, total_iterations = self.optimize_short_block(
                        X_opt_tr, y_opt_tr, X_opt_val, y_opt_val, t, weight=w_opt_tr
                    )

                    del X_opt_tr, X_opt_val, y_opt_tr, y_opt_val, w_opt_tr, X_tr_optuna
                    MemoryManager.trim_memory()

                    model = LGBMRegressor(**best_params)
                    total_steps = (
                        total_steps
                        - (40000 if t == "temp" else 30000)
                        + total_iterations
                    )
                    step += total_iterations
                else:
                    model = self.get_advanced_models(
                        block_name, t, lengths=X_tr_lengths, X_tr=X_tr
                    )
                    logger.info(f"🔍 Training {t} | {block_name} without Optuna...")

                # Practice
                if block_name in ["short", "long"]:
                    # ✅ LIGHTGBM CORRECTION: eval_X/eval_y
                    model.fit(
                        X_tr,
                        y_tr,
                        eval_set=[(X_val, y_val)],
                        sample_weight=weights,
                        callbacks=[
                            early_stopping(stopping_rounds=300),
                            log_evaluation(2000),
                        ],
                    )
                    best_it = (
                        model.best_iteration_
                        if (model.best_iteration_ and model.best_iteration_ > 0)
                        else model.n_estimators
                    )

                    if block_name == "short":
                        total_steps = total_steps - steps_short_block + best_it
                    if block_name == "long":
                        total_steps = total_steps - steps_long_block + best_it
                    step += best_it
                else:
                    # Stacking
                    logger.info(f"🔍 Stacking for {t} | {block_name}...")
                    job_dump(
                        {"X": X_val, "y": y_val},
                        self.path / f"val_{t}_{block_name}.pkl",
                    )
                    MemoryManager.trim_memory()

                    del X_val, y_val
                    MemoryManager.trim_memory()

                    fit_data = {"X": X_tr, "y": y_tr}
                    job_dump(
                        fit_data, filename=self.path / f"fit_data_{t}_{block_name}.pkl"
                    )
                    del X_tr, y_tr, fit_data
                    MemoryManager.trim_memory()

                    model.fit(
                        target_type=t,
                        filename=self.path / f"fit_data_{t}_{block_name}.pkl",
                        path=self.path,
                        val_path=self.path / f"val_{t}_{block_name}.pkl",
                    )
                    step += steps_stacking

                    X_tr = job_load(self.path / f"fit_data_{t}_{block_name}.pkl")["X"]
                    y_tr = job_load(self.path / f"fit_data_{t}_{block_name}.pkl")["y"]
                    X_val = job_load(self.path / f"val_{t}_{block_name}.pkl")["X"]
                    y_val = job_load(self.path / f"val_{t}_{block_name}.pkl")["y"]

                    for name in [
                        f"val_{t}_{block_name}.pkl",
                        f"fit_data_{t}_{block_name}.pkl",
                    ]:
                        path = self.path / name
                        if path.exists():
                            path.unlink()

                # Delete time_step
                if hasattr(X_tr, "drop") and "time_step" in X_tr.columns:
                    X_tr = X_tr.drop(columns=["time_step"])
                if hasattr(X_val, "drop") and "time_step" in X_val.columns:
                    X_val = X_val.drop(columns=["time_step"])

                # Validation predictions
                preds_val = model.predict(X_val)

                # Importance features
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                else:
                    importances = None

                if importances is not None:
                    feat_imp = Series(importances, index=model.feature_names_in_)
                    top_5 = feat_imp.sort_values(ascending=False).head(5)

                    logger.info(f"🔍 Top 5 for {t} ({block_name}):")
                    for rank, (name, val) in enumerate(top_5.items(), 1):
                        logger.info(f"{rank}. {name} ({val:.0f})")

                # Calibration loading
                X_cal = read_parquet(self.path / f"X_cal_{t}_{block_name}.parquet")
                y_cal = job_load(self.path / f"y_cal_{t}_{block_name}.pkl")

                if "time_step" in X_cal.columns:
                    X_cal = X_cal.drop(columns=["time_step"])

                preds_val = model.predict(X_val)
                preds_cal = model.predict(X_cal)

                # Dates Alignment
                val_dates_series = to_datetime(
                    concat([Series(d) for d in val_dates_list], ignore_index=True)
                )
                cal_dates_series = to_datetime(
                    concat([Series(d) for d in cal_dates_list], ignore_index=True)
                )

                if t not in self.bias_correctors:
                    self.bias_correctors[t] = {}

                start_idx = 0
                start_idx_cal = 0

                bias_corrections_val_list = []
                bias_corrections_cal_list = []

                def get_season_name(month):
                    if month in [12, 1, 2]:
                        return "winter"
                    if month in [3, 4, 5]:
                        return "spring"
                    if month in [6, 7, 8]:
                        return "summer"
                    return "autumn"

                for h, length, cal_length in zip(
                    horizons_valides, X_val_lengths, X_cal_lengths
                ):
                    h_int = int(h)
                    end_idx = start_idx + length
                    end_idx_cal = start_idx_cal + cal_length

                    y_true_h_cal = y_cal.iloc[start_idx_cal:end_idx_cal].values
                    preds_h_cal = preds_cal[start_idx_cal:end_idx_cal]
                    dates_h_cal = cal_dates_series.iloc[
                        start_idx_cal:end_idx_cal
                    ].reset_index(drop=True)
                    dates_h_val = val_dates_series.iloc[start_idx:end_idx].reset_index(
                        drop=True
                    )

                    errors_cal = y_true_h_cal - preds_h_cal

                    cal_hours = dates_h_cal.dt.hour
                    cal_seasons = dates_h_cal.dt.month.apply(get_season_name)

                    df_err_cal = DataFrame(
                        {"season": cal_seasons, "hour": cal_hours, "err": errors_cal},
                        copy=False,
                    )

                    global_hourly_bias = (
                        df_err_cal.groupby("hour")["err"].mean().to_dict()
                    )
                    seasonal_bias = (
                        df_err_cal.groupby(["season", "hour"])["err"].mean().to_dict()
                    )

                    self.bias_correctors[t][h_int] = {
                        "global": global_hourly_bias,
                        "seasonal": seasonal_bias,
                    }

                    val_hours = dates_h_val.dt.hour
                    val_seasons = dates_h_val.dt.month.apply(get_season_name)

                    bias_h_val = array(
                        [
                            seasonal_bias.get(
                                (s, h_pt), global_hourly_bias.get(h_pt, 0.0)
                            )
                            for s, h_pt in zip(val_seasons, val_hours)
                        ]
                    )

                    bias_h_cal = array(
                        [
                            seasonal_bias.get(
                                (s, h_pt), global_hourly_bias.get(h_pt, 0.0)
                            )
                            for s, h_pt in zip(cal_seasons, cal_hours)
                        ]
                    )

                    bias_corrections_val_list.append(bias_h_val)
                    bias_corrections_cal_list.append(bias_h_cal)

                    start_idx = end_idx
                    start_idx_cal = end_idx_cal

                # Application of corrections
                preds_val_corrected = array(preds_val).ravel() + concatenate(
                    bias_corrections_val_list
                )
                preds_cal_corrected = array(preds_cal).ravel() + concatenate(
                    bias_corrections_cal_list
                )

                # --- CALCULATION OF METRICS ---
                if t == "precip":
                    preds_val_corrected = maximum(0, preds_val_corrected)
                    preds_cal_corrected = maximum(0, preds_cal_corrected)
                elif t in ["wind", "gusts", "cloud", "hum"]:
                    preds_val_corrected = maximum(0, preds_val_corrected)
                    preds_cal_corrected = maximum(0, preds_cal_corrected)

                    if t in ["hum", "cloud"]:
                        preds_val_corrected = minimum(100, preds_val_corrected)
                        preds_cal_corrected = minimum(100, preds_cal_corrected)

                # Deleting temp files
                for name in [
                    f"X_cal_{t}_{block_name}.parquet",
                    f"y_cal_{t}_{block_name}.pkl",
                ]:
                    path = self.path / name
                    if path.exists():
                        path.unlink()

                # Quantiles
                m_low, m_high = None, None
                quantile_params = {
                    "objective": "quantile",
                    "n_estimators": 3000,
                    "learning_rate": 0.025,
                    "max_depth": 5,
                    "num_leaves": 31,
                    "min_child_samples": 250,
                    "colsample_bytree": 0.5,
                    "subsample": 0.8,
                    "reg_alpha": 1.0,
                    "reg_lambda": 5.0,
                    "n_jobs": -1,
                    "verbose": -1,
                }
                if WeatherConfig.HAS_GPU:
                    quantile_params.update(
                        {
                            "device_type": "gpu",
                            "max_bin": 63,
                            "gpu_use_dp": False,
                        }
                    )

                if t == "temp":
                    logger.info('Quantiles Temperature...')
                    # ✅ LIGHTGBM CORRECTION: eval_X/eval_y
                    m_low = LGBMRegressor(alpha=0.1, **quantile_params).fit(
                        X_tr,
                        y_tr,
                        eval_set=[(X_val, y_val)],
                        callbacks=[
                            log_evaluation(0),
                            early_stopping(stopping_rounds=100),
                        ],
                    )
                    m_high = LGBMRegressor(alpha=0.9, **quantile_params).fit(
                        X_tr,
                        y_tr,
                        eval_set=[(X_val, y_val)],
                        callbacks=[
                            log_evaluation(0),
                            early_stopping(stopping_rounds=100),
                        ],
                    )
                    m_low_best_it = (
                        m_low.best_iteration_
                        if (m_low.best_iteration_ and m_low.best_iteration_ > 0)
                        else m_low.n_estimators
                    )
                    m_high_best_it = (
                        m_high.best_iteration_
                        if (m_high.best_iteration_ and m_high.best_iteration_ > 0)
                        else m_high.n_estimators
                    )
                    total_steps = (
                        total_steps - steps_quantile + m_low_best_it + m_high_best_it
                    )
                    step = step + m_low_best_it + m_high_best_it

                if t == "precip":
                    logger.info('Quantiles Rain...')
                    precip_quantile_params = quantile_params.copy()
                    precip_quantile_params["metric"] = "tweedie"
                    precip_quantile_params["tweedie_variance_power"] = 1.5

                    # ✅ LIGHTGBM CORRECTION: eval_X/eval_y
                    m_low = LGBMRegressor(alpha=0.1, **precip_quantile_params).fit(
                        X_tr,
                        y_tr,
                        eval_set=[(X_val, y_val)],
                        callbacks=[
                            log_evaluation(0),
                            early_stopping(stopping_rounds=100),
                        ],
                    )
                    m_high = LGBMRegressor(alpha=0.9, **precip_quantile_params).fit(
                        X_tr,
                        y_tr,
                        eval_set=[(X_val, y_val)],
                        callbacks=[
                            log_evaluation(0),
                            early_stopping(stopping_rounds=100),
                        ],
                    )
                    m_high_best_it = (
                        m_high.best_iteration_
                        if (m_high.best_iteration_ and m_high.best_iteration_ > 0)
                        else m_high.n_estimators
                    )
                    m_low_best_it = (
                        m_low.best_iteration_
                        if (m_low.best_iteration_ and m_low.best_iteration_ > 0)
                        else m_low.n_estimators
                    )
                    total_steps = (
                        total_steps - steps_quantile + m_low_best_it + m_high_best_it
                    )
                    step = step + m_low_best_it + m_high_best_it

                bias_corrector = self.bias_correctors[t]
                bundle = {
                    "model": model,
                    "m_low": m_low,
                    "m_high": m_high,
                    "bias_corrector": bias_corrector,
                    "features": used_features,
                }
                job_dump(bundle, self.path / f"model_{t}_{block_name}.pkl", compress=6)
                job_dump(meta_data, self.path / "global_meta.pkl")

                # Assessment
                from sklearn.metrics import (
                    mean_absolute_error,
                    mean_pinball_loss,
                    root_mean_squared_error,
                )

                y_tr_metric, y_val_metric, y_cal_metric = y_tr, y_val, y_cal
                preds_val_metric, preds_val_corr_metric, preds_cal_corr_metric = (
                    preds_val,
                    preds_val_corrected,
                    preds_cal_corrected,
                )
                preds_tr_base = model.predict(X_tr)

                if t in ["wind", "gusts", "hum"]:
                    main_metric_str = f"Val RMSE: {root_mean_squared_error(y_val_metric, preds_val_metric):.3f} | Corr: {root_mean_squared_error(y_val_metric, preds_val_corr_metric):.3f}"
                else:
                    main_metric_str = f"Val MAE: {mean_absolute_error(y_val_metric, preds_val_metric):.3f} | Corr: {mean_absolute_error(y_val_metric, preds_val_corr_metric):.3f}"

                q10_str, q90_str = "N/A", "N/A"
                if m_low is not None:
                    q10_preds = m_low.predict(X_val)
                    if t == "cloud":
                        q10_preds = q10_preds * 100.0
                    q10_str = f"Pinball: {mean_pinball_loss(y_val_metric, q10_preds, alpha=0.1):.3f}"

                if m_high is not None:
                    q90_preds = m_high.predict(X_val)
                    if t == "cloud":
                        q90_preds = q90_preds * 100.0
                    q90_str = f"Pinball: {mean_pinball_loss(y_val_metric, q90_preds, alpha=0.9):.3f}"

                mae_val_base = mean_absolute_error(y_val_metric, preds_val_metric)
                mae_tr_base = mean_absolute_error(y_tr_metric, preds_tr_base)
                ratio_overfit = mae_val_base / mae_tr_base if mae_tr_base > 0 else nan

                mae_val_corr = mean_absolute_error(y_val_metric, preds_val_corr_metric)
                mae_cal_corr = mean_absolute_error(y_cal_metric, preds_cal_corr_metric)
                ratio_calib = mae_val_corr / mae_cal_corr if mae_cal_corr > 0 else nan

                logger.info(f"""[{t} | {block_name.upper()}] {(time() - start_time_model) / 60:.1f}min. {step}/{total_steps}
    ETA: {(total_steps / step) * (time() - start_time) / 3600:.1f}h | {main_metric_str}
    MAE Phys: {mae_val_base:.2f} -> Corr: {mae_val_corr:.2f} | Gain: {mae_val_corr / mae_val_base:.2f}
    Overfit: {ratio_overfit:.2f} | Calib: {ratio_calib:.2f}
    Quantiles: [10% {q10_str}] | [90% {q90_str}]""")

                # Final Cleaning
                if hasattr(model, "final_estimator_"):
                    if hasattr(model.final_estimator_, "free_dataset"):
                        model.final_estimator_.free_dataset()

                for pipe in getattr(model, "estimators_", []):
                    if hasattr(pipe, "named_steps") and "model" in pipe.named_steps:
                        actual_model = pipe.named_steps["model"]
                        if hasattr(actual_model, "free_dataset"):
                            actual_model.free_dataset()
                    elif hasattr(pipe, "free_dataset"):
                        pipe.free_dataset()

                if hasattr(model, "free_dataset"):
                    model.free_dataset()

                del X_tr, y_tr, X_val, y_val, X_cal, y_cal
                del (
                    model,
                    preds_val,
                    preds_cal,
                    bias_corrector,
                    preds_cal_corrected,
                    weights,
                )
                del bias_corrections_val_list, bias_corrections_cal_list, importances
                MemoryManager.trim_memory()

        logger.info(
            f"\n🎉 Training completed in {(time() - start_time) / 3600:.1f}h !"
        )
