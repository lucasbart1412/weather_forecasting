from infrastructure.config import WeatherConfig
from feature_engineering.target_transforms import decode_rain_target, encode_rain_target
from ml.bias_correction import apply_bias_correction
from training.bias_corrector import fit_bias_corrector
from training.calibration import split_calibration_data
from training.metrics import (
    has_multiple_classes,
    regression_metrics,
    select_f1_threshold,
)
from training.progress import TrainingEtaEstimator
from training._runtime import *
from training.memory_impl import MemoryManager
from numpy.random import RandomState
from time import perf_counter


def _constrain_prediction(target: str, values):
    """Apply the same physical bounds used for forecast output."""
    bounded = (
        maximum(0, values)
        if target in ["precip", "wind", "gusts", "hum", "cloud"]
        else values
    )
    if target in ["hum", "cloud"]:
        bounded = minimum(100, bounded)
    return bounded


def _metric_values(target: str, values):
    """Convert encoded precipitation values to physical units for evaluation."""
    if target == "precip":
        return decode_rain_target(values)
    return values


def _primary_error(target: str, actual, predicted) -> float:
    metrics = regression_metrics(actual, predicted)
    if target in ["wind", "gusts", "hum"]:
        return metrics["rmse"]
    return metrics["mae"]


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
                    target_values = df[y_col].shift(-h)
                    if t == "precip":
                        target_values = encode_rain_target(target_values)
                    df[f"target_{t}_h{h}"] = target_values

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

        df = df.sort_values("date").reset_index(drop=True)

        missing = [f for f in base_features if f not in df.columns]
        del df
        MemoryManager.trim_memory()

        if missing:
            raise ValueError(f"Training data is missing required features: {missing}")

        # Build a work plan that excludes reusable artifacts from the ETA.
        pending_tasks = []
        for planned_block in self.BLOCKS:
            for planned_target in WeatherConfig.TARGETS:
                model_path = self.path / f"model_{planned_target}_{planned_block}.pkl"
                should_train = not model_path.exists()
                if model_path.exists():
                    if planned_target != "precip":
                        should_train = False
                    else:
                        try:
                            bundle = job_load(model_path)
                        except Exception as exc:
                            logger.warning(
                                f"Unable to inspect existing model {model_path}: {exc}"
                            )
                            should_train = True
                        else:
                            should_train = not (
                                isinstance(bundle, dict)
                                and bundle.get("target_transform") == "log1p"
                                and bundle.get("precip_strategy_version") == 3
                                and (
                                    self.path
                                    / f"model_precip_prob_{planned_block}.pkl"
                                ).exists()
                            )
                if should_train:
                    pending_tasks.append((planned_block, planned_target))

        eta_estimator = TrainingEtaEstimator(pending_tasks)
        start_time = perf_counter()
        logger.info(
            f"\n🚀 Training by Blocks ({len(pending_tasks)} model tasks planned)"
        )

        # Main Loop
        for block_name, horizons in self.BLOCKS.items():
            for t in WeatherConfig.TARGETS:
                start_time_model = perf_counter()

                # Reuse only artifacts built with the current precipitation pipeline.
                if (block_name, t) not in pending_tasks:
                    logger.info("Model already trained with the current pipeline.")
                    continue

                logger.info(
                    f"⏳ Block '{block_name}' | Target '{t}' "
                    f"({eta_estimator.completed_count + 1}/{len(pending_tasks)})"
                )

                X_tr_list, y_tr_list = [], []
                X_val_list, y_val_list = [], []
                X_cal_list, y_cal_list = [], []

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
                    raise ValueError(
                        f"No train or validation data available for {t}/{block_name}."
                    )

                # Chronological weights
                weights_list = []
                for _h, df_x in zip(horizons, X_tr_list):
                    w_chrono = exp(linspace(0, 1.2, len(df_x)))
                    w_horizon = 1.0
                    weights_list.append(w_chrono * w_horizon)

                # Keep the training-set boundaries needed by the stacking trainer.
                X_tr_lengths = [len(data) for data in X_tr_list]

                # Float32 concatenation
                X_tr = concat(X_tr_list, ignore_index=True).astype("float32")
                y_tr = concat(y_tr_list, ignore_index=True).astype("float32")
                X_val = concat(X_val_list, ignore_index=True).astype("float32")
                y_val = concat(y_val_list, ignore_index=True).astype("float32")
                X_cal = concat(X_cal_list, ignore_index=True).astype("float32")
                y_cal = concat(y_cal_list, ignore_index=True).astype("float32")

                del X_tr_list, y_tr_list, X_val_list, y_val_list, X_cal_list, y_cal_list
                MemoryManager.trim_memory()

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

                (
                    X_cal,
                    y_cal,
                    X_cal_threshold,
                    y_cal_threshold,
                    X_cal_eval,
                    y_cal_eval,
                ) = split_calibration_data(
                    X_cal, y_cal, gap_steps=max(horizons)
                )

                # Keep bias-corrector fitting and calibration evaluation disjoint.
                X_cal.to_parquet(self.path / f"X_cal_{t}_{block_name}.parquet")
                job_dump(y_cal, self.path / f"y_cal_{t}_{block_name}.pkl")
                X_cal_eval.to_parquet(
                    self.path / f"X_cal_eval_{t}_{block_name}.parquet"
                )
                job_dump(
                    y_cal_eval,
                    self.path / f"y_cal_eval_{t}_{block_name}.pkl",
                )
                X_cal_threshold.to_parquet(
                    self.path / f"X_cal_threshold_{t}_{block_name}.parquet"
                )
                job_dump(
                    y_cal_threshold,
                    self.path / f"y_cal_threshold_{t}_{block_name}.pkl",
                )

                del (
                    X_cal,
                    y_cal,
                    X_cal_threshold,
                    y_cal_threshold,
                    X_cal_eval,
                    y_cal_eval,
                )
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
                conditional_precipitation = (
                    block_name == "medium" and t == "precip"
                )
                precip_val_probs = None
                optimal_threshold = 0.5
                if t == "precip":
                    y_tr_bin = (decode_rain_target(y_tr) > 0.1).astype(int)
                    y_val_bin = (decode_rain_target(y_val) > 0.1).astype(int)
                    if not has_multiple_classes(y_tr_bin):
                        raise ValueError(
                            f"Rain classifier for {block_name} requires both "
                            "wet and dry training samples."
                        )

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

                    # ✅ LIGHTGBM CORRECTION: eval_X/eval_y
                    prob_model.fit(
                        X_tr_prob,
                        y_tr_bin,
                        eval_set=[(X_val_prob, y_val_bin)],
                        eval_metric=(
                            "auc"
                            if has_multiple_classes(y_val_bin)
                            else "binary_logloss"
                        ),
                        callbacks=[early_stopping(100), log_evaluation(0)],
                    )

                    precip_val_probs = prob_model.predict_proba(X_val_prob)[:, 1]
                    X_cal_prob = read_parquet(
                        self.path / f"X_cal_threshold_{t}_{block_name}.parquet"
                    )
                    if "time_step" in X_cal_prob.columns:
                        X_cal_prob = X_cal_prob.drop(columns=["time_step"])
                    y_cal_bin = (
                        decode_rain_target(
                            job_load(
                                self.path
                                / f"y_cal_threshold_{t}_{block_name}.pkl"
                            )
                        )
                        > 0.1
                    ).astype(int)
                    cal_probs = prob_model.predict_proba(X_cal_prob)[:, 1]
                    optimal_threshold, calibration_f1 = select_f1_threshold(
                        y_cal_bin,
                        cal_probs,
                        arange(0.1, 0.9, 0.05),
                    )

                    logger.info(
                        f"💡 Optimal threshold: {optimal_threshold:.2f} "
                        f"(threshold-tuning F1: {calibration_f1:.4f})"
                    )

                    binary_preds = (precip_val_probs >= optimal_threshold).astype(int)
                    validation_f1 = f1_score(
                        y_val_bin,
                        binary_preds,
                        pos_label=1,
                        zero_division=0,
                    )
                    logger.info(f"Validation F1: {validation_f1:.4f}")
                    error_rate = (1 - accuracy_score(y_val_bin, binary_preds)) * 100
                    logger.info(f"Classification error: {error_rate:.2f}%")

                    brier = brier_score_loss(y_val_bin, precip_val_probs)
                    logger.info(f"Brier Score: {brier:.4f}")

                    report = classification_report(
                        y_val_bin,
                        binary_preds,
                        labels=[0, 1],
                        target_names=["No rain", "Rain"],
                        zero_division=0,
                    )
                    logger.info(f"\n  📊 Report Classification:\n{report}")

                    job_dump(
                        prob_model, self.path / f"model_precip_prob_{block_name}.pkl"
                    )
                    del (
                        prob_model,
                        X_tr_prob,
                        X_val_prob,
                        X_cal_prob,
                        y_tr_bin,
                        y_val_bin,
                        y_cal_bin,
                        cal_probs,
                    )
                    MemoryManager.trim_memory()

                # Optuna for short block
                if block_name == "short" and t in ["temp", "precip", "cloud"]:
                    sample_size = min(len(X_tr), 130000)
                    indices = sort(
                        RandomState(42).choice(
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
                    best_params, _ = self.optimize_short_block(
                        X_opt_tr, y_opt_tr, X_opt_val, y_opt_val, t, weight=w_opt_tr
                    )

                    del X_opt_tr, X_opt_val, y_opt_tr, y_opt_val, w_opt_tr, X_tr_optuna
                    MemoryManager.trim_memory()

                    model = LGBMRegressor(**best_params)
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
                else:
                    # Stacking
                    logger.info(f"🔍 Stacking for {t} | {block_name}...")

                    if conditional_precipitation:
                        train_wet_mask = decode_rain_target(y_tr) > 0.1
                        val_wet_mask = decode_rain_target(y_val) > 0.1
                        if not train_wet_mask.any() or not val_wet_mask.any():
                            raise ValueError(
                                "Medium-horizon precipitation intensity training "
                                "requires wet samples in both train and validation."
                            )

                        job_dump(
                            {"X": X_val, "y": y_val},
                            self.path / f"eval_val_{t}_{block_name}.pkl",
                        )
                        X_fit = X_tr.loc[train_wet_mask]
                        y_fit = y_tr.loc[train_wet_mask]
                        X_val_fit = X_val.loc[val_wet_mask]
                        y_val_fit = y_val.loc[val_wet_mask]
                    else:
                        X_fit, y_fit = X_tr, y_tr
                        X_val_fit, y_val_fit = X_val, y_val

                    job_dump(
                        {"X": X_val_fit, "y": y_val_fit},
                        self.path / f"val_{t}_{block_name}.pkl",
                    )
                    MemoryManager.trim_memory()

                    del X_val, y_val
                    MemoryManager.trim_memory()

                    fit_data = {"X": X_fit, "y": y_fit}
                    job_dump(
                        fit_data, filename=self.path / f"fit_data_{t}_{block_name}.pkl"
                    )
                    del X_tr, y_tr, X_fit, y_fit, X_val_fit, y_val_fit, fit_data
                    MemoryManager.trim_memory()

                    model.fit(
                        target_type=t,
                        filename=self.path / f"fit_data_{t}_{block_name}.pkl",
                        path=self.path,
                        val_path=self.path / f"val_{t}_{block_name}.pkl",
                    )

                    X_tr = job_load(self.path / f"fit_data_{t}_{block_name}.pkl")["X"]
                    y_tr = job_load(self.path / f"fit_data_{t}_{block_name}.pkl")["y"]
                    if conditional_precipitation:
                        eval_data = job_load(
                            self.path / f"eval_val_{t}_{block_name}.pkl"
                        )
                        X_val, y_val = eval_data["X"], eval_data["y"]
                        del eval_data
                    else:
                        X_val = job_load(
                            self.path / f"val_{t}_{block_name}.pkl"
                        )["X"]
                        y_val = job_load(
                            self.path / f"val_{t}_{block_name}.pkl"
                        )["y"]

                    for name in [
                        f"val_{t}_{block_name}.pkl",
                        f"fit_data_{t}_{block_name}.pkl",
                        *(
                            [f"eval_val_{t}_{block_name}.pkl"]
                            if conditional_precipitation
                            else []
                        ),
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
                    total_importance = feat_imp.sum()
                    if total_importance > 0:
                        feat_imp = feat_imp / total_importance
                    top_5 = feat_imp.sort_values(ascending=False).head(5)

                    logger.info(f"🔍 Top 5 for {t} ({block_name}):")
                    for rank, (name, val) in enumerate(top_5.items(), 1):
                        logger.info(f"{rank}. {name} ({val:.2%})")

                # Calibration loading
                X_cal = read_parquet(self.path / f"X_cal_{t}_{block_name}.parquet")
                y_cal = job_load(self.path / f"y_cal_{t}_{block_name}.pkl")
                X_cal_eval = read_parquet(
                    self.path / f"X_cal_eval_{t}_{block_name}.parquet"
                )
                y_cal_eval = job_load(
                    self.path / f"y_cal_eval_{t}_{block_name}.pkl"
                )

                if "time_step" in X_cal.columns:
                    X_cal = X_cal.drop(columns=["time_step"])
                if "time_step" in X_cal_eval.columns:
                    X_cal_eval = X_cal_eval.drop(columns=["time_step"])

                preds_val = model.predict(X_val)
                preds_cal_fit = model.predict(X_cal)
                preds_cal = model.predict(X_cal_eval)
                preds_val_unthresholded = preds_val.copy()

                # Fit bias correction on calibration-fit rows only.
                if conditional_precipitation:
                    cal_wet_mask = decode_rain_target(y_cal) > 0.1
                    if not cal_wet_mask.any():
                        raise ValueError(
                            "Medium-horizon precipitation bias correction requires "
                            "wet calibration-fit samples."
                        )
                    bias_corrector = fit_bias_corrector(
                        X_cal.loc[cal_wet_mask],
                        y_cal.loc[cal_wet_mask],
                        preds_cal_fit[cal_wet_mask],
                    )
                else:
                    bias_corrector = fit_bias_corrector(
                        X_cal, y_cal, preds_cal_fit
                    )
                preds_val_corrected = apply_bias_correction(
                    X_val,
                    bias_corrector,
                    t,
                    Series(preds_val, index=X_val.index),
                ).to_numpy()
                preds_cal_corrected = apply_bias_correction(
                    X_cal_eval,
                    bias_corrector,
                    t,
                    Series(preds_cal, index=X_cal_eval.index),
                ).to_numpy()

                # --- CALCULATION OF METRICS ---
                if t == "precip":
                    if precip_val_probs is None:
                        raise RuntimeError(
                            "Precipitation validation probabilities were not computed."
                        )
                    if len(precip_val_probs) != len(preds_val):
                        raise ValueError(
                            "Precipitation probabilities and regression predictions "
                            "must align."
                        )
                    rain_signal = precip_val_probs >= optimal_threshold
                    prob_model = job_load(
                        self.path / f"model_precip_prob_{block_name}.pkl"
                    )
                    X_cal_prob = (
                        X_cal_eval.drop(columns=["time_step"])
                        if "time_step" in X_cal_eval.columns
                        else X_cal_eval
                    )
                    cal_rain_probs = prob_model.predict_proba(X_cal_prob)[:, 1]
                    del prob_model, X_cal_prob
                    if len(cal_rain_probs) != len(preds_cal_corrected):
                        raise ValueError(
                            "Precipitation calibration probabilities and regression "
                            "predictions must align."
                        )
                    preds_val = where(rain_signal, preds_val, 0.0)
                    preds_val_corrected = where(
                        rain_signal, preds_val_corrected, 0.0
                    )
                    preds_cal = where(
                        cal_rain_probs >= optimal_threshold, preds_cal, 0.0
                    )
                    preds_cal_corrected = where(
                        cal_rain_probs >= optimal_threshold,
                        preds_cal_corrected,
                        0.0,
                    )
                preds_val = _constrain_prediction(t, preds_val)
                preds_val_corrected = _constrain_prediction(
                    t, preds_val_corrected
                )
                preds_cal = _constrain_prediction(t, preds_cal)
                preds_cal_corrected = _constrain_prediction(
                    t, preds_cal_corrected
                )

                metric_actual = _metric_values(t, y_val)
                raw_validation_error = _primary_error(
                    t, metric_actual, _metric_values(t, preds_val)
                )
                corrected_validation_error = _primary_error(
                    t, metric_actual, _metric_values(t, preds_val_corrected)
                )
                if corrected_validation_error >= raw_validation_error:
                    logger.warning(
                        f"Bias correction rejected for {t}/{block_name}: "
                        f"validation error {corrected_validation_error:.4f} "
                        f">= raw error {raw_validation_error:.4f}."
                    )
                    bias_corrector = None
                    preds_val_corrected = preds_val.copy()
                    preds_cal_corrected = preds_cal.copy()

                # Deleting temp files
                for name in [
                    f"X_cal_{t}_{block_name}.parquet",
                    f"y_cal_{t}_{block_name}.pkl",
                    f"X_cal_eval_{t}_{block_name}.parquet",
                    f"y_cal_eval_{t}_{block_name}.pkl",
                    f"X_cal_threshold_{t}_{block_name}.parquet",
                    f"y_cal_threshold_{t}_{block_name}.pkl",
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
                bundle = {
                    "model": model,
                    "m_low": m_low,
                    "m_high": m_high,
                    "bias_corrector": bias_corrector,
                    "features": used_features,
                    "target_transform": "log1p" if t == "precip" else None,
                }
                if t == "precip":
                    bundle["optimal_threshold"] = optimal_threshold
                    bundle["precip_strategy_version"] = 3
                job_dump(bundle, self.path / f"model_{t}_{block_name}.pkl", compress=6)
                job_dump(meta_data, self.path / "global_meta.pkl")

                # Assessment uses physical units and disjoint calibration evaluation rows.
                from sklearn.metrics import mean_pinball_loss

                y_tr_metric = _metric_values(t, y_tr)
                y_val_metric = _metric_values(t, y_val)
                y_cal_metric = _metric_values(t, y_cal_eval)
                preds_val_metric = _metric_values(t, preds_val)
                preds_val_corr_metric = _metric_values(t, preds_val_corrected)
                preds_cal_corr_metric = _metric_values(t, preds_cal_corrected)
                preds_tr_base = _constrain_prediction(t, model.predict(X_tr))
                preds_tr_metric = _metric_values(t, preds_tr_base)

                raw_metrics = regression_metrics(y_val_metric, preds_val_metric)
                corrected_metrics = regression_metrics(
                    y_val_metric, preds_val_corr_metric
                )
                calibration_metrics = regression_metrics(
                    y_cal_metric, preds_cal_corr_metric
                )
                train_metrics = regression_metrics(y_tr_metric, preds_tr_metric)

                if t in ["wind", "gusts", "hum"]:
                    primary_raw = raw_metrics["rmse"]
                    primary_corrected = corrected_metrics["rmse"]
                    primary_name = "RMSE"
                else:
                    primary_raw = raw_metrics["mae"]
                    primary_corrected = corrected_metrics["mae"]
                    primary_name = "MAE"

                q10_str, q90_str = "N/A", "N/A"
                quantile_metric_mask = (
                    y_val_metric > 0.1
                    if conditional_precipitation
                    else slice(None)
                )
                quantile_metric_target = y_val_metric[quantile_metric_mask]
                if m_low is not None:
                    q10_preds = m_low.predict(X_val)
                    if t == "cloud":
                        q10_preds = q10_preds * 100.0
                    elif t == "precip":
                        q10_preds = decode_rain_target(q10_preds)
                    q10_str = (
                        f"{mean_pinball_loss(quantile_metric_target, q10_preds[quantile_metric_mask], alpha=0.1):.3f}"
                    )

                if m_high is not None:
                    q90_preds = m_high.predict(X_val)
                    if t == "cloud":
                        q90_preds = q90_preds * 100.0
                    elif t == "precip":
                        q90_preds = decode_rain_target(q90_preds)
                    q90_str = (
                        f"{mean_pinball_loss(quantile_metric_target, q90_preds[quantile_metric_mask], alpha=0.9):.3f}"
                    )

                if t == "precip":
                    train_wet = y_tr_metric > 0.1
                    validation_wet = y_val_metric > 0.1
                    if train_wet.any() and validation_wet.any():
                        train_metrics = regression_metrics(
                            y_tr_metric[train_wet], preds_tr_metric[train_wet]
                        )
                        wet_validation_predictions = _metric_values(
                            t,
                            _constrain_prediction(t, preds_val_unthresholded),
                        )
                        wet_validation_metrics = regression_metrics(
                            y_val_metric[validation_wet],
                            wet_validation_predictions[validation_wet],
                        )
                        overfit_ratio = (
                            wet_validation_metrics["mae"] / train_metrics["mae"]
                            if train_metrics["mae"] > 0
                            else nan
                        )
                    else:
                        overfit_ratio = nan
                else:
                    overfit_ratio = (
                        raw_metrics["mae"] / train_metrics["mae"]
                        if train_metrics["mae"] > 0
                        else nan
                    )

                error_delta = primary_corrected - primary_raw
                mae_delta = corrected_metrics["mae"] - raw_metrics["mae"]
                mae_ratio = (
                    corrected_metrics["mae"] / raw_metrics["mae"]
                    if raw_metrics["mae"] > 0
                    else nan
                )
                task_duration = perf_counter() - start_time_model
                eta_estimator.record(block_name, task_duration)
                eta_seconds, eta_coverage = eta_estimator.estimate_remaining()
                if eta_seconds == 0:
                    eta_text = "remaining tasks complete"
                elif eta_seconds is None:
                    eta_text = "warming up"
                else:
                    eta_text = (
                        f"~{eta_seconds / 3600:.1f}h remaining "
                        f"(block-history coverage {eta_coverage:.0%})"
                    )

                logger.info(
                    f"[{t} | {block_name.upper()}] {task_duration / 60:.1f}min. "
                    f"{eta_estimator.completed_count}/{len(pending_tasks)} tasks | "
                    f"ETA: {eta_text}\n"
                    f"    Val MAE raw/corrected: {raw_metrics['mae']:.3f}/"
                    f"{corrected_metrics['mae']:.3f} (delta {mae_delta:+.3f}, "
                    f"ratio {mae_ratio:.3f}) | "
                    f"RMSE raw/corrected: {raw_metrics['rmse']:.3f}/"
                    f"{corrected_metrics['rmse']:.3f}\n"
                    f"    Pearson r raw/corrected: "
                    f"{raw_metrics['pearson_r']:.3f}/"
                    f"{corrected_metrics['pearson_r']:.3f} | "
                    f"{primary_name} delta: {error_delta:+.3f}\n"
                    f"    Train MAE: {train_metrics['mae']:.3f} | "
                    f"Calibration holdout MAE: {calibration_metrics['mae']:.3f} | "
                    f"Wet-event MAE val/train: {overfit_ratio:.3f}\n"
                    f"    Quantile pinball: q10={q10_str} | q90={q90_str}"
                )

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

                del X_tr, y_tr, X_val, y_val, X_cal, y_cal, X_cal_eval, y_cal_eval
                del (
                    model,
                    preds_val,
                    preds_cal,
                    preds_cal_fit,
                    bias_corrector,
                    preds_cal_corrected,
                    weights,
                )
                del importances
                MemoryManager.trim_memory()

        logger.info(
            f"\n🎉 Training completed in {(perf_counter() - start_time) / 3600:.1f}h !"
        )