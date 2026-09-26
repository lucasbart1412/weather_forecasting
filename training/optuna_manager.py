from training._runtime import *


class OptunaTrainingMixin:
    def optimize_short_block(
        self,
        X_tr: DataFrame,
        y_tr: Series,
        X_val: DataFrame,
        y_val: Series,
        target_name: str,
        weight: Optional[Series] = None,
    ) -> Tuple[Dict, int]:
        """Optuna optimization for short block."""
        if len(X_tr) < 500:
            raise ValueError('Minimum 500 samples required for Optuna.')

        logger.info(f"🚀 Optuna for {target_name} (Short)...")
        db_path = self.path / f"optuna_study_{target_name}.db"
        storage_url = f"sqlite:///{db_path}"
        iter_per_trial = {}
        first_trial = None

        # Metrics by Target
        if target_name in ["wind", "gusts", "hum"]:
            internal_metric, prune_metric = "rmse", "rmse"
        elif target_name == "precip":
            internal_metric, prune_metric = "tweedie", "tweedie"
        else:
            internal_metric, prune_metric = "mae", "l1"

        # NumPy Conversion (CRITICAL for Optuna)
        base_param = {}
        base_param = self.add_feature_penalty(param=base_param, X_tr=X_tr)
        feature_penalties = base_param["feature_penalty"]
        feature_names_list = list(X_tr.columns)

        # NumPy Conversion (CRITICAL for Optuna)
        X_tr_np = ascontiguousarray(X_tr.to_numpy(dtype=float32))
        X_val_np = ascontiguousarray(X_val.to_numpy(dtype=float32))
        y_tr_np = ascontiguousarray(
            y_tr.to_numpy(dtype=float32) if hasattr(y_tr, "to_numpy") else y_tr
        )
        y_val_np = ascontiguousarray(
            y_val.to_numpy(dtype=float32) if hasattr(y_val, "to_numpy") else y_val
        )
        weight_np = (
            ascontiguousarray(
                weight.to_numpy(dtype=float32)
                if hasattr(weight, "to_numpy")
                else weight
            )
            if weight is not None
            else None
        )

        def objective(trial, target_type=target_name):
            nonlocal first_trial
            if first_trial is None:
                first_trial = trial.number

            colsample_min = 0.5 if target_type == "precip" else 0.4
            colsample_max = 0.75 if target_type == "precip" else 0.6

            param = {
                "n_jobs": -1,
                "metric": internal_metric,
                "verbosity": -1,
                "importance_type": "gain",
                "boosting_type": "gbdt",
                "n_estimators": 2000,
                "learning_rate": trial.suggest_float("learning_rate", 0.04, 0.15, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 25, 50),
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "min_child_samples": trial.suggest_int("min_child_samples", 200, 500),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 15.0, log=True),
                "colsample_bytree": trial.suggest_float("colsample_bytree", colsample_min, colsample_max),
                "subsample": trial.suggest_float("subsample", 0.7, 0.9),
                "feature_penalty": feature_penalties,
            }

            # --- MAJOR ⚡ OPTIMIZATION: GPU injection if available ---
            if WeatherConfig.HAS_GPU:
                param.update({
                    'device_type': 'gpu',
                    'max_bin': 63,
                    'gpu_use_dp': False
                })
            else:
                param.update({'max_bin': 255})

            # Goal by Target
            if target_type == "temp":
                param.update({"objective": "huber"})
            elif target_type == "precip":
                param.update({"objective": "tweedie", "tweedie_variance_power": 1.5})
            elif target_type == "cloud":
                param.update({"objective": "huber"})
            else:
                param.update({"objective": "regression"})

            pruning_callback = LightGBMPruningCallback(trial, prune_metric)
            current_iteration_tracker = [0]

            def tracking_callback(env):
                current_iteration_tracker[0] = env.iteration

            model = LGBMRegressor(**param)
            try:
                # ✅ LIGHTGBM CORRECTION: eval_X/eval_y instead of eval_set
                model.fit(
                    X_tr_np,
                    y_tr_np,
                    eval_set=[(X_tr_np, y_tr_np), (X_val_np, y_val_np)],
                    eval_names=["training", "valid_0"],
                    sample_weight=weight_np,
                    feature_name=feature_names_list,
                    callbacks=[
                        early_stopping(stopping_rounds=150, verbose=False),
                        log_evaluation(0),
                        tracking_callback,
                        pruning_callback,
                    ],
                )
                best_it = (
                    model.best_iteration_
                    if (model.best_iteration_ and model.best_iteration_ > 0)
                    else model.n_estimators
                )
                iter_per_trial[trial.number] = best_it
            except TrialPruned:
                pruned_iteration = current_iteration_tracker[0] + 1
                iter_per_trial[trial.number] = pruned_iteration
                if "model" in locals():
                    del model
                    gc.collect()
                raise

            preds_val = model.predict(X_val_np)
            train_preds = model.predict(X_tr_np)

            if target_type in ["wind", "gusts", "hum"]:
                score_val = root_mean_squared_error(y_val_np, preds_val)
                score_train = root_mean_squared_error(y_tr_np, train_preds)
            elif target_type == "precip":
                score_val = model.best_score_["valid_0"]["tweedie"]
                score_train = model.best_score_["training"]["tweedie"]
            else:
                score_val = mean_absolute_error(y_val_np, preds_val)
                score_train = mean_absolute_error(y_tr_np, train_preds)

            ratio = score_val / score_train if score_train != 0 else float("inf")

            if "model" in locals():
                del model
                gc.collect()

            # Penalty overfit
            tolerance = 1.25
            penalty_factor = 0.2
            penalty = (
                0.0
                if ratio <= tolerance
                else score_val * penalty_factor * (exp(ratio - tolerance) - 1)
            )

            return score_val + penalty

        study = create_study(
            study_name=f"study_{target_name}",
            direction="minimize",
            storage=storage_url,
            load_if_exists=True,
            pruner=pruners.MedianPruner(),
        )

        study.optimize(objective, n_trials=50 if target_name == "temp" else 30)

        best_params = study.best_params
        best_params.update(
            {
                "n_estimators": 10000,
                "boosting_type": "gbdt",
                "metric": internal_metric,
                "verbosity": -1,
                "n_jobs": -1,
                "importance_type": "gain",
                "max_bin": 63 if WeatherConfig.HAS_GPU else 255,
            }
        )

        if WeatherConfig.HAS_GPU:
            best_params.update(
                {
                    "device_type": "gpu",
                    "gpu_use_dp": False,
                }
            )

        # Final Adjustments
        if target_name == "temp":
            best_params.update({"objective": "huber"})
            best_params["n_estimators"] = 10000
        elif target_name == "precip":
            best_params.update({"objective": "tweedie", "tweedie_variance_power": 1.5})
        elif target_name == "cloud":
            best_params.update({"objective": "huber"})
        else:
            best_params.update({"objective": "regression"})

        best_params["learning_rate"] = max(0.01, best_params["learning_rate"] / 4)
        return best_params, sum(iter_per_trial.values())

