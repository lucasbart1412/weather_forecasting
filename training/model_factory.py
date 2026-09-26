from pandas import DataFrame

from training._runtime import *
from training.stacking_impl import TimeSeriesStackingTrainer
from typing import Any, List, Optional

class ModelFactoryMixin:
    def get_advanced_models(
        self,
        block_name: str,
        target_type: str,
        lengths: Optional[List] = None,
        X_tr: Optional[DataFrame] = None,
    ) -> Any:
        """MEDIUM BLOCK (Stacking H30-H72)"""

        # =====================================================================
        # MEDIUM BLOCK (Stacking H30-H72)
        # =====================================================================
        if block_name == "medium":
            lgb_params = {
                "n_estimators": 10000,
                "learning_rate": 0.02,
                "num_leaves": 31,
                "max_depth": 5,
                "min_child_samples": 450,
                "reg_alpha": 4.0,
                "reg_lambda": 5.0,
                "colsample_bytree": 0.5,
                "subsample": 0.75,
                "n_jobs": -1,
                "verbose": -1,
            }

            xgb_params = {
                "n_estimators": 10000,
                "learning_rate": 0.02,
                "max_depth": 5,
                "min_child_weight": 450,
                "reg_alpha": 4.0,
                "reg_lambda": 5.0,
                "subsample": 0.75,
                "colsample_bytree": 0.5,
                "n_jobs": -1,
                "verbosity": 0,
            }

            cat_params = {
                "iterations": 10000,
                "learning_rate": 0.02,
                "depth": 5,
                "l2_leaf_reg": 6.0,
                "bootstrap_type": "Bernoulli",
                "subsample": 0.75,
                "allow_writing_files": False,
                "logging_level": "Silent",
            }

            # --- GPU ⚡ ACTIVATION (MEDIUM BLOCK) ---
            if WeatherConfig.HAS_GPU:
                lgb_params.update({'device_type': 'gpu', 'max_bin': 63, 'gpu_use_dp': False})
                xgb_params.update({'device': 'cuda', 'tree_method': 'hist'})
                cat_params.update({'task_type': 'GPU'})

            meta_cv = TimeSeriesSplit(n_splits=3)
            final_estimator = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=meta_cv)

            # Adaptations by target
            if target_type in ["temp", "dew"]:
                lgb_params.update({"objective": "huber", "alpha": 1.2})
                xgb_params.update({"objective": "reg:absoluteerror"})
                cat_params.update({"loss_function": "MAE"})

            elif target_type == "precip":
                lgb_params.update(
                    {
                        "objective": "tweedie",
                        "tweedie_variance_power": 1.5,
                        "learning_rate": 0.015,
                        "colsample_bytree": 0.7,
                    }
                )
                xgb_params.update(
                    {
                        "objective": "reg:tweedie",
                        "tweedie_variance_power": 1.5,
                        "colsample_bytree": 0.7,
                    }
                )
                cat_params.update({"loss_function": "Tweedie:variance_power=1.5"})
                final_estimator = TweedieRegressor(power=1.5, link="log", max_iter=500)

            elif target_type in ["wind", "gusts"]:
                lgb_params.update({"objective": "huber", "alpha": 1.3})
                xgb_params.update({"objective": "reg:absoluteerror"})
                cat_params.update({"loss_function": "MAE"})

            elif target_type == "cloud":
                lgb_params.update(
                    {
                        "objective": "regression_l1",
                        "metric": "mae",
                        "colsample_bytree": 0.5,
                    }
                )
                xgb_params.update(
                    {
                        "objective": "reg:absoluteerror",
                        "eval_metric": "mae",
                        "colsample_bytree": 0.5,
                    }
                )
                cat_params.update({"loss_function": "MAE"})
                meta_cv = TimeSeriesSplit(n_splits=3)
                final_estimator = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=meta_cv)

            elif target_type == "hum":
                lgb_params.update({"objective": "regression", "metric": "rmse"})
                xgb_params.update({"objective": "reg:squarederror"})
                cat_params.update({"loss_function": "RMSE"})

            # RIGHTS
            Lgb = LGBMRegressor(**lgb_params)
            Xgb = XGBRegressor(**xgb_params)
            Cat = CatBoostRegressor(**cat_params)

            final_lgb_params = lgb_params.copy()
            final_xgb_params = xgb_params.copy()
            final_cat_params = cat_params.copy()

            # Anti-overfit safety
            if lengths is None or X_tr is None:
                final_lgb_params["n_estimators"] = 2000
                final_xgb_params["n_estimators"] = 2000
                final_cat_params["iterations"] = 2000

            return TimeSeriesStackingTrainer(
                estimators=[("cat", Cat), ("lgb", Lgb), ("xgb", Xgb)],
                final_estimator=final_estimator,
                cv=self.custom_ridgecv_tscv if (lengths and X_tr is not None) else 5,
                final_estimators=[
                    ("cat", CatBoostRegressor(**final_cat_params)),
                    ("lgb", LGBMRegressor(**final_lgb_params)),
                    ("xgb", XGBRegressor(**final_xgb_params)),
                ],
            )

        # =====================================================================
        # SHORT BLOCK (H1-H18) - Without Optuna for some targets
        # =====================================================================
        elif block_name == "short":
            params = {
                "metric": "mae",
                "boosting_type": "gbdt",
                "max_depth": 6,
                "num_leaves": 40,
                "min_child_samples": 415,
                "learning_rate": 0.02,
                "n_estimators": 10000,
                "reg_alpha": 0.11,
                "reg_lambda": 1.2,
                "colsample_bytree": 0.46,
                "subsample": 0.81,
                "importance_type": "gain",
                "verbose": -1,
                "n_jobs": -1,
            }

            # --- GPU ⚡ ACTIVATION (SHORT BLOCK) ---
            if WeatherConfig.HAS_GPU:
                params.update({'device_type': 'gpu', 'max_bin': 63, 'gpu_use_dp': False})

            if target_type in ["temp", "dew"]:
                params.update({"objective": "huber", "alpha": 1.2})
                params["n_estimators"] = 10000
            elif target_type == "precip":
                params.update(
                    {
                        "objective": "tweedie",
                        "tweedie_variance_power": 1.5,
                        "colsample_bytree": 0.7,
                    }
                )
                params["max_depth"] = 5
                params["reg_alpha"] = 0.5
                params["reg_lambda"] = 2.0
            elif target_type == "cloud":
                params.update({"objective": "regression_l1", "metric": "mae"})
                params["n_estimators"] = 10000
            elif target_type in ["wind", "gusts"]:
                params["n_estimators"] = 10000
                params.update({"objective": "huber", "alpha": 1.3})
                params["metric"] = "rmse"
                params["min_child_samples"] = 250
                params["num_leaves"] = 50
            else:
                params.update({"objective": "regression"})

            if target_type == "tide":
                params = {
                    "boosting_type": "gbdt",
                    "objective": "regression_l1",
                    "metric": "mae",
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "max_depth": 6,
                    "min_child_samples": 200,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                    "colsample_bytree": 0.7,
                    "subsample": 0.9,
                    "n_estimators": 10000,
                    "importance_type": "gain",
                    "verbosity": -1,
                    "n_jobs": 1,
                }
                if WeatherConfig.HAS_GPU:
                    params.update(
                        {
                            "device_type": "gpu",
                            "max_bin": 63,
                            "gpu_use_dp": False,
                        }
                    )

            params = self.add_feature_penalty(param=params, X_tr=X_tr)
            return LGBMRegressor(**params)

        # =====================================================================
        # LONG BLOCK (H84-H168) - Focus on macro trends
        # =====================================================================
        else:
            params = {
                "n_estimators": 2000,
                "learning_rate": 0.015,
                "num_leaves": 15,
                "max_depth": 4,
                "min_child_samples": 500,
                "reg_alpha": 5.0,
                "reg_lambda": 5.5,
                "colsample_bytree": 0.4,
                "subsample": 0.6,
                "n_jobs": -1,
                "verbose": -1,
            }

            # --- GPU ⚡ ACTIVATION (LONG BLOCK) ---
            if WeatherConfig.HAS_GPU:
                params.update({'device_type': 'gpu', 'max_bin': 63, 'gpu_use_dp': False})

            if target_type in ["temp", "dew"]:
                params.update({"objective": "huber", "metric": "mae"})
            elif target_type == "precip":
                params.update(
                    {
                        "objective": "tweedie",
                        "tweedie_variance_power": 1.5,
                        "colsample_bytree": 0.6,
                    }
                )
            elif target_type == "cloud":
                params.update(
                    {
                        "objective": "regression_l1",
                        "metric": "mae",
                        "colsample_bytree": 0.4,
                    }
                )
            elif target_type in ["wind", "gusts"]:
                params.update({"objective": "regression", "metric": "rmse"})
            else:
                params.update({"objective": "regression"})

            return LGBMRegressor(**params)

