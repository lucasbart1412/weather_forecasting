from datetime import datetime
from typing import Any, List, Optional, Tuple
import warnings

import numpy as np
from training._runtime import *
from training.memory_impl import MemoryManager
from sklearn.base import BaseEstimator, RegressorMixin, clone
from models import InplacePredictPreprocessor

class TimeSeriesStackingTrainer(BaseEstimator, RegressorMixin):
    """
    Post-fit attributes

    Memory optimizations:
    - NumPy buffers pre-allocated to avoid copies
    - Aggressive release after each fold
    - Strict float32 conversion
    """

    def __init__(
        self,
        estimators: list[tuple[str, BaseEstimator]],
        final_estimator: BaseEstimator,
        cv: Any,
        early_stopping_rounds: int = 75,
        final_estimators: Optional[List[Tuple[str, BaseEstimator]]] = None,
    ):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.early_stopping_rounds = early_stopping_rounds
        self.final_estimators = (
            final_estimators if final_estimators is not None else estimators
        )

        # Post-fit attributes
        self.estimators_ = []
        self.final_estimator_ = None
        self.is_fitted_ = False
        self.best_iterations_ = {}
        self.feature_names_in_ = None
        self.meta_scaler_means_ = None
        self.meta_scaler_stds_ = None
        self.target_type_ = None

    def _get_early_stopping_params(
        self,
        model_name: str,
        model_obj: BaseEstimator,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        """
        Returns the early stopping settings appropriate for each bookstore.

        ⚠️ FIX LIGHTGBM: Use eval_X/eval_y instead of eval_set (deprecated)
        """
        name_lower = model_name.lower()
        params = {}

        if "lgb" in name_lower or "lightgbm" in name_lower:
            params["eval_set"] = [(X_val, y_val)]
            params["callbacks"] = [
                early_stopping(
                    stopping_rounds=self.early_stopping_rounds, verbose=False
                ),
                log_evaluation(0),
            ]
        elif "xgb" in name_lower:
            params["eval_set"] = [(X_val, y_val)]
            params["verbose"] = False
            model_obj.set_params(early_stopping_rounds=self.early_stopping_rounds)
        elif "cat" in name_lower:
            params["eval_set"] = [(X_val, y_val)]
            params["early_stopping_rounds"] = self.early_stopping_rounds

        return params

    def _apply_time_series_imputation(self, X_np: np.ndarray) -> np.ndarray:
        """Chronological strict imputation (no future leakage -> past)."""
        mask = np.isnan(X_np)
        if not mask.any():
            return X_np

        n_samples, n_features = X_np.shape
        for j in range(n_features):
            col = X_np[:, j]
            col_mask = mask[:, j]
            if not col_mask.any():
                continue

            first_valid_idx = np.where(~col_mask)[0]
            if len(first_valid_idx) == 0:
                np.putmask(col, np.isnan(col), 0.0)
                continue

            last_valid_val = col[first_valid_idx[0]]
            for i in range(n_samples):
                if col_mask[i]:
                    col[i] = last_valid_val
                else:
                    last_valid_val = col[i]
        return X_np

    def _inplace_impute_and_scale(
        self,
        X_mat: np.ndarray,
        feature_names: Optional[List[str]] = None,
        means: Optional[np.ndarray] = None,
        stds: Optional[np.ndarray] = None,
        medians: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Imputation + Secure in-place Z-score Scaling."""
        X_mat = np.asarray(X_mat, dtype=np.float32).copy()

        # 'precip' column identification
        precip_idx = -1
        if feature_names is not None:
            try:
                precip_idx = list(feature_names).index("precip")
            except ValueError:
                pass

        # Allocation
        nan_mask = np.isnan(X_mat)
        if nan_mask.any():
            if medians is None:
                raise ValueError('Training medians required.')

            for j in range(X_mat.shape[1]):
                mask_j = nan_mask[:, j]
                if mask_j.any():
                    if j == precip_idx:
                        np.putmask(X_mat[:, j], mask_j, 0.0)
                    else:
                        np.putmask(X_mat[:, j], mask_j, medians[j])

        # Scaling
        if means is None or stds is None:
            means = np.mean(X_mat, axis=0, dtype=np.float32)
            stds = np.std(X_mat, axis=0, ddof=0, dtype=np.float32)
            np.putmask(stds, stds == 0.0, 1.0)

            if precip_idx != -1:
                np.putmask(stds, stds == 0.0, 1.0)
                np.putmask(means, np.isnan(means), 0.0)
        else:
            if precip_idx != -1:
                np.putmask(stds, stds == 0.0, 1.0)
                np.putmask(means, np.isnan(means), 0.0)

        X_mat -= means
        X_mat /= stds

        return X_mat, means, stds, medians

    def fit(
        self,
        filename: str,
        path: Optional[str] = None,
        val_path: Optional[str] = None,
        target_type: Optional[str] = None,
    ) -> "TimeSeriesStackingTrainer":
        """
        Drives stacking with OOF time cross-validation.

        Args:
            filename: Path to pickle file containing {'X': DataFrame, 'y': Series}
            path: Output directory for artifacts
            val_path: Path to independent final validation set
            target_type: Target type ('temp', 'precip', etc.)
        """
        # Single loading
        data = job_load(filename)
        X, y = data["X"], data["y"]
        del data
        MemoryManager.trim_memory()

        self.final_estimator_ = clone(self.final_estimator)

        # Split generation
        if callable(self.cv):
            splits = list(self.cv(X))
        elif isinstance(self.cv, int):
            splits = list(TimeSeriesSplit(n_splits=self.cv, gap=72).split(X, y))
        else:
            splits = list(self.cv)

        # NumPy float32 conversion
        if hasattr(X, "columns"):
            cols_to_keep = [c for c in X.columns if c != "time_step"]
            self.feature_names_in_ = np.asarray(cols_to_keep)
            X_numpy = np.ascontiguousarray(X[cols_to_keep].to_numpy(dtype=np.float32))
        else:
            self.feature_names_in_ = None
            X_numpy = np.ascontiguousarray(np.asarray(X, dtype=np.float32))

        y_numpy = (
            y.to_numpy(dtype=np.float32, copy=False).flatten()
            if hasattr(y, "to_numpy")
            else np.asarray(y, dtype=np.float32).flatten()
        )

        del X, y
        MemoryManager.trim_memory()

        # In-place allocation
        X_numpy = self._apply_time_series_imputation(X_numpy)
        MemoryManager.trim_memory()

        logger.info('🧮 Calculating global medians...')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            global_medians = np.nanmedian(X_numpy, axis=0)
        global_medians[np.isnan(global_medians)] = 0.0

        # Size Calculation
        max_train_size = max(len(tr) for tr, _ in splits)
        max_val_size = max(len(val) for _, val in splits)
        n_features = X_numpy.shape[1]
        n_estimators = len(self.estimators)
        total_val_samples = sum(len(val) for _, val in splits)

        # Pre-allocated buffers (Zero allocation during the loop)
        X_tr_buffer = np.empty((max_train_size, n_features), dtype=np.float32)
        y_tr_buffer = np.empty(max_train_size, dtype=np.float32)
        X_val_buffer = np.empty((max_val_size, n_features), dtype=np.float32)
        y_val_buffer = np.empty(max_val_size, dtype=np.float32)

        X_meta = np.empty((total_val_samples, n_estimators), dtype=np.float32)
        y_meta = np.empty(total_val_samples, dtype=np.float32)
        meta_ptr = 0

        self.best_iterations_ = {name: [] for name, _ in self.estimators}

        # Loop oof
        for i, (train_idx, val_idx) in enumerate(splits, start=1):
            logger.info(
                f"--- Fold {i}/{len(splits)} : Train={len(train_idx)}, Val={len(val_idx)} ---"
            )
            if len(val_idx) == 0:
                continue

            tr_len, val_len = len(train_idx), len(val_idx)

            # Extraction via take() (no copy)
            X_tr_fold = X_tr_buffer[:tr_len]
            np.take(X_numpy, train_idx, axis=0, out=X_tr_fold)
            y_tr_fold = y_tr_buffer[:tr_len]
            np.take(y_numpy, train_idx, axis=0, out=y_tr_fold)

            X_val_fold = X_val_buffer[:val_len]
            np.take(X_numpy, val_idx, axis=0, out=X_val_fold)
            y_val_fold = y_val_buffer[:val_len]
            np.take(y_numpy, val_idx, axis=0, out=y_val_fold)

            # Imputation + Scaling
            X_tr_fold, f_means, f_stds, f_medians = self._inplace_impute_and_scale(
                X_tr_fold,
                feature_names=list(self.feature_names_in_)
                if self.feature_names_in_ is not None
                else None,
            )
            X_val_fold, _, _, _ = self._inplace_impute_and_scale(
                X_val_fold,
                means=f_means,
                stds=f_stds,
                medians=f_medians,
                feature_names=list(self.feature_names_in_)
                if self.feature_names_in_ is not None
                else None,
            )

            y_meta[meta_ptr : meta_ptr + val_len] = y_val_fold

            # Basic estimator training
            for est_idx, (name, est) in enumerate(self.estimators):
                logger.info(f"Training '{name}'...")
                model_cloned = clone(est)
                name_lower = name.lower()
                stop_params = self._get_early_stopping_params(
                    name, model_cloned, X_val_fold, y_val_fold
                )

                if "cat" in name_lower:
                    model_cloned.set_params(
                        allow_writing_files=False, save_snapshot=False
                    )
                    stop_params["eval_set"] = [(X_val_fold, y_val_fold)]
                    model_cloned.fit(X_tr_fold, y_tr_fold, **stop_params)
                elif "lgb" in name_lower or "lightgbm" in name_lower:
                    feature_list = (
                        list(self.feature_names_in_)
                        if self.feature_names_in_ is not None
                        else "auto"
                    )
                    # ✅ LIGHTGBM CORRECTION: eval_X/eval_y in stop_params
                    model_cloned.fit(
                        X_tr_fold, y_tr_fold, feature_name=feature_list, **stop_params
                    )
                else:
                    model_cloned.fit(X_tr_fold, y_tr_fold, **stop_params)

                # Tracking best iteration
                if "lgb" in name_lower:
                    self.best_iterations_[name].append(
                        getattr(
                            model_cloned, "best_iteration_", model_cloned.n_estimators
                        )
                    )
                elif "xgb" in name_lower:
                    self.best_iterations_[name].append(model_cloned.best_iteration)
                elif "cat" in name_lower:
                    self.best_iterations_[name].append(
                        model_cloned.get_best_iteration()
                    )

                # Oof Prediction
                X_meta[meta_ptr : meta_ptr + val_len, est_idx] = model_cloned.predict(
                    X_val_fold
                ).flatten()

                # Aggressive Release
                MemoryManager.free_lgbm_booster(model_cloned)
                del model_cloned
                MemoryManager.trim_memory()

            meta_ptr += val_len
            MemoryManager.trim_memory()

        # Oof backup (optional)
        if path is not None:
            Path(path).mkdir(parents=True, exist_ok=True)
            job_dump(
                {
                    "model_names": [name for name, _ in self.estimators],
                    "oof_predictions": X_meta,
                    "true_targets": y_meta,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                filename=f"{path}/oof_predictions.pkl",
            )

        # Meta-normalization
        self.target_type_ = target_type
        if target_type != "precip":
            self.meta_scaler_means_ = X_meta.mean(axis=0)
            self.meta_scaler_stds_ = X_meta.std(axis=0)

            X_meta -= self.meta_scaler_means_
            safe_stds = np.where(self.meta_scaler_stds_ == 0, 1.0, self.meta_scaler_stds_)
            X_meta /= safe_stds
            X_meta = np.nan_to_num(X_meta, nan=0.0, posinf=0.0, neginf=0.0)

        # Fit meta-model
        self.final_estimator_.fit(X_meta, y_meta)

        # Final normalization of X_numpy
        X_numpy, final_means, final_stds, final_medians = (
            self._inplace_impute_and_scale(
                X_numpy,
                feature_names=list(self.feature_names_in_)
                if self.feature_names_in_ is not None
                else None,
            )
        )
        MemoryManager.trim_memory()

        # Loading final validation
        X_val, y_val = None, None
        if val_path is not None:
            logger.info('💾 Loading final validation...')
            val_data = job_load(val_path)
            X_v_raw, y_v_raw = val_data["X"], val_data["y"]
            del val_data

            if hasattr(X_v_raw, "columns"):
                cols_to_keep = [c for c in X_v_raw.columns if c != "time_step"]
                X_val = np.ascontiguousarray(X_v_raw[cols_to_keep].to_numpy(dtype=np.float32))
            else:
                X_val = np.ascontiguousarray(np.asarray(X_v_raw, dtype=np.float32))

            y_val = (
                y_v_raw.to_numpy(dtype=np.float32, copy=False).flatten()
                if hasattr(y_v_raw, "to_numpy")
                else np.asarray(y_v_raw, dtype=np.float32).flatten()
            )
            del X_v_raw, y_v_raw

            X_val, _, _, _ = self._inplace_impute_and_scale(
                X_val,
                means=final_means,
                stds=final_stds,
                medians=final_medians,
                feature_names=list(self.feature_names_in_)
                if self.feature_names_in_ is not None
                else None,
            )
            MemoryManager.trim_memory()
        else:
            X_val = X_numpy
            y_val = y_numpy

        # Final training on 100% of the data
        logger.info('\n--- Final training on 100% of the data ---')
        self.estimators_ = []

        for name, est in self.final_estimators:
            logger.info(f"Final training '{name}'...")
            model_cloned = clone(est)
            name_lower = name.lower()

            stop_params = self._get_early_stopping_params(
                name, model_cloned, X_val, y_val
            )

            if "lgb" in name_lower or "lightgbm" in name_lower:
                model_cloned.set_params(free_raw_data=True)
                feature_list = (
                    list(self.feature_names_in_)
                    if self.feature_names_in_ is not None
                    else "auto"
                )
                model_cloned.fit(
                    X_numpy, y_numpy, feature_name=feature_list, **stop_params
                )
            elif "cat" in name_lower:
                model_cloned.set_params(allow_writing_files=False, save_snapshot=False)
                model_cloned.fit(X_numpy, y_numpy, **stop_params)
            else:
                model_cloned.fit(X_numpy, y_numpy, **stop_params)

            # Encapsulation in Pipeline
            full_pipe = Pipeline(
                [
                    (
                        "preprocessor",
                        InplacePredictPreprocessor(
                            final_medians,
                            final_means,
                            final_stds,
                            list(self.feature_names_in_)
                            if self.feature_names_in_ is not None
                            else None,
                        ),
                    ),
                    ("model", model_cloned),
                ]
            )
            self.estimators_.append(full_pipe)

            MemoryManager.free_lgbm_booster(model_cloned)
            del model_cloned
            MemoryManager.trim_memory()

        logger.info('✅ Stacking completed')
        self.is_fitted_ = True

        # Final Cleaning
        if val_path is not None:
            del X_val, y_val
        del X_numpy, y_numpy
        MemoryManager.trim_memory()

        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        """Average importance of features on all base models."""
        importances_list = []

        for pipe in self.estimators_:
            if hasattr(pipe, "named_steps") and "model" in pipe.named_steps:
                model_step = pipe.named_steps["model"]
            else:
                model_step = pipe

            if hasattr(model_step, "feature_importances_"):
                imp = model_step.feature_importances_
                total_imp = imp.sum()
                if total_imp > 0:
                    importances_list.append(imp / total_imp)
                else:
                    importances_list.append(imp)

        if not importances_list:
            raise AttributeError("No model has a 'feature_importances_' attribute")

        return mean(array(importances_list), axis=0)

    def predict(self, X: Any) -> np.ndarray:
        """Generates predictions via trained stacking."""
        if not self.is_fitted_:
            raise RuntimeError('Untrained estimator.')

        X_local = X.copy() if hasattr(X, "copy") else X
        del X
        gc.collect()

        if hasattr(X_local, "columns"):
            cols_to_keep = [c for c in X_local.columns if c != "time_step"]
            X_local = np.ascontiguousarray(X_local[cols_to_keep].to_numpy(dtype=np.float32))

        X_local = self._apply_time_series_imputation(X_local)

        if hasattr(X_local, "drop") and "time_step" in X_local.columns:
            X_local = X_local.drop(columns=["time_step"])

        meta_features = []
        for pipe in self.estimators_:
            pred = np.asarray(pipe.predict(X_local).flatten(), dtype=np.float32)
            meta_features.append(pred)

        X_meta = np.asarray(np.column_stack(meta_features), dtype=np.float32)

        if self.meta_scaler_means_ is not None and self.meta_scaler_stds_ is not None:
            X_meta -= self.meta_scaler_means_
            safe_stds = np.where(self.meta_scaler_stds_ == 0, 1.0, self.meta_scaler_stds_)
            X_meta /= safe_stds

        X_meta = np.nan_to_num(X_meta, nan=0.0, posinf=0.0, neginf=0.0)

        return self.final_estimator_.predict(X_meta)


# ============================================================================
# MODEL MANAGER - Time Block Orchestration
