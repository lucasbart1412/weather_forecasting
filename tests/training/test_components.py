import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import pandas as pd

from training.calibration import split_calibration_data
from training.memory import MemoryManager
from training.model_manager import ModelManager
from training.preprocessing_data import optimize_memory
from training.model_factory import ModelFactoryMixin
from training.metrics import (
    has_multiple_classes,
    regression_metrics,
    select_f1_threshold,
)
from training.progress import TrainingEtaEstimator
from training.stacking_regressor import TimeSeriesStackingTrainer


class ImportanceEstimator(BaseEstimator):
    def __init__(self, importances):
        self.importances = importances

    @property
    def feature_importances_(self):
        return np.asarray(self.importances)


def test_training_components_are_importable():
    assert callable(MemoryManager.trim_memory)
    assert callable(ModelManager)
    assert callable(optimize_memory)
    assert callable(TimeSeriesStackingTrainer)


def test_feature_importances_averages_normalized_base_model_importances():
    trainer = TimeSeriesStackingTrainer([], None, None)
    trainer.estimators_ = [
        Pipeline(
            [
                ("preprocessor", ImportanceEstimator([1, 1, 1])),
                ("model", ImportanceEstimator([2, 0, 2])),
            ]
        ),
        ImportanceEstimator([0, 3, 1]),
        ImportanceEstimator([0, 0, 0]),
    ]

    assert np.allclose(trainer.feature_importances_, [0.25, 0.375, 0.375])


def test_medium_precipitation_models_use_regularized_tweedie_losses():
    trainer = ModelFactoryMixin().get_advanced_models("medium", "precip")

    assert trainer.estimators[0][1].get_params()["loss_function"] == (
        "Tweedie:variance_power=1.5"
    )
    assert trainer.estimators[0][1].get_params()["l2_leaf_reg"] == 10.0
    assert trainer.estimators[1][1].get_params()["objective"] == "tweedie"
    assert trainer.estimators[1][1].get_params()["tweedie_variance_power"] == 1.5
    assert trainer.estimators[1][1].get_params()["reg_alpha"] == 2.0
    assert trainer.estimators[1][1].get_params()["reg_lambda"] == 5.0
    assert trainer.estimators[2][1].get_params()["objective"] == "reg:tweedie"
    assert trainer.estimators[2][1].get_params()["reg_alpha"] == 2.0
    assert trainer.estimators[2][1].get_params()["reg_lambda"] == 5.0


def test_regression_metrics_report_real_pearson_correlation():
    metrics = regression_metrics([1, 2, 3], [1, 2, 4])

    assert metrics["mae"] == pytest.approx(1 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert 0.0 < metrics["pearson_r"] < 1.0


def test_precipitation_threshold_is_selected_from_calibration_labels():
    threshold, score = select_f1_threshold(
        [0, 0, 1, 1],
        [0.05, 0.3, 0.4, 0.9],
        [0.2, 0.35, 0.5],
    )

    assert threshold == 0.35
    assert score == 1.0


def test_binary_class_check_accepts_numpy_labels():
    assert has_multiple_classes(np.array([0, 1, 0]))
    assert not has_multiple_classes(np.array([1, 1, 1]))


def test_eta_uses_observed_median_duration_per_block():
    estimator = TrainingEtaEstimator(
        [("short", "temp"), ("short", "wind"), ("medium", "temp")]
    )

    assert estimator.estimate_remaining() == (None, 0.0)
    estimator.record("short", 60.0)
    estimated_seconds, coverage = estimator.estimate_remaining()

    assert estimated_seconds == 120.0
    assert coverage == 0.5
    estimator.record("short", 100.0)
    estimated_seconds, coverage = estimator.estimate_remaining()
    assert estimated_seconds == 80.0
    assert coverage == 0.0


def test_calibration_split_separates_fit_threshold_and_evaluation_periods():
    features = pd.DataFrame({"time_step": np.arange(1000)})
    targets = pd.Series(np.arange(1000))

    fit_X, fit_y, threshold_X, threshold_y, eval_X, eval_y = split_calibration_data(
        features, targets, gap_steps=100
    )

    assert fit_X["time_step"].max() < threshold_X["time_step"].min() - 100
    assert threshold_X["time_step"].max() < eval_X["time_step"].min() - 100
    assert fit_y.index.equals(fit_X.index)
    assert threshold_y.index.equals(threshold_X.index)
    assert eval_y.index.equals(eval_X.index)
