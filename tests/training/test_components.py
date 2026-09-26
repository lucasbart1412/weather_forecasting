from training.memory import MemoryManager
from training.model_manager import ModelManager
from training.preprocessing_data import optimize_memory
from training.stacking_regressor import TimeSeriesStackingTrainer


def test_training_components_are_importable():
    assert callable(MemoryManager.trim_memory)
    assert callable(ModelManager)
    assert callable(optimize_memory)
    assert callable(TimeSeriesStackingTrainer)
