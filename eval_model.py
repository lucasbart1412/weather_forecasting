"""Historical facade of the evaluation module."""

from model_evaluation.cli import main
from model_evaluation.evaluator_impl import ModelEvaluator

__all__ = ["ModelEvaluator", "main"]


if __name__ == "__main__":
    main()
