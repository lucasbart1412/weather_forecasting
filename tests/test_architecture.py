from pathlib import Path

ROOT = Path(__file__).parents[1]

MODULAR_PACKAGES = (
    "infrastructure",
    "data_fetcher",
    "feature_engineering",
    "ml",
    "training",
    "model_evaluation",
    "correlation_plotter",
    "presentation",
)

OVERSIZED_MODULES = {
    "infrastructure/config.py",
    "data_fetcher/data_fetcher.py",
    "feature_engineering/physics.py",
    "ml/batch_predictor.py",
    "presentation/dashboard_helpers.py",
    "training/training_loop.py",
    "training/stacking_impl.py",
}


def test_refactored_package_entrypoints_exist():
    expected = {
        "infrastructure/config.py",
        "data_fetcher/data_fetcher.py",
        "feature_engineering/feature_engineer_impl.py",
        "ml/models_impl.py",
        "ml/weather_predictor_impl.py",
        "training/pipeline.py",
        "model_evaluation/evaluator_impl.py",
        "correlation_plotter/analysis_impl.py",
        "presentation/weather_icons.py",
    }
    assert all((ROOT / path).exists() for path in expected)


def test_modular_packages_have_package_initializers():
    assert all((ROOT / package / "__init__.py").exists() for package in MODULAR_PACKAGES)


def test_public_compatibility_facades_exist():
    expected = {
        "feature_engineering/feature_engineering.py",
        "ml/models_impl.py",
        "ml/weather_predictor_impl.py",
        "training/model_manager_impl.py",
    }
    assert all((ROOT / path).exists() for path in expected)


def test_regular_modular_modules_respect_line_limit():
    oversized = {}
    for package in MODULAR_PACKAGES:
        for path in (ROOT / package).glob("*.py"):
            relative_path = path.relative_to(ROOT).as_posix()
            line_count = sum(1 for _ in path.open(encoding="utf-8"))
            if relative_path not in OVERSIZED_MODULES and line_count > 300:
                oversized[relative_path] = line_count
    assert oversized == {}


def test_oversized_modules_are_explicitly_documented():
    architecture = (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert all(f"`{path}`" in architecture for path in OVERSIZED_MODULES)
