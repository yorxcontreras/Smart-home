from pathlib import Path

from src.data_loader import FEATURE_COLUMNS, load_dataset
from src.modeling import MODEL_NAMES, build_pipeline, compare_models, train_and_evaluate


DATASET = Path(__file__).parents[1] / "data" / "imoveis_exemplo.csv"


def test_all_models_build() -> None:
    for model_name in MODEL_NAMES:
        assert build_pipeline(model_name) is not None


def test_training_produces_metrics_and_predictions() -> None:
    data, _ = load_dataset(DATASET)
    result = train_and_evaluate(data, "Random Forest", {"n_estimators": 20, "max_depth": 8})
    assert set(result.metrics) == {"MAE", "RMSE", "R²"}
    assert result.test_rows == len(result.predictions)
    assert list(result.pipeline.feature_names_in_) == FEATURE_COLUMNS


def test_comparison_contains_every_model() -> None:
    data, _ = load_dataset(DATASET)
    comparison = compare_models(data)
    assert set(comparison["modelo"]) == set(MODEL_NAMES)
    assert comparison["RMSE"].is_monotonic_increasing

