"""Treinamento e avaliação dos modelos de regressão."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .data_loader import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN


MODEL_NAMES = [
    "Regressão Linear",
    "Árvore de Decisão",
    "Random Forest",
    "KNN",
]

DEFAULT_PARAMETERS: dict[str, dict[str, Any]] = {
    "Regressão Linear": {},
    "Árvore de Decisão": {"max_depth": 8, "min_samples_leaf": 2},
    "Random Forest": {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 1},
    "KNN": {"n_neighbors": 7, "weights": "distance"},
}


@dataclass
class ModelResult:
    """Artefatos e métricas de uma execução de treinamento."""

    model_name: str
    pipeline: Pipeline
    metrics: dict[str, float]
    predictions: pd.DataFrame
    train_rows: int
    test_rows: int


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # Compatibilidade com scikit-learn anterior a 1.2.
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(model_name: str, parameters: dict[str, Any] | None = None) -> Pipeline:
    """Monta pré-processamento e estimador sem vazamento entre treino e teste."""
    parameters = parameters or {}
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _one_hot_encoder()),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    if model_name == "Regressão Linear":
        estimator = LinearRegression()
    elif model_name == "Árvore de Decisão":
        estimator = DecisionTreeRegressor(random_state=42, **parameters)
    elif model_name == "Random Forest":
        estimator = RandomForestRegressor(random_state=42, n_jobs=-1, **parameters)
    elif model_name == "KNN":
        estimator = KNeighborsRegressor(**parameters)
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")

    return Pipeline(steps=[("preprocessing", preprocessing), ("model", estimator)])


def train_and_evaluate(
    dataframe: pd.DataFrame,
    model_name: str,
    parameters: dict[str, Any] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelResult:
    """Treina um modelo e calcula métricas em um conjunto de teste separado."""
    x = dataframe[FEATURE_COLUMNS]
    y = dataframe[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    pipeline = build_pipeline(model_name, parameters)
    pipeline.fit(x_train, y_train)
    predicted = pipeline.predict(x_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predicted))),
        "R²": float(r2_score(y_test, predicted)),
    }
    prediction_frame = pd.DataFrame(
        {
            "valor_real": y_test.to_numpy(),
            "valor_previsto": predicted,
            "residuo": y_test.to_numpy() - predicted,
        }
    ).reset_index(drop=True)
    return ModelResult(
        model_name=model_name,
        pipeline=pipeline,
        metrics=metrics,
        predictions=prediction_frame,
        train_rows=len(x_train),
        test_rows=len(x_test),
    )


def compare_models(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Compara todos os modelos com parâmetros iniciais reproduzíveis."""
    rows = []
    for model_name in MODEL_NAMES:
        result = train_and_evaluate(dataframe, model_name, DEFAULT_PARAMETERS[model_name])
        rows.append({"modelo": model_name, **result.metrics})
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
