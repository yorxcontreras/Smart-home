"""Leitura, normalização e validação das bases de imóveis."""

from __future__ import annotations

import re
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd


TARGET_COLUMN = "preco"
CATEGORICAL_FEATURES = ["cidade", "bairro"]
NUMERIC_FEATURES = [
    "area_m2",
    "quartos",
    "banheiros",
    "vagas",
    "ano_construcao",
    "distancia_metro_km",
]
GEO_COLUMNS = ["latitude", "longitude"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
REQUIRED_COLUMNS = FEATURE_COLUMNS + GEO_COLUMNS + [TARGET_COLUMN]

COLUMN_DESCRIPTIONS = {
    "cidade": "Cidade do imóvel",
    "bairro": "Bairro do imóvel",
    "area_m2": "Área privativa em metros quadrados",
    "quartos": "Quantidade de quartos",
    "banheiros": "Quantidade de banheiros",
    "vagas": "Quantidade de vagas",
    "ano_construcao": "Ano em que o imóvel foi construído",
    "distancia_metro_km": "Distância até uma estação de metrô, em quilômetros",
    "latitude": "Latitude do imóvel",
    "longitude": "Longitude do imóvel",
    "preco": "Preço anunciado em reais",
}

ALIASES = {
    "municipio": "cidade",
    "district": "bairro",
    "neighborhood": "bairro",
    "area": "area_m2",
    "area_m": "area_m2",
    "metragem": "area_m2",
    "bedrooms": "quartos",
    "rooms": "quartos",
    "bathrooms": "banheiros",
    "parking": "vagas",
    "garage": "vagas",
    "ano": "ano_construcao",
    "year_built": "ano_construcao",
    "distancia_metro": "distancia_metro_km",
    "metro_km": "distancia_metro_km",
    "lat": "latitude",
    "lon": "longitude",
    "lng": "longitude",
    "price": "preco",
    "valor": "preco",
}


class DataValidationError(ValueError):
    """Erro amigável de formato ou qualidade da base."""


def _normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return ALIASES.get(text, text)


def normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Converte nomes comuns de colunas para o esquema canônico do projeto."""
    renamed = dataframe.rename(columns={column: _normalize_name(column) for column in dataframe.columns})
    if renamed.columns.duplicated().any():
        duplicates = sorted(set(renamed.columns[renamed.columns.duplicated()].tolist()))
        raise DataValidationError(
            "Algumas colunas ficaram duplicadas após a normalização: " + ", ".join(duplicates)
        )
    return renamed


def _read_bytes(payload: bytes, suffix: str) -> pd.DataFrame:
    buffer = BytesIO(payload)
    suffix = suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(buffer, sep=None, engine="python")
        except UnicodeDecodeError:
            buffer.seek(0)
            return pd.read_csv(buffer, sep=None, engine="python", encoding="latin-1")
    if suffix == ".json":
        try:
            return pd.read_json(buffer)
        except ValueError:
            buffer.seek(0)
            return pd.read_json(buffer, lines=True)
    if suffix == ".xml":
        return pd.read_xml(buffer)
    raise DataValidationError("Formato não suportado. Envie um arquivo CSV, JSON ou XML.")


def clean_and_validate(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Limpa a base e devolve um relatório simples de qualidade."""
    if dataframe.empty:
        raise DataValidationError("O arquivo enviado não contém registros.")

    cleaned = normalize_columns(dataframe.copy())
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in cleaned.columns]
    if missing_columns:
        raise DataValidationError(
            "Colunas obrigatórias ausentes: " + ", ".join(missing_columns)
        )

    original_rows = len(cleaned)
    duplicate_rows = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().copy()

    for column in NUMERIC_FEATURES + GEO_COLUMNS + [TARGET_COLUMN]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        cleaned[column] = cleaned[column].astype("string").str.strip()
        cleaned.loc[cleaned[column].isin(["", "nan", "None"]), column] = pd.NA

    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    rows_with_missing = int(cleaned[REQUIRED_COLUMNS].isna().any(axis=1).sum())
    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS).copy()

    valid_values = (
        (cleaned["area_m2"] > 0)
        & (cleaned["preco"] > 0)
        & (cleaned["quartos"] >= 0)
        & (cleaned["banheiros"] >= 0)
        & (cleaned["vagas"] >= 0)
        & (cleaned["distancia_metro_km"] >= 0)
        & cleaned["latitude"].between(-90, 90)
        & cleaned["longitude"].between(-180, 180)
    )
    invalid_value_rows = int((~valid_values).sum())
    cleaned = cleaned.loc[valid_values].reset_index(drop=True)

    if len(cleaned) < 20:
        raise DataValidationError(
            "Após a limpeza restaram menos de 20 registros válidos. "
            "Use uma base maior para treinar e testar os modelos."
        )

    report = {
        "linhas_originais": original_rows,
        "duplicadas_removidas": duplicate_rows,
        "linhas_com_ausencias_removidas": rows_with_missing,
        "linhas_invalidas_removidas": invalid_value_rows,
        "linhas_validas": len(cleaned),
    }
    return cleaned, report


def load_dataset(source: str | Path | BinaryIO, filename: str | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    """Carrega uma base a partir de caminho ou arquivo enviado pelo Streamlit."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        payload = path.read_bytes()
        suffix = path.suffix
    else:
        payload = source.getvalue() if hasattr(source, "getvalue") else source.read()
        effective_name = filename or getattr(source, "name", "")
        suffix = Path(effective_name).suffix

    try:
        dataframe = _read_bytes(payload, suffix)
    except DataValidationError:
        raise
    except Exception as exc:
        raise DataValidationError(f"Não foi possível ler o arquivo: {exc}") from exc
    return clean_and_validate(dataframe)
