from io import BytesIO

import pandas as pd
import pytest

from src.data_loader import DataValidationError, clean_and_validate, load_dataset


def valid_frame(rows: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cidade": ["São Paulo"] * rows,
            "bairro": ["Centro"] * rows,
            "area_m2": range(50, 50 + rows),
            "quartos": [2] * rows,
            "banheiros": [1] * rows,
            "vagas": [1] * rows,
            "ano_construcao": [2010] * rows,
            "distancia_metro_km": [1.0] * rows,
            "latitude": [-23.55] * rows,
            "longitude": [-46.63] * rows,
            "preco": range(300_000, 300_000 + rows),
        }
    )


def test_clean_and_validate_removes_duplicates() -> None:
    frame = valid_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    cleaned, report = clean_and_validate(frame)
    assert len(cleaned) == 30
    assert report["duplicadas_removidas"] == 1


def test_aliases_are_normalized() -> None:
    frame = valid_frame().rename(columns={"preco": "valor", "area_m2": "metragem"})
    cleaned, _ = clean_and_validate(frame)
    assert "preco" in cleaned.columns
    assert "area_m2" in cleaned.columns


def test_missing_column_raises_friendly_error() -> None:
    with pytest.raises(DataValidationError, match="Colunas obrigatórias ausentes"):
        clean_and_validate(valid_frame().drop(columns=["preco"]))


def test_csv_upload_is_supported() -> None:
    payload = valid_frame().to_csv(index=False).encode("utf-8")
    uploaded = BytesIO(payload)
    cleaned, report = load_dataset(uploaded, "imoveis.csv")
    assert len(cleaned) == 30
    assert report["linhas_validas"] == 30

