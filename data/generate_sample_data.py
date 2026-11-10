"""Gera uma base fictícia e reprodutível para a demonstração do projeto."""

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_PATH = Path(__file__).with_name("imoveis_exemplo.csv")
RNG = np.random.default_rng(42)

NEIGHBORHOODS = {
    "São Paulo": {
        "Pinheiros": (-23.5614, -46.6902, 13_500),
        "Vila Mariana": (-23.5898, -46.6346, 11_200),
        "Mooca": (-23.5582, -46.5965, 8_200),
        "Santana": (-23.4977, -46.6253, 7_900),
    },
    "Campinas": {
        "Cambuí": (-22.8977, -47.0536, 9_200),
        "Taquaral": (-22.8774, -47.0490, 7_800),
        "Barão Geraldo": (-22.8203, -47.0691, 6_600),
    },
    "Santos": {
        "Gonzaga": (-23.9675, -46.3322, 10_300),
        "Ponta da Praia": (-23.9890, -46.3077, 11_100),
        "Boqueirão": (-23.9660, -46.3210, 9_800),
    },
}


def build_dataset(rows: int = 420) -> pd.DataFrame:
    records = []
    city_names = list(NEIGHBORHOODS)
    for _ in range(rows):
        city = str(RNG.choice(city_names, p=[0.55, 0.25, 0.20]))
        neighborhood = str(RNG.choice(list(NEIGHBORHOODS[city])))
        latitude, longitude, base_m2 = NEIGHBORHOODS[city][neighborhood]
        area = int(np.clip(RNG.normal(88, 38), 28, 260))
        bedrooms = int(np.clip(round(area / 37 + RNG.normal(0, 0.55)), 1, 5))
        bathrooms = int(np.clip(round(bedrooms * 0.72 + RNG.normal(0.4, 0.45)), 1, 5))
        parking = int(np.clip(round(area / 75 + RNG.normal(0, 0.5)), 0, 4))
        year = int(RNG.integers(1968, 2025))
        metro_distance = float(np.clip(RNG.gamma(1.8, 0.65), 0.08, 5.5))
        age_discount = max(0, 2025 - year) * 620
        accessibility_bonus = max(0, 2.5 - metro_distance) * 15_000
        price = (
            area * base_m2
            + bathrooms * 18_000
            + parking * 32_000
            + accessibility_bonus
            - age_discount
            + RNG.normal(0, 65_000)
        )
        records.append(
            {
                "cidade": city,
                "bairro": neighborhood,
                "area_m2": area,
                "quartos": bedrooms,
                "banheiros": bathrooms,
                "vagas": parking,
                "ano_construcao": year,
                "distancia_metro_km": round(metro_distance, 2),
                "latitude": round(latitude + RNG.normal(0, 0.008), 6),
                "longitude": round(longitude + RNG.normal(0, 0.008), 6),
                "preco": int(max(price, 120_000)),
            }
        )
    return pd.DataFrame(records)


if __name__ == "__main__":
    build_dataset().to_csv(OUTPUT_PATH, index=False)
    print(f"Base criada em {OUTPUT_PATH}.")

