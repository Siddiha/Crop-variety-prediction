"""Generate sample CSVs under data/ so notebooks/Datathon.ipynb can run end-to-end.

Each plant row includes a target variety plus feature values sampled around variety-specific
centroids so a classifier can learn variety labels (unlike a cartesian merge of all
varieties per PlantID, which duplicates identical features under different labels).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Variety -> typical growing attributes (IDs match lookup tables below)
VARIETY_PROFILES: dict[str, dict[str, float | int]] = {
    "Little Gem": {
        "PlantID": 2,
        "PlantTypeID": 1,
        "SoilTextureID": 2,
        "PHRangeID": 4,
        "OrganicMatterID": 2,
        "SalinityLevelID": 1,
        "ZoneID": 10.0,
        "HumidityID": 2,
    },
    "Looseleaf": {
        "PlantID": 2,
        "PlantTypeID": 1,
        "SoilTextureID": 2,
        "PHRangeID": 3,
        "OrganicMatterID": 2,
        "SalinityLevelID": 1,
        "ZoneID": 9.0,
        "HumidityID": 3,
    },
    "Oak Leaf": {
        "PlantID": 2,
        "PlantTypeID": 1,
        "SoilTextureID": 3,
        "PHRangeID": 3,
        "OrganicMatterID": 3,
        "SalinityLevelID": 1,
        "ZoneID": 10.0,
        "HumidityID": 2,
    },
    "Crisphead": {
        "PlantID": 2,
        "PlantTypeID": 1,
        "SoilTextureID": 1,
        "PHRangeID": 4,
        "OrganicMatterID": 2,
        "SalinityLevelID": 2,
        "ZoneID": 8.0,
        "HumidityID": 1,
    },
    "Red Leaf Lettuce": {
        "PlantID": 2,
        "PlantTypeID": 1,
        "SoilTextureID": 2,
        "PHRangeID": 2,
        "OrganicMatterID": 1,
        "SalinityLevelID": 1,
        "ZoneID": 9.0,
        "HumidityID": 2,
    },
    "Cherry": {
        "PlantID": 3,
        "PlantTypeID": 2,
        "SoilTextureID": 2,
        "PHRangeID": 4,
        "OrganicMatterID": 3,
        "SalinityLevelID": 1,
        "ZoneID": 10.0,
        "HumidityID": 3,
    },
    "Roma": {
        "PlantID": 3,
        "PlantTypeID": 2,
        "SoilTextureID": 3,
        "PHRangeID": 3,
        "OrganicMatterID": 2,
        "SalinityLevelID": 1,
        "ZoneID": 9.0,
        "HumidityID": 2,
    },
    "Beefsteak": {
        "PlantID": 3,
        "PlantTypeID": 2,
        "SoilTextureID": 2,
        "PHRangeID": 4,
        "OrganicMatterID": 2,
        "SalinityLevelID": 2,
        "ZoneID": 10.0,
        "HumidityID": 2,
    },
    "Grape": {
        "PlantID": 3,
        "PlantTypeID": 2,
        "SoilTextureID": 1,
        "PHRangeID": 2,
        "OrganicMatterID": 2,
        "SalinityLevelID": 1,
        "ZoneID": 9.0,
        "HumidityID": 3,
    },
    "Basil Genovese": {
        "PlantID": 4,
        "PlantTypeID": 3,
        "SoilTextureID": 2,
        "PHRangeID": 3,
        "OrganicMatterID": 2,
        "SalinityLevelID": 1,
        "ZoneID": 10.0,
        "HumidityID": 2,
    },
    "Thai": {
        "PlantID": 4,
        "PlantTypeID": 3,
        "SoilTextureID": 3,
        "PHRangeID": 2,
        "OrganicMatterID": 3,
        "SalinityLevelID": 1,
        "ZoneID": 9.0,
        "HumidityID": 3,
    },
    "Sweet": {
        "PlantID": 4,
        "PlantTypeID": 3,
        "SoilTextureID": 2,
        "PHRangeID": 4,
        "OrganicMatterID": 2,
        "SalinityLevelID": 1,
        "ZoneID": 8.0,
        "HumidityID": 2,
    },
}

NAME_MAP = {2: "Lettuce", 3: "Tomato", 4: "Basil"}


def _jitter_zone(val: float, rng: np.random.Generator) -> float:
    return float(np.clip(val + rng.normal(0, 0.35), 8.0, 10.0))


def _jitter_id(val: int, rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(np.clip(val + rng.integers(-1, 2), lo, hi))


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    humidity = pd.DataFrame(
        {
            "HumidityID": [1, 2, 3],
            "Classification": ["Low", "Moderate", "High"],
        }
    )
    ph = pd.DataFrame(
        {
            "PHRangeID": [1, 2, 3, 4],
            "PHRange": ["5.5 - 6.0", "6.1 - 6.5", "6.6 - 7.0", "7.1 - 7.5"],
            "SoilType": ["Sandy", "Loamy", "Clay", "Loamy"],
        }
    )
    soil = pd.DataFrame(
        {
            "SoilTextureID": [1, 2, 3],
            "SoilTexture": ["Sandy Soil", "Loamy Soil", "Clay Soil"],
        }
    )
    plant_types = pd.DataFrame(
        {
            "PlantTypeID": [1, 2, 3],
            "PlantType": ["Vegetable", "Fruit", "Herb"],
        }
    )
    org = pd.DataFrame(
        {
            "OrganicMatterID": [1, 2, 3],
            "OrganicMatterContent": [
                "Low (1% - 2%)",
                "Moderate (2% - 4%)",
                "High (4% - 6%)",
            ],
        }
    )
    saline = pd.DataFrame(
        {
            "SalinityLevelID": [1, 2],
            "SalinityLevel": ["< 2", ">= 2"],
            "Classification": ["Non-Saline", "Saline"],
        }
    )
    hard = pd.DataFrame(
        {
            "Zone": ["10a", "10b", "9a", "8b"],
            "TemperatureStartRange": [30.0, 32.0, 20.0, 15.0],
            "TemperatureEndRange": [40.0, 39.0, 28.0, 22.0],
        }
    )

    varieties: list[dict[str, object]] = []
    for pid, names in [
        (2, ["Little Gem", "Looseleaf", "Oak Leaf", "Crisphead", "Red Leaf Lettuce"]),
        (3, ["Cherry", "Roma", "Beefsteak", "Grape"]),
        (4, ["Basil Genovese", "Thai", "Sweet"]),
    ]:
        for n in names:
            varieties.append({"PlantID": pid, "PlantVarietyName": n})
    var_df = pd.DataFrame(varieties)

    n = 2000
    variety_names = list(VARIETY_PROFILES.keys())
    records = []
    for i in range(n):
        vname = str(rng.choice(variety_names))
        prof = VARIETY_PROFILES[vname]
        pid = int(prof["PlantID"])
        records.append(
            {
                "PlantID": pid,
                "PlantName": NAME_MAP[pid],
                "PlantVarietyName": vname,
                "SoilTextureID": _jitter_id(int(prof["SoilTextureID"]), rng, 1, 3),
                "PHRangeID": _jitter_id(int(prof["PHRangeID"]), rng, 1, 4),
                "OrganicMatterID": _jitter_id(int(prof["OrganicMatterID"]), rng, 1, 3),
                "SalinityLevelID": _jitter_id(int(prof["SalinityLevelID"]), rng, 1, 2),
                "ZoneID": _jitter_zone(float(prof["ZoneID"]), rng),
                "HumidityID": _jitter_id(int(prof["HumidityID"]), rng, 1, 3),
                "PlantTypeID": int(prof["PlantTypeID"]),
                "PlantDescription": f"Sample row {i}",
            }
        )
    plant = pd.DataFrame(records)

    humidity.to_csv(DATA / "HumidityLookup.csv", index=False)
    ph.to_csv(DATA / "PHRangeLookup.csv", index=False)
    soil.to_csv(DATA / "SoilTextureLookup.csv", index=False)
    plant_types.to_csv(DATA / "PlantTypeLookup.csv", index=False)
    org.to_csv(DATA / "OrganicMatterLookup.csv", index=False)
    saline.to_csv(DATA / "SalinityLookup.csv", index=False)
    hard.to_csv(DATA / "PlantHardinessZoneLookup.csv", index=False)
    var_df.to_csv(DATA / "PlantVariety.csv", index=False)
    plant.to_csv(DATA / "Plant.csv", index=False)
    print(f"Wrote sample data to {DATA}")


if __name__ == "__main__":
    main()
