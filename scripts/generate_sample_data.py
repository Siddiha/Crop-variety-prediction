"""Generate sample CSVs under data/ so notebooks/Datathon.ipynb can run end-to-end."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


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

    n = 500
    plant_ids = rng.choice([2, 3, 4], size=n)
    name_map = {2: "Lettuce", 3: "Tomato", 4: "Basil"}
    records = []
    for i in range(n):
        pid = int(plant_ids[i])
        records.append(
            {
                "PlantID": pid,
                "PlantName": name_map[pid],
                "SoilTextureID": int(rng.choice([1, 2, 3])),
                "PHRangeID": int(rng.choice([1, 2, 3, 4])),
                "OrganicMatterID": int(rng.choice([1, 2, 3])),
                "SalinityLevelID": int(rng.choice([1, 2])),
                "ZoneID": float(rng.choice([8.0, 9.0, 10.0])),
                "HumidityID": int(rng.choice([1, 2, 3])),
                "PlantTypeID": int(rng.choice([1, 2, 3])),
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
