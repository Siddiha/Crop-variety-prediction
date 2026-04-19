"""Update join cell to skip PlantVariety merge when Plant.csv already has PlantVarietyName."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "Datathon.ipynb"

CELL = '''# Join the dataframes
joins = (
    df_plant
    .merge(df_Ph[["PHRangeID", "PHRange", "SoilType"]], on="PHRangeID", how="left")
    .merge(
        df_Hard_new[["UnifiedZone", "TemperatureStartRange", "TemperatureEndRange"]],
        left_on="ZoneID",
        right_on="UnifiedZone",
        how="left",
    )
    .merge(df_PlantType[["PlantTypeID", "PlantType"]], on="PlantTypeID", how="left")
    .merge(df_Soil[["SoilTextureID", "SoilTexture"]], on="SoilTextureID", how="left")
    .merge(df_Hum[["HumidityID", "Classification"]], on="HumidityID", how="left")
    .merge(
        df_OrgMat[["OrganicMatterID", "OrganicMatterContent"]],
        on="OrganicMatterID",
        how="left",
    )
    .merge(
        df_Saline[["SalinityLevelID", "SalinityLevel", "Classification"]],
        on="SalinityLevelID",
        how="left",
        suffixes=("", "_Salinity"),
    )
)
if "PlantVarietyName" not in df_plant.columns:
    result = joins.merge(df_Var[["PlantID", "PlantVarietyName"]], on="PlantID", how="left")
else:
    result = joins

result = result.drop(columns=["PlantDescription"])

print(result.head())
'''


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    nb["cells"][3]["source"] = [line + "\n" for line in CELL.splitlines()]
    for c in nb["cells"]:
        c["execution_count"] = None
        if c.get("cell_type") == "code":
            c["outputs"] = []
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated merge cell in {NB}")


if __name__ == "__main__":
    main()
