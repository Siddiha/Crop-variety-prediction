"""Shared data loading and cleaning for training and inference (matches notebooks/Datathon.ipynb)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def resolve_data_dir(cwd: Path | None = None) -> Path:
    cwd = cwd or Path.cwd().resolve()
    if (cwd / "data" / "HumidityLookup.csv").exists():
        return cwd / "data"
    if (cwd.parent / "data" / "HumidityLookup.csv").exists():
        return cwd.parent / "data"
    return cwd.parent / "data"


def load_joined_tables(data_dir: Path) -> pd.DataFrame:
    df_hum = pd.read_csv(data_dir / "HumidityLookup.csv")
    df_ph = pd.read_csv(data_dir / "PHRangeLookup.csv")
    df_soil = pd.read_csv(data_dir / "SoilTextureLookup.csv")
    df_var = pd.read_csv(data_dir / "PlantVariety.csv")
    df_plant_type = pd.read_csv(data_dir / "PlantTypeLookup.csv")
    df_hard = pd.read_csv(data_dir / "PlantHardinessZoneLookup.csv")
    df_saline = pd.read_csv(data_dir / "SalinityLookup.csv")
    df_org = pd.read_csv(data_dir / "OrganicMatterLookup.csv")
    df_plant = pd.read_csv(data_dir / "Plant.csv")

    df_hard = df_hard.copy()
    df_hard["UnifiedZone"] = df_hard["Zone"].astype(str).str.extract(r"(\d+)")
    merged_zone = (
        df_hard.dropna(subset=["UnifiedZone"])
        .groupby("UnifiedZone", as_index=False)
        .agg({"TemperatureStartRange": "min", "TemperatureEndRange": "max"})
    )
    merged_zone["UnifiedZone"] = merged_zone["UnifiedZone"].astype(float)
    df_hard_new = merged_zone

    joins = (
        df_plant.merge(df_ph[["PHRangeID", "PHRange", "SoilType"]], on="PHRangeID", how="left")
        .merge(
            df_hard_new[["UnifiedZone", "TemperatureStartRange", "TemperatureEndRange"]],
            left_on="ZoneID",
            right_on="UnifiedZone",
            how="left",
        )
        .merge(df_plant_type[["PlantTypeID", "PlantType"]], on="PlantTypeID", how="left")
        .merge(df_soil[["SoilTextureID", "SoilTexture"]], on="SoilTextureID", how="left")
        .merge(df_hum[["HumidityID", "Classification"]], on="HumidityID", how="left")
        .merge(
            df_org[["OrganicMatterID", "OrganicMatterContent"]],
            on="OrganicMatterID",
            how="left",
        )
        .merge(
            df_saline[["SalinityLevelID", "SalinityLevel", "Classification"]],
            on="SalinityLevelID",
            how="left",
            suffixes=("", "_Salinity"),
        )
    )
    if "PlantVarietyName" not in df_plant.columns:
        result = joins.merge(df_var[["PlantID", "PlantVarietyName"]], on="PlantID", how="left")
    else:
        result = joins

    return result.drop(columns=["PlantDescription"])


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.dropna()
    cleaned = cleaned.copy()
    cleaned["OrganicMatterContent"] = cleaned["OrganicMatterContent"].replace(
        {
            "Moderate (2% - 4%)": "Moderate",
            "Low (1% - 2%)": "Low",
            "High (4% - 6%)": "High",
        }
    )
    return cleaned
