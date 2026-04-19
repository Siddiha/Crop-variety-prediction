"""Tests for scripts/crop_pipeline.py — load, join, and clean logic."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from crop_pipeline import resolve_data_dir


# ---------------------------------------------------------------------------
# resolve_data_dir
# ---------------------------------------------------------------------------

def test_resolve_data_dir_from_root():
    result = resolve_data_dir(ROOT)
    assert result.name == "data"
    assert (result / "HumidityLookup.csv").exists()


def test_resolve_data_dir_from_notebooks():
    result = resolve_data_dir(ROOT / "notebooks")
    assert result.name == "data"
    assert result.exists()


# ---------------------------------------------------------------------------
# load_joined_tables
# ---------------------------------------------------------------------------

def test_load_returns_dataframe(joined_df):
    assert isinstance(joined_df, pd.DataFrame)


def test_load_not_empty(joined_df):
    assert len(joined_df) > 0


def test_load_has_variety_name(joined_df):
    assert "PlantVarietyName" in joined_df.columns


def test_load_has_soil_texture(joined_df):
    assert "SoilTexture" in joined_df.columns


def test_load_has_temperature_ranges(joined_df):
    assert "TemperatureStartRange" in joined_df.columns
    assert "TemperatureEndRange" in joined_df.columns


def test_load_has_plant_type(joined_df):
    assert "PlantType" in joined_df.columns


def test_load_drops_plant_description(joined_df):
    assert "PlantDescription" not in joined_df.columns


def test_load_has_organic_matter(joined_df):
    assert "OrganicMatterContent" in joined_df.columns


def test_load_has_salinity(joined_df):
    assert "SalinityLevel" in joined_df.columns


# ---------------------------------------------------------------------------
# clean_dataframe
# ---------------------------------------------------------------------------

def test_clean_no_nulls(cleaned_df):
    assert cleaned_df.isnull().sum().sum() == 0


def test_clean_is_subset_of_joined(joined_df, cleaned_df):
    assert len(cleaned_df) <= len(joined_df)


def test_clean_organic_matter_labels_normalized(cleaned_df):
    raw_labels = {"Moderate (2% - 4%)", "Low (1% - 2%)", "High (4% - 6%)"}
    remaining = set(cleaned_df["OrganicMatterContent"].unique())
    assert not raw_labels.intersection(remaining)


def test_clean_organic_matter_values_valid(cleaned_df):
    valid = {"Low", "Moderate", "High"}
    actual = set(cleaned_df["OrganicMatterContent"].unique())
    assert actual.issubset(valid)


def test_clean_variety_name_present(cleaned_df):
    assert "PlantVarietyName" in cleaned_df.columns
    assert cleaned_df["PlantVarietyName"].notna().all()
