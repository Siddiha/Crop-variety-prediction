"""Shared pytest fixtures for the crop-variety-prediction test suite."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pytest

from crop_pipeline import clean_dataframe, load_joined_tables


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return ROOT / "data"


@pytest.fixture(scope="session")
def joined_df(data_dir):
    return load_joined_tables(data_dir)


@pytest.fixture(scope="session")
def cleaned_df(joined_df):
    return clean_dataframe(joined_df)
