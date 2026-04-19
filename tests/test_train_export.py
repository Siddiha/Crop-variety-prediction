"""Tests for scripts/train_export_model.py — training and artifact output."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from train_export_model import train_and_save


@pytest.fixture(scope="module")
def trained_model_dir(data_dir, tmp_path_factory):
    """Run train_and_save once per module and return the output dir."""
    out = tmp_path_factory.mktemp("models")
    train_and_save(data_dir, out)
    return out


# ---------------------------------------------------------------------------
# Artifact existence
# ---------------------------------------------------------------------------

def test_rf_joblib_created(trained_model_dir):
    assert (trained_model_dir / "rf_variety.joblib").exists()


def test_encoder_joblib_created(trained_model_dir):
    assert (trained_model_dir / "variety_encoder.joblib").exists()


def test_feature_columns_json_created(trained_model_dir):
    assert (trained_model_dir / "feature_columns.json").exists()


def test_training_meta_json_created(trained_model_dir):
    assert (trained_model_dir / "training_meta.json").exists()


# ---------------------------------------------------------------------------
# Artifact content
# ---------------------------------------------------------------------------

def test_rf_is_random_forest(trained_model_dir):
    model = joblib.load(trained_model_dir / "rf_variety.joblib")
    assert isinstance(model, RandomForestClassifier)


def test_encoder_is_label_encoder(trained_model_dir):
    enc = joblib.load(trained_model_dir / "variety_encoder.joblib")
    assert isinstance(enc, LabelEncoder)


def test_feature_columns_nonempty(trained_model_dir):
    cols = json.loads((trained_model_dir / "feature_columns.json").read_text())
    assert isinstance(cols, list)
    assert len(cols) > 0


def test_meta_accuracy_in_range(trained_model_dir):
    meta = json.loads((trained_model_dir / "training_meta.json").read_text())
    assert "accuracy_holdout" in meta
    assert 0.0 <= meta["accuracy_holdout"] <= 1.0


def test_meta_has_best_params(trained_model_dir):
    meta = json.loads((trained_model_dir / "training_meta.json").read_text())
    assert "best_params" in meta
    params = meta["best_params"]
    assert "n_estimators" in params
    assert "max_depth" in params
    assert "min_samples_leaf" in params


def test_meta_row_count_positive(trained_model_dir):
    meta = json.loads((trained_model_dir / "training_meta.json").read_text())
    assert meta["n_rows"] > 0


def test_meta_class_count_positive(trained_model_dir):
    meta = json.loads((trained_model_dir / "training_meta.json").read_text())
    assert meta["n_classes"] > 1


def test_encoder_classes_match_meta(trained_model_dir):
    enc = joblib.load(trained_model_dir / "variety_encoder.joblib")
    meta = json.loads((trained_model_dir / "training_meta.json").read_text())
    assert len(enc.classes_) == meta["n_classes"]
