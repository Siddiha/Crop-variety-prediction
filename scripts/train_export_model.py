"""Train Random Forest (same settings as the notebook) and save joblib artifacts under models/."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from crop_pipeline import clean_dataframe, load_joined_tables, resolve_data_dir

DROP_COLS = [
    "PlantVarietyName",
    "PlantVarietyName_enc",
    "PlantID",
    "SoilTextureID",
    "PHRangeID",
    "OrganicMatterID",
    "SalinityLevelID",
    "ZoneID",
    "HumidityID",
    "PlantTypeID",
    "PlantName",
    "PHRange",
]

MODEL_DIR = ROOT / "models"
ARTIFACT_RF = MODEL_DIR / "rf_variety.joblib"
ARTIFACT_ENC = MODEL_DIR / "variety_encoder.joblib"
ARTIFACT_COLS = MODEL_DIR / "feature_columns.json"
ARTIFACT_META = MODEL_DIR / "training_meta.json"


def train_and_save(data_dir: Path | None, out_dir: Path | None) -> None:
    data_dir = data_dir or resolve_data_dir()
    out_dir = out_dir or MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_joined_tables(data_dir)
    cleaned_df = clean_dataframe(raw)

    variety_encoder = LabelEncoder()
    cleaned_df = cleaned_df.copy()
    cleaned_df["PlantVarietyName_enc"] = variety_encoder.fit_transform(
        cleaned_df["PlantVarietyName"]
    )

    X = cleaned_df.drop(columns=DROP_COLS)
    y = cleaned_df["PlantVarietyName_enc"]

    X_encoded = pd.get_dummies(X, drop_first=True)
    X_encoded.columns = X_encoded.columns.str.replace("[^\\w\\s]", "", regex=True)
    feature_columns = list(X_encoded.columns)

    strat = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=strat
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 15],
        "min_samples_leaf": [1, 2],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    joblib.dump(model, out_dir / ARTIFACT_RF.name)
    joblib.dump(variety_encoder, out_dir / ARTIFACT_ENC.name)
    (out_dir / ARTIFACT_COLS.name).write_text(
        json.dumps(feature_columns, indent=2), encoding="utf-8"
    )
    meta = {
        "accuracy_holdout": float(acc),
        "best_params": grid.best_params_,
        "n_rows": int(len(cleaned_df)),
        "n_classes": int(len(variety_encoder.classes_)),
        "data_dir": str(data_dir.resolve()),
    }
    (out_dir / ARTIFACT_META.name).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Best params:", grid.best_params_)
    print(f"Holdout accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Saved: {out_dir / ARTIFACT_RF.name}")
    print(f"Saved: {out_dir / ARTIFACT_ENC.name}")
    print(f"Saved: {out_dir / ARTIFACT_COLS.name}")
    print(f"Saved: {out_dir / ARTIFACT_META.name}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train and export crop variety model.")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Folder with CSVs (default: auto-detect from cwd)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: models/)",
    )
    args = p.parse_args()
    train_and_save(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
