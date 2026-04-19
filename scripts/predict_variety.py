"""Load saved model and predict variety for a row from Plant.csv (demo)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

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


def main() -> None:
    p = argparse.ArgumentParser(description="Predict crop variety from a data row.")
    p.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models",
        help="Directory with rf_variety.joblib, variety_encoder.joblib, feature_columns.json",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="CSV folder (default: auto-detect)",
    )
    p.add_argument(
        "--row",
        type=int,
        default=0,
        help="Row index in the cleaned joined table (after same pipeline as training)",
    )
    args = p.parse_args()

    mdir = args.model_dir
    rf_path = mdir / "rf_variety.joblib"
    enc_path = mdir / "variety_encoder.joblib"
    cols_path = mdir / "feature_columns.json"
    for f in (rf_path, enc_path, cols_path):
        if not f.is_file():
            print(f"Missing {f}. Run: python scripts/train_export_model.py", file=sys.stderr)
            sys.exit(1)

    data_dir = args.data_dir or resolve_data_dir()
    raw = load_joined_tables(data_dir)
    cleaned = clean_dataframe(raw)

    variety_encoder = joblib.load(enc_path)
    model = joblib.load(rf_path)
    feature_columns: list[str] = json.loads(cols_path.read_text(encoding="utf-8"))

    cleaned = cleaned.copy()
    cleaned["PlantVarietyName_enc"] = variety_encoder.transform(cleaned["PlantVarietyName"])

    if args.row < 0 or args.row >= len(cleaned):
        print(f"Row {args.row} out of range (0..{len(cleaned) - 1})", file=sys.stderr)
        sys.exit(1)

    row_df = cleaned.iloc[[args.row]]
    true_name = row_df["PlantVarietyName"].iloc[0]

    X = row_df.drop(columns=DROP_COLS)
    Xe = pd.get_dummies(X, drop_first=True)
    Xe.columns = Xe.columns.str.replace("[^\\w\\s]", "", regex=True)
    Xe = Xe.reindex(columns=feature_columns, fill_value=0)

    pred_enc = model.predict(Xe)[0]
    pred_name = variety_encoder.inverse_transform([pred_enc])[0]

    print(f"Row index:     {args.row}")
    print(f"Actual variety: {true_name}")
    print(f"Predicted:      {pred_name}")


if __name__ == "__main__":
    main()
