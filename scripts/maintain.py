"""Maintain sample data and notebooks/Datathon.ipynb without hand-editing JSON.

Examples:
  python scripts/maintain.py sample-data   # regenerate data/*.csv (demo dataset)
  python scripts/maintain.py merge-cell    # join cell when Plant.csv includes PlantVarietyName
  python scripts/maintain.py ml-cells      # variety target, GridSearchCV, confusion matrix
  python scripts/maintain.py all           # sample-data, then merge-cell, then ml-cells
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "Datathon.ipynb"

MERGE_CELL = """# Join the dataframes
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
"""

CELL_14 = """from sklearn.preprocessing import LabelEncoder

variety_encoder = LabelEncoder()
cleaned_df_copy4["PlantVarietyName_enc"] = variety_encoder.fit_transform(
    cleaned_df_copy4["PlantVarietyName"]
)

cleaned_df_copy4.head()
"""

CELL_15 = """variety_mapping = dict(
    zip(variety_encoder.classes_, range(len(variety_encoder.classes_)))
)
print(variety_mapping)
"""

CELL_17 = """from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# Predict crop variety from growing conditions; keep PlantType as a categorical feature
X = cleaned_df_copy4.drop(
    columns=[
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
)
y = cleaned_df_copy4["PlantVarietyName_enc"]

X_encoded = pd.get_dummies(X, drop_first=True)
X_encoded.columns = X_encoded.columns.str.replace("[^\\w\\s]", "", regex=True)

_strat = y if y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=_strat
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 15],
    "min_samples_leaf": [1, 2],
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)
print("Best params:", grid_search.best_params_)

rf_model = grid_search.best_estimator_
y_pred_rf = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.2f}")
print()
print("Classification Report for Random Forest:")
print()
print(
    classification_report(
        y_test,
        y_pred_rf,
        labels=list(range(len(variety_encoder.classes_))),
        target_names=list(variety_encoder.classes_),
        zero_division=0,
    )
)

fig, ax = plt.subplots(figsize=(12, 10))
_labels = list(range(len(variety_encoder.classes_)))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_rf,
    labels=_labels,
    display_labels=list(variety_encoder.classes_),
    ax=ax,
    xticks_rotation=45,
    colorbar=True,
)
plt.title("Random Forest - confusion matrix (crop variety)")
plt.tight_layout()
plt.show()
"""


def _lines(src: str) -> list[str]:
    return [line + "\n" for line in src.splitlines()]


def _clear_outputs(nb: dict[str, Any]) -> None:
    for c in nb["cells"]:
        c["execution_count"] = None
        if c.get("cell_type") == "code":
            c["outputs"] = []


def cmd_merge_cell() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    nb["cells"][3]["source"] = _lines(MERGE_CELL)
    _clear_outputs(nb)
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated merge cell (index 3) in {NB}")


def cmd_ml_cells() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    cells[14]["source"] = _lines(CELL_14)
    cells[15]["source"] = _lines(CELL_15)
    cells[17]["source"] = _lines(CELL_17)
    _clear_outputs(nb)
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated ML cells (14, 15, 17) in {NB}")


SAMPLE_DATA_SCRIPT = ROOT / "scripts" / "generate_sample_data.py"


def cmd_sample_data() -> None:
    if not SAMPLE_DATA_SCRIPT.is_file():
        print(f"Missing {SAMPLE_DATA_SCRIPT}", file=sys.stderr)
        sys.exit(1)
    runpy.run_path(str(SAMPLE_DATA_SCRIPT), run_name="__main__")


def cmd_all() -> None:
    cmd_sample_data()
    cmd_merge_cell()
    cmd_ml_cells()
    print(
        "Done. Re-run the notebook to refresh outputs, e.g.\n"
        + "  jupyter nbconvert --to notebook --execute notebooks/Datathon.ipynb --inplace"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate demo data and/or rewrite canonical cells in Datathon.ipynb."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sample = sub.add_parser(
        "sample-data",
        help="Run scripts/generate_sample_data.py (writes CSVs under data/)",
    )
    p_sample.set_defaults(_run=lambda: cmd_sample_data())

    p_merge = sub.add_parser(
        "merge-cell",
        help="Set join cell to skip PlantVariety merge when variety is on Plant.csv",
    )
    p_merge.set_defaults(_run=lambda: cmd_merge_cell())

    p_ml = sub.add_parser(
        "ml-cells",
        help="Set variety encoding + RandomForest/GridSearch/confusion-matrix cells",
    )
    p_ml.set_defaults(_run=lambda: cmd_ml_cells())

    p_all = sub.add_parser(
        "all",
        help="sample-data, then merge-cell, then ml-cells (clears notebook outputs)",
    )
    p_all.set_defaults(_run=lambda: cmd_all())

    args = parser.parse_args()
    args._run()


if __name__ == "__main__":
    main()
