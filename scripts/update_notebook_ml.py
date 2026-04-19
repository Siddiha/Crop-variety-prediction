"""Update ML cells in notebooks/Datathon.ipynb (variety target, GridSearch, confusion matrix)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "Datathon.ipynb"

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

CELL_17 = r"""from sklearn.model_selection import train_test_split, GridSearchCV
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
X_encoded.columns = X_encoded.columns.str.replace(r"[^\w\s]", "", regex=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
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
print("\\nClassification Report for Random Forest:\\n")
print(
    classification_report(
        y_test,
        y_pred_rf,
        labels=list(range(len(variety_encoder.classes_))),
        target_names=list(variety_encoder.classes_),
        zero_division=0,
    )
)

fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_rf,
    display_labels=variety_encoder.classes_,
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


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    cells[14]["source"] = _lines(CELL_14)
    cells[15]["source"] = _lines(CELL_15)
    cells[17]["source"] = _lines(CELL_17)

    for c in cells:
        c["execution_count"] = None
        if c.get("cell_type") == "code":
            c["outputs"] = []

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated ML cells in {NB}")


if __name__ == "__main__":
    main()
