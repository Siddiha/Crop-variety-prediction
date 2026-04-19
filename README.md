# Crop variety prediction (predictive modeling)

This project joins agronomic lookup tables, builds features, and trains a **Random Forest** classifier with **`GridSearchCV`** to predict **crop variety** (`PlantVarietyName`) from soil, climate zone, humidity, salinity, organic matter, and related fields. **Plant type** (e.g. vegetable vs fruit) is kept as an input feature so the model can separate variety within type.

Choosing varieties that fit local conditions supports higher yields, lower risk, and more efficient use of water and inputs.

**Recognition:** Developed for a Datathon; the team placed **2nd runner-up**. [View the digital badge](https://badgr.com/public/assertions/rpl3BidYQJKToosP9B4jLg?identity__email=ogupta@horizon.csueastbay.edu).

## Table of contents

- [Overview](#overview)
- [Project structure](#project-structure)
- [Data](#data)
- [Technologies](#technologies)
- [Setup](#setup)
- [Run the notebook](#run-the-notebook)
- [How it works](#how-it-works)
- [Screenshots](#screenshots)
- [Example code](#example-code)
- [Features](#features)
- [Status](#status)
- [Challenges & learnings](#challenges--learnings)
- [License](#license)
- [Contact](#contact)

## Overview

The notebook loads CSVs from `data/`, merges hardiness zones, pH, soil texture, humidity, salinity, organic matter, plant type, and variety tables, cleans the result, then trains a **Random Forest** with **hyperparameter grid search**. Evaluation includes **accuracy**, a **classification report**, a **confusion matrix**, and **feature importance** plots.

## Project structure

```
Crop-variety-prediction/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/                    # CSV inputs (see Data section)
├── images/                  # Figures for the README (EDA, metrics, recommendations)
├── docs/                    # Extra materials (e.g. presentation slides), optional
├── notebooks/
│   └── Datathon.ipynb       # Main analysis and modeling
└── scripts/
    ├── generate_sample_data.py   # Builds synthetic CSVs under data/ for a runnable demo
    └── maintain.py               # Rewrite canonical notebook cells (see below)
```

## Data

- **Bundled demo:** The repo includes **synthetic sample CSVs** in `data/` so `notebooks/Datathon.ipynb` runs end-to-end after `pip install -r requirements.txt`. Regenerate them with:
  ```bash
  python scripts/generate_sample_data.py
  ```
- **Original Datathon data:** If you have the competition files, replace the contents of `data/` with those CSVs (same filenames expected by the notebook: `Plant.csv`, `PlantVariety.csv`, lookups, etc.). Metrics and plots will then reflect the real dataset.

The notebook resolves `data/` whether you start Jupyter from the **repository root** or the **`notebooks/`** folder (see first code cell).

### Maintainer tools

`scripts/maintain.py` can regenerate the **demo CSVs** and/or overwrite **canonical cells** in `notebooks/Datathon.ipynb` (for example after manual edits):

```bash
python scripts/maintain.py sample-data   # same as: python scripts/generate_sample_data.py
python scripts/maintain.py merge-cell    # join cell: variety on Plant.csv vs lookup merge
python scripts/maintain.py ml-cells      # encoding + Random Forest / GridSearch / confusion matrix
python scripts/maintain.py all           # sample-data, then merge-cell, then ml-cells
python scripts/maintain.py --help
```

`merge-cell`, `ml-cells`, and `all` clear saved notebook outputs; run the notebook again (or use `nbconvert --execute`) so outputs stay in sync.

## Technologies

- Python 3.10+
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn (Random Forest, `GridSearchCV`, metrics, confusion matrix display)
- Jupyter

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/Siddiha/Crop-variety-prediction.git
cd Crop-variety-prediction
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv .venv
```

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **macOS / Linux:** `source .venv/bin/activate`

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

## Run the notebook

From the **repository root**:

```bash
jupyter notebook notebooks/Datathon.ipynb
```

Or from `notebooks/`:

```bash
cd notebooks
jupyter notebook Datathon.ipynb
```

To re-run all cells and save outputs into the `.ipynb` file:

```bash
jupyter nbconvert --to notebook --execute notebooks/Datathon.ipynb --inplace
```

## How it works

1. **Load & explore** — Read lookup tables and `Plant.csv`; inspect distributions and missing values.
2. **Preprocess** — Drop incomplete rows where needed; normalize organic-matter labels for modeling.
3. **Feature engineering** — Merge lookups into one table; unify hardiness zones in memory (no local absolute paths).
4. **Encode target** — `LabelEncoder` on **crop variety** (`PlantVarietyName`).
5. **Train** — One-hot encode features (including **plant type**), then **`GridSearchCV`** over `n_estimators`, `max_depth`, and `min_samples_leaf`.
6. **Evaluate** — Accuracy, per-class report, **confusion matrix**, and **feature importance** plots.

## Screenshots

Figures below are from an **earlier Datathon run** (illustrative). Your notebook output will depend on the **data** in `data/` (demo sample vs original files).

### EDA and trends

![EDA](./images/EDA.png)

Exploratory views of varieties and zones.

### Model accuracy

![Model accuracy](./images/ModelAccuracy.png)

![Categories](./images/MA2.png)

Model diagnostics from the original run (example).

### Recommendations

![Recommendations](./images/Recommendations.png)

Recommendations aligned to zones, soil, and environmental factors.

## Example code

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 15],
    "min_samples_leaf": [1, 2],
}
grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best params:", grid_search.best_params_)
```

## Features

**Current**

- **Variety** prediction from engineered soil and climate features (with **plant type** as an input).
- **Grid search** for Random Forest hyperparameters.
- **Confusion matrix** and classification report for multi-class variety labels.
- Visual summaries (Seaborn count plots, feature importance).

**Possible next steps**

- Interactive assistant for farmer-facing Q&A.
- Weather or forecast features in the feature set.

## Status

Initial version is complete; contributions and refinements are welcome.

## Challenges & learnings

- **Challenges:** Class imbalance can affect metrics; grid search adds runtime; many variety classes make confusion matrices dense.
- **Learnings:** Tabular joins for agronomic data, `GridSearchCV`, and reporting with confusion matrices and feature importance.

## License

This project is released under the [MIT License](./LICENSE).

## Contact

- **Email:** FathimaSiddka62@gmail.com
