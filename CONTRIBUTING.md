# Contributing

## Environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Regenerate demo data

```bash
python scripts/generate_sample_data.py
# or: python scripts/maintain.py sample-data
```

## Notebook maintenance

Canonical cells in `notebooks/Datathon.ipynb` can be restored without hand-editing JSON:

```bash
python scripts/maintain.py merge-cell   # join step for Plant.csv + variety
python scripts/maintain.py ml-cells     # encoding + Random Forest / GridSearch / confusion matrix
python scripts/maintain.py all          # sample-data + merge-cell + ml-cells
python scripts/maintain.py --help
```

After `merge-cell`, `ml-cells`, or `all`, re-run the notebook or use:

```bash
jupyter nbconvert --to notebook --execute notebooks/Datathon.ipynb --inplace
```

## Train and export a fitted model (optional)

Writes `models/*.joblib` and JSON sidecars (ignored by git):

```bash
python scripts/train_export_model.py
python scripts/predict_variety.py --row 0
```

## CI

Pull requests run `.github/workflows/notebook.yml`, which installs dependencies, builds demo data, and executes `Datathon.ipynb` end-to-end.
