"""Patch notebooks/Datathon.ipynb: portable data paths, in-memory zones, imports, dedupe plots."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "Datathon.ipynb"


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    cells[0]["source"] = [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from pathlib import Path\n",
        "\n",
        "_cwd = Path.cwd().resolve()\n",
        "if (_cwd / \"data\" / \"HumidityLookup.csv\").exists():\n",
        "    DATA_DIR = _cwd / \"data\"\n",
        "elif (_cwd.parent / \"data\" / \"HumidityLookup.csv\").exists():\n",
        "    DATA_DIR = _cwd.parent / \"data\"\n",
        "else:\n",
        "    DATA_DIR = _cwd.parent / \"data\"\n",
        "\n",
        "sns.set_theme(style=\"whitegrid\")\n",
    ]

    cells[1]["source"] = [
        "df_Hum = pd.read_csv(DATA_DIR / \"HumidityLookup.csv\")\n",
        "df_Ph = pd.read_csv(DATA_DIR / \"PHRangeLookup.csv\")\n",
        "df_Soil = pd.read_csv(DATA_DIR / \"SoilTextureLookup.csv\")\n",
        "df_Var = pd.read_csv(DATA_DIR / \"PlantVariety.csv\")\n",
        "df_PlantType = pd.read_csv(DATA_DIR / \"PlantTypeLookup.csv\")\n",
        "df_Hard = pd.read_csv(DATA_DIR / \"PlantHardinessZoneLookup.csv\")\n",
        "df_Saline = pd.read_csv(DATA_DIR / \"SalinityLookup.csv\")\n",
        "df_OrgMat = pd.read_csv(DATA_DIR / \"OrganicMatterLookup.csv\")\n",
        "df_plant = pd.read_csv(DATA_DIR / \"Plant.csv\")\n",
    ]

    cells[2]["source"] = [
        "# Unify hardiness zones: aggregate temperature ranges by numeric zone\n",
        "df_Hard = df_Hard.copy()\n",
        "df_Hard[\"UnifiedZone\"] = df_Hard[\"Zone\"].astype(str).str.extract(r\"(\\d+)\")\n",
        "merged_zone_data = (\n",
        "    df_Hard.dropna(subset=[\"UnifiedZone\"])\n",
        "    .groupby(\"UnifiedZone\", as_index=False)\n",
        "    .agg({\"TemperatureStartRange\": \"min\", \"TemperatureEndRange\": \"max\"})\n",
        ")\n",
        "merged_zone_data[\"UnifiedZone\"] = merged_zone_data[\"UnifiedZone\"].astype(float)\n",
        "df_Hard_new = merged_zone_data\n",
    ]

    # Remove old cell that re-read unified zones from a local Downloads path
    del cells[3]

    # Remove duplicate feature-importance cell (keep the version with invert_yaxis)
    to_remove: list[int] = []
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if (
            "Random Forest Feature Importance" in src
            and "invert_yaxis" not in src
            and "feature_importances" in src
        ):
            to_remove.append(i)
    for i in sorted(to_remove, reverse=True):
        del cells[i]

    # Fix optional export path in commented cell
    for c in cells:
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if "cleaned_df2.csv" in src and "/Users/osheen" in src:
            c["source"] = [
                "# Optional: export cleaned table\n",
                "# cleaned_df.to_csv(DATA_DIR / \"cleaned_df.csv\", index=False)\n",
            ]

    # Drop trailing empty code cell if present
    while cells and cells[-1]["cell_type"] == "code" and not "".join(
        cells[-1].get("source", [])
    ).strip():
        cells.pop()

    for c in cells:
        c["execution_count"] = None
        if c.get("cell_type") == "code":
            c["outputs"] = []

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Patched {NB}")


if __name__ == "__main__":
    main()
