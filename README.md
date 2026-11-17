# Site-Robust Camera-Trap Species Classification

Camera traps in East African parks take far more photos than people can review, and most
of those photos are empty. This project builds a service that filters out empty frames,
identifies the species in the rest, and flags uncertain predictions for a person to check.
The model is evaluated on camera sites it never saw during training, because that is how
it would be used in practice.

**Course:** DSA 8401 Applied Machine Learning, Strathmore University
**Author:** Diana Adhiambo Owuoth (093481)
**Data:** [Snapshot Serengeti](https://lila.science/datasets/snapshot-serengeti), Season 1, via LILA BC (CDLA-Permissive-1.0)

> Status: data pipeline complete. Modelling, API and deployment are in progress.

## The data at a glance (Season 1)

| | |
|---|---|
| Photos / bursts / cameras | 411,414 / 148,226 / 168 |
| Empty bursts | 82.7% |
| Bursts after cleaning | 145,831 |
| Classes | 12 species + `other` + `empty` |
| Split by camera (train / val / test) | 117 / 26 / 25 cameras |
| Photos used for modelling | 8,069 (one per burst) |

The full audit, including the missing-data analysis, is in
[`notebooks/01_data_audit.ipynb`](notebooks/01_data_audit.ipynb).

## Data pipeline

Every setting lives in [`configs/config.yaml`](configs/config.yaml). Run the steps in order
from the project folder:

| Step | Command | Output |
|---|---|---|
| 1. Download the labels | `python -m camtrap.data.download` | `data/raw/SnapshotSerengetiS01.json` |
| 2. Clean | `python -m camtrap.data.clean` | `data/processed/bursts.csv` |
| 3. Split by camera and pick the sample | `python -m camtrap.data.split` | `data/processed/bursts_split.csv`, `sample.csv` |
| 4. Download and resize the photos | `python -m camtrap.data.download_images` | `data/images/<split>/<class>/` |

Step 4 downloads about 5 GB, so it was run on Google Colab. No data is stored in this
repository; the pipeline recreates it.

## Project layout

```
configs/        pipeline settings (config.yaml)
notebooks/      data audit and analysis
src/camtrap/
  data/         download, cleaning, camera split, photo download
  models/       baseline and CNN models
  evaluation/   metrics and error analysis
  serving/      FastAPI service
tests/          unit tests
reports/        figures, final report and slides
data/, models/  created by the pipeline (not committed)
```

## Setup

Python 3.10 or later.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## AI assistance

See [`AI_USE.md`](AI_USE.md).
