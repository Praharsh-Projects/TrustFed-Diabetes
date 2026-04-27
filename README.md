# TrustFed-Diabetes

This repository is a **run-only handover package** for the final diabetes federated-learning dashboard.

It is designed so that another machine can:

- clone the repo
- install the dependencies
- open the **same final bundled results**
- do that **without retraining the models**

## What this repo includes

- runnable project code under `src/`
- launcher and reproducibility scripts under `scripts/`
- both datasets in:
  - `data/raw/cdc_diabetes_health_indicators.csv`
  - `data/raw/pima_diabetes.csv`
- the bundled final result package in:
  - `results/full_cdc_polished_summary/`
- the final full-CDC retraining configs:
  - `configs/full_cdc_polished.json`
  - `configs/full_cdc_visual_verify.json`
- a small verification test suite in `tests/test_core.py`

This repo does **not** include:

- LaTeX thesis source files
- huge raw training-run folders
- old thesis-export bundles that are not needed to run the project

## Fastest way to get the same final results without retraining

Clone the repo:

```powershell
git clone https://github.com/Praharsh-Projects/TrustFed-Diabetes.git
cd TrustFed-Diabetes
```

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Optional environment check:

```powershell
py -m unittest tests.test_core
```

Run the dashboard from the bundled final results:

```powershell
py scripts/run_dashboard.py
```

Then open:

```text
http://127.0.0.1:8050
```

## Deploy the same bundled dashboard

This repo now includes a production entrypoint in `app.py` and a `Dockerfile`, so you can deploy the bundled final dashboard without retraining.

### Local production-style run

After installing the requirements, you can run the deployable entrypoint directly:

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:8050
```

### Docker run

Build the image:

```powershell
docker build -t trustfed-diabetes .
```

Run the container:

```powershell
docker run -p 8050:8050 trustfed-diabetes
```

Then open:

```text
http://127.0.0.1:8050
```

### Deploy on a cloud host

Any host that can run a Python container can serve this project directly from the bundled `results/full_cdc_polished_summary/` folder.

- default start command: `python app.py`
- default port: `8050`
- host binding: `0.0.0.0`

If your platform injects a `PORT` environment variable, `app.py` uses it automatically.

## Deploy a free public live link on Hugging Face Spaces

This project now includes a Space-ready deployment bundle and a publisher script.

### 1. Build the Space bundle

```powershell
py scripts/build_hf_space_bundle.py
```

This creates a deployment subset in:

```text
deploy/huggingface-space
```

It includes only:

- `app.py`
- `Dockerfile`
- `requirements.txt`
- `src/fl_diabetes/`
- the runtime dashboard files from `results/full_cdc_polished_summary/`

It excludes datasets, tests, configs, local logs, and unused large training-summary files.

### 2. Publish the Space

Install the Hugging Face Hub client:

```powershell
py -m pip install huggingface_hub
```

Set your Hugging Face token in PowerShell:

```powershell
$env:HF_TOKEN="your_hugging_face_token"
```

Publish the public Docker Space:

```powershell
py scripts/publish_hf_space.py --space-id YOUR_USERNAME/trustfed-diabetes-live
```

If the Space does not exist yet, the script creates it as a public Docker Space and uploads the deployment bundle.

The live link will be:

```text
https://huggingface.co/spaces/YOUR_USERNAME/trustfed-diabetes-live
```

Important note:

- this deployment is view-only
- it serves the bundled final results
- it does not retrain any models
- free Spaces can cold-start after being idle

## Why the results are the same without training

The folder `results/full_cdc_polished_summary/` already contains the precomputed final dashboard tables.

That means:

- opening the dashboard reads saved CSV outputs
- it does **not** retrain the models
- the displayed results should therefore match the shipped result bundle on another machine

This is the correct path if your goal is to review the final project evidence and dashboard exactly as packaged.

## Optional: retrain the final full-CDC project from scratch

Use this only if you want to regenerate the full experiment outputs.

Run the final full-CDC matrix:

```powershell
py scripts/run_matrix.py --config configs/full_cdc_polished.json
```

Aggregate the results:

```powershell
py scripts/aggregate_results.py --results-dir results/full_cdc_polished_runs --output-dir results/full_cdc_polished_summary
```

Validate the dashboard artifacts:

```powershell
py scripts/validate_dashboard_artifacts.py --summary-dir results/full_cdc_polished_summary
```

Open the dashboard:

```powershell
py scripts/run_dashboard.py
```

Note: if you retrain from scratch on another machine, the results should be very close, but tiny differences can still happen because of library versions and numerical training behavior.

## Included datasets

The repo already includes both datasets:

```text
data/raw/cdc_diabetes_health_indicators.csv
data/raw/pima_diabetes.csv
```

## Important interpretation note

The confusion matrix in the dashboard shows the **held-out test fold**, not all `253,680` CDC rows mixed together.

That is why the confusion counts do not sum to the full raw dataset size. This is the correct evaluation setup.
