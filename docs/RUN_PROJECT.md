# TrustFed-Diabetes Run Guide

This guide is the exact setup and run path for the current full-CDC thesis version of the project.

## 1. What is included in this GitHub repo

This repository includes:

- source code
- configs
- scripts
- tests
- lightweight full-CDC summary artifacts
- thesis-ready exported tables and figures

This repository does **not** include the huge raw experiment run folders, because they are too large for GitHub. Those can be reproduced from the commands below.

## 2. Requirements

- Windows with PowerShell
- Python `3.11`
- enough disk space for full CDC experiment outputs

## 3. Clone the project

```powershell
git clone https://github.com/Praharsh-Projects/TrustFed-Diabetes.git
cd TrustFed-Diabetes
```

## 4. Create and activate a virtual environment

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.venv\Scripts\Activate.ps1
```

## 5. Put the dataset in the expected location

Place the CDC dataset CSV here:

```text
data/raw/cdc_diabetes_health_indicators.csv
```

If you want to try automatic download first:

```powershell
py scripts/download_datasets.py --dataset cdc
```

If automatic download fails, place the file manually in `data/raw/`.

## 6. Run the dashboard from the included full-CDC summaries

This is the fastest way to open the thesis dashboard without retraining:

```powershell
py scripts/run_dashboard.py --results-dir results/full_cdc_polished_summary --visual-results-dir results/full_cdc_visual_summary --port 8057
```

Then open:

```text
http://127.0.0.1:8057
```

## 7. Reproduce the full-CDC broad audit run

This reruns the full-data CDC thesis experiment matrix:

```powershell
py scripts/run_matrix.py --config configs/full_cdc_polished.json
```

Aggregate the results:

```powershell
py scripts/aggregate_results.py --results-dir results/full_cdc_polished_runs --output-dir results/full_cdc_polished_summary
```

Validate the summary:

```powershell
py scripts/validate_dashboard_artifacts.py --summary-dir results/full_cdc_polished_summary
```

## 8. Reproduce the full-CDC visual artifact run

This creates the saved prediction-based artifacts used by the showcase visuals:

```powershell
py scripts/run_matrix.py --config configs/full_cdc_visual_verify.json
```

Aggregate the visual results:

```powershell
py scripts/aggregate_results.py --results-dir results/full_cdc_visual_runs --output-dir results/full_cdc_visual_summary
```

Validate the visual summary:

```powershell
py scripts/validate_dashboard_artifacts.py --summary-dir results/full_cdc_visual_summary
```

## 9. Export thesis-ready assets

```powershell
py scripts/export_thesis_assets.py --summary-dir results/full_cdc_polished_summary --output-dir thesis_assets/full_cdc_polished
```

This generates:

- final summary tables
- HTML figures
- conclusion summary files

## 10. Open the final dashboard after rerunning everything

```powershell
py scripts/run_dashboard.py --results-dir results/full_cdc_polished_summary --visual-results-dir results/full_cdc_visual_summary --port 8057
```

## 11. Run tests

```powershell
py -m unittest tests.test_core
```

## 12. Important interpretation note

The confusion matrices in the dashboard show the **held-out test split**, not all rows mixed together. That is correct methodology.

For the full CDC dataset:

- raw data size is about `253,680` rows
- the dashboard confusion matrices reflect the **test fold** of that full dataset

## 13. GitHub update workflow

After making changes:

```powershell
git status
git add .
git commit -m "Describe the change clearly"
git push origin main
```

If you only changed documentation or dashboard wording, say that in the commit message.
