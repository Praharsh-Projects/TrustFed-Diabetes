# TrustFed-Diabetes Complete Professor Handover

This document is the professor-facing handover for the full project. It explains exactly what is included in the repository, where the datasets are stored, which runs are supporting comparisons, which run is the final thesis evidence package, and the exact commands required to rerun the project from the beginning on a clean Windows machine.

## 1. What this handover includes

This repository now contains:

- full project source code
- configurations for all major experiment tracks
- scripts for running experiments, aggregating results, validating dashboard artifacts, and exporting thesis assets
- unit tests
- both datasets inside the repository
- current full-CDC summary artifacts
- final thesis tables and exported evidence
- the detailed full-CDC Results chapter draft in both Markdown and Word format

The two datasets are stored here:

```text
data/raw/cdc_diabetes_health_indicators.csv
data/raw/pima_diabetes.csv
```

The detailed Results chapter draft is stored here:

```text
thesis_assets/full_cdc_polished/RESULTS_SECTION_DRAFT.md
thesis_assets/full_cdc_polished/RESULTS_SECTION_DRAFT.docx
```

## 2. How the project is organized

The project should be understood as three levels of execution.

### Level 1. Environment and correctness check

This is the minimal verification path:

- create a Python virtual environment
- install requirements
- verify both datasets are present
- run unit tests
- optionally open the dashboard from the included full-CDC summaries

### Level 2. Broad two-dataset comparison run

This is the broader supporting comparison package:

- configuration: `configs/polished_core.json`
- datasets: `Pima` and `CDC`
- purpose: broad comparison across `Centralized`, `Local-Only`, `Average Local-Only`, and `Federated`

This run supports the wider comparison story of the project.

### Level 3. Final full-CDC thesis run

This is the final thesis evidence package:

- broad audit configuration: `configs/full_cdc_polished.json`
- visual artifact configuration: `configs/full_cdc_visual_verify.json`
- dataset focus: full `CDC`

This run is the source of:

- the final dashboard
- the final confusion matrices
- the final calibration, fairness, explainability, and communication views
- the detailed full-CDC Results chapter

In short:

- `Pima` is part of the overall project and the broad comparison story
- `full CDC` is the final thesis evidence source

## 3. Exact from-scratch commands

### 3.1 Clone the repository

```powershell
git clone https://github.com/Praharsh-Projects/TrustFed-Diabetes.git
cd TrustFed-Diabetes
```

### 3.2 Create and activate the virtual environment

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

### 3.3 Verify the datasets are present

```powershell
Get-Item data/raw/cdc_diabetes_health_indicators.csv
Get-Item data/raw/pima_diabetes.csv
```

### 3.4 Run tests first

```powershell
py -m unittest tests.test_core
```

### 3.5 Run the broad two-dataset comparison from scratch

```powershell
py scripts/run_matrix.py --config configs/polished_core.json
py scripts/aggregate_results.py --results-dir results/polished_runs --output-dir results/polished_summary
py scripts/validate_dashboard_artifacts.py --summary-dir results/polished_summary
py scripts/export_thesis_assets.py --summary-dir results/polished_summary --output-dir thesis_assets/polished
```

### 3.6 Run the final full-CDC broad audit from scratch

```powershell
py scripts/run_matrix.py --config configs/full_cdc_polished.json
py scripts/aggregate_results.py --results-dir results/full_cdc_polished_runs --output-dir results/full_cdc_polished_summary
py scripts/validate_dashboard_artifacts.py --summary-dir results/full_cdc_polished_summary
py scripts/export_thesis_assets.py --summary-dir results/full_cdc_polished_summary --output-dir thesis_assets/full_cdc_polished
```

### 3.7 Run the full-CDC visual package from scratch

```powershell
py scripts/run_matrix.py --config configs/full_cdc_visual_verify.json
py scripts/aggregate_results.py --results-dir results/full_cdc_visual_runs --output-dir results/full_cdc_visual_summary
py scripts/validate_dashboard_artifacts.py --summary-dir results/full_cdc_visual_summary
```

### 3.8 Launch the final dashboard

```powershell
py scripts/run_dashboard.py --results-dir results/full_cdc_polished_summary --visual-results-dir results/full_cdc_visual_summary --port 8057
```

Open:

```text
http://127.0.0.1:8057
```

### 3.9 Open the final written results draft

Markdown source:

```text
thesis_assets/full_cdc_polished/RESULTS_SECTION_DRAFT.md
```

Word file:

```text
thesis_assets/full_cdc_polished/RESULTS_SECTION_DRAFT.docx
```

## 4. What outputs to inspect after rerunning

### Broad two-dataset comparison outputs

```text
results/polished_summary/dashboard_metrics.csv
results/polished_summary/dashboard_calibration.csv
results/polished_summary/dashboard_fairness.csv
results/polished_summary/dashboard_rounds.csv
```

### Final full-CDC outputs

```text
results/full_cdc_polished_summary/dashboard_showcase_metrics.csv
results/full_cdc_polished_summary/dashboard_metrics.csv
results/full_cdc_polished_summary/dashboard_calibration.csv
results/full_cdc_polished_summary/dashboard_fairness.csv
results/full_cdc_polished_summary/dashboard_thresholds.csv
results/full_cdc_polished_summary/dashboard_shap.csv
results/full_cdc_polished_summary/dashboard_score_ceiling.csv
```

### Prediction-backed visual outputs

```text
results/full_cdc_visual_summary/dashboard_confusion.csv
results/full_cdc_visual_summary/dashboard_curves.csv
results/full_cdc_visual_summary/dashboard_local_explanations.csv
results/full_cdc_visual_summary/dashboard_stability.csv
```

### Final thesis outputs

```text
thesis_assets/full_cdc_polished/thesis_conclusion_summary.md
thesis_assets/full_cdc_polished/tables/
thesis_assets/full_cdc_polished/RESULTS_SECTION_DRAFT.md
thesis_assets/full_cdc_polished/RESULTS_SECTION_DRAFT.docx
```

## 5. Important interpretation notes

The confusion matrix uses the **held-out test fold**, not all `253,680` CDC rows mixed together. That is why the confusion counts do not sum to the full raw dataset size.

The score-distribution chart shows **normalized within-class distributions**, not raw full-dataset counts. It is intended to show how positive and negative score distributions are shaped, not to count every case directly.

The broad audit package is the source of the final best-row claims and the overall thesis comparison narrative.

The visual package is the source of the prediction-backed charts such as:

- confusion matrices
- score distributions
- representative local explanations
- explanation stability visuals

`full_cdc_polished` is the final thesis evidence package.

`polished_core` is the broader two-dataset supporting comparison package.

## 6. Final handover interpretation

If the project is being reviewed as a complete thesis handover, the cleanest reading is:

- both datasets are included directly in the repository
- the professor can install the environment from scratch
- the professor can rerun the broad two-dataset comparison
- the professor can rerun the final full-CDC thesis package
- the professor can open the final dashboard
- the professor can inspect the detailed written Results chapter that corresponds to the final full-CDC evidence package

This makes the handover reproducible, auditable, and complete without depending on a separate hidden data folder or undocumented local steps.
