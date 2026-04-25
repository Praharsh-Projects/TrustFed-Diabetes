# TrustFed-Diabetes

This workspace contains a runnable implementation for the thesis proposal:
privacy-preserving diabetes prediction with federated simulation, calibration,
explainability, fairness checks, communication logging, and a dashboard artifact.

For the exact clone/setup/run steps for the current full-CDC thesis build, use:

- [`docs/RUN_PROJECT.md`](docs/RUN_PROJECT.md)

The core experiment engine is dependency-light: `numpy`, `pandas`,
`scikit-learn`, `plotly`, and `shap`. The interactive audit dashboard uses
Dash, and public UCI dataset downloads use `ucimlrepo` when available.

## What Is Built

- Public-data-ready loaders for the Pima Indians Diabetes dataset, plus an
  offline synthetic diabetes-like fallback for smoke tests.
- IID and non-IID Dirichlet client partitioning.
- Centralized baselines for Logistic Regression, Random Forest, Gradient
  Boosting, Decision Tree, shallow MLP, and XGBoost.
- NumPy federated simulators for logistic regression and shallow MLP with
  `fedavg` and `fedprox`.
- Global and federated post-hoc probability calibration with isotonic or
  sigmoid scaling.
- Decision-threshold tuning from the calibration split for F1-oriented and
  Youden-J operating points.
- Metrics for performance, calibration, fairness, communication cost, and
  explanation stability.
- SHAP summary generation with a deterministic fallback for linear models.
- Static Plotly dashboard generation from saved experiment artifacts.
- Interactive Dash audit dashboard over curated aggregate tables.
- Showcase-first dashboard mode with ROC/PR curves, confusion matrices, score
  distributions, and threshold tradeoff visuals backed by saved prediction
  artifacts.
- Config-driven experiment matrix execution.
- Dashboard artifact validation and thesis asset export for figures and tables.

## Quick Start

To open the included full-CDC dashboard without retraining:

```powershell
py scripts/run_dashboard.py --results-dir results/full_cdc_polished_summary --visual-results-dir results/full_cdc_visual_summary --port 8057
```

Then open:

```text
http://127.0.0.1:8057
```

## Reproducibility Note

The raw experiment run folders are intentionally not kept in the GitHub package because they are extremely large. The repository keeps:

- source code
- configs
- scripts
- tests
- full-CDC summary artifacts
- thesis-ready exported evidence

The heavy raw run directories can be regenerated from the commands in [`docs/RUN_PROJECT.md`](docs/RUN_PROJECT.md).

Run the offline smoke experiment:

```powershell
py scripts/run_experiment.py --dataset synthetic --synthetic-samples 500 --rounds 4 --local-epochs 1 --output-dir outputs/synthetic_smoke
```

Open the generated dashboard:

```text
outputs/synthetic_smoke/dashboard.html
```

The config-driven smoke command is:

```powershell
py scripts/run_experiment.py --config configs/smoke.json
```

## Using The Pima Dataset

Place a Pima CSV at:

```text
data/raw/pima_diabetes.csv
```

The loader accepts either standard column names:

```text
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
```

or a CSV where the final column is the target.

Then run:

```powershell
py scripts/run_experiment.py --dataset pima --data-path data/raw/pima_diabetes.csv --output-dir outputs/pima_fedavg
```

Download public datasets when network access is available:

```powershell
py scripts/download_datasets.py --dataset pima
py scripts/download_datasets.py --dataset cdc
```

If automatic download fails, place the files manually at:

```text
data/raw/pima_diabetes.csv
data/raw/cdc_diabetes_health_indicators.csv
```

## Useful Experiment Variants

Non-IID experiment with stronger heterogeneity:

```powershell
py scripts/run_experiment.py --dataset synthetic --partition non_iid --alpha 0.3 --output-dir outputs/non_iid_alpha_03
```

Sigmoid calibration:

```powershell
py scripts/run_experiment.py --dataset synthetic --calibration sigmoid --output-dir outputs/sigmoid_calibration
```

F1-oriented decision threshold tuning:

```powershell
py scripts/run_experiment.py --dataset synthetic --decision-threshold-strategy calib_f1_optimal --output-dir outputs/f1_threshold_view
```

## Output Artifacts

Each run writes:

- `summary.json`: experiment configuration and headline metrics.
- `metrics.csv`: centralized and federated performance/calibration metrics.
- `round_history.csv`: federated training dynamics and communication cost.
- `calibration_bins.csv`: reliability diagram bins.
- `fairness.csv`: subgroup selection, TPR/FPR, demographic parity, and equalized
  odds summaries.
- `shap_summary.csv`: top feature attributions.
- `stability.csv`: explanation/ranking stability over FL rounds.
- `dashboard.html`: static dashboard for review/demo.
- `client_manifest.csv`: reproducible client row assignments.
- `client_metrics.csv`: per-client federated validation metrics.
- `communication.csv`: server/client communication accounting.
- `local_explanations.csv`: local instance explanation rows.
- `test_predictions.csv` and `calibration_predictions.csv` when
  `save_prediction_artifacts` is enabled for showcase-style reruns.

Aggregate and export thesis assets:

```powershell
py scripts/run_matrix.py --config configs/core.json
py scripts/aggregate_results.py --results-dir results/runs --output-dir results/summary
py scripts/export_thesis_assets.py --summary-dir results/summary --output-dir thesis_assets
```

Polished smoke and thesis-grade runs:

```powershell
py scripts/run_matrix.py --config configs/polished_smoke.json
py scripts/aggregate_results.py --results-dir results/polished_smoke_runs --output-dir results/polished_smoke_summary
py scripts/validate_dashboard_artifacts.py --summary-dir results/polished_smoke_summary
py scripts/export_thesis_assets.py --summary-dir results/polished_smoke_summary --output-dir thesis_assets/polished_smoke

py scripts/run_matrix.py --config configs/polished_core.json
py scripts/aggregate_results.py --results-dir results/polished_runs --output-dir results/polished_summary
py scripts/validate_dashboard_artifacts.py --summary-dir results/polished_summary
py scripts/export_thesis_assets.py --summary-dir results/polished_summary --output-dir thesis_assets/polished
```

Run the interactive dashboard:

```powershell
py scripts/run_dashboard.py --results-dir results/polished_smoke_summary
```

Showcase-first score track:

```powershell
py scripts/run_matrix.py --config configs/showcase_smoke.json
py scripts/aggregate_results.py --results-dir results/showcase_smoke_runs --output-dir results/showcase_smoke_summary
py scripts/validate_dashboard_artifacts.py --summary-dir results/showcase_smoke_summary
py scripts/export_thesis_assets.py --summary-dir results/showcase_smoke_summary --output-dir thesis_assets/showcase_smoke

py scripts/run_matrix.py --config configs/showcase_core.json
py scripts/aggregate_results.py --results-dir results/showcase_runs --output-dir results/showcase_summary
py scripts/validate_dashboard_artifacts.py --summary-dir results/showcase_summary
py scripts/export_thesis_assets.py --summary-dir results/showcase_summary --output-dir thesis_assets/showcase
py scripts/run_dashboard.py --results-dir results/showcase_summary
```

## Proposal Alignment

The proposal asks for a federated learning framework for diabetes prediction
that compares ML/DL models, handles IID and non-IID settings, adds SHAP-based
explainability, and improves probability reliability through calibration.

The blueprint adds reviewer-facing rigor: measurable research questions,
calibration metrics, fairness, communication cost, and explanation stability.
This implementation produces those measurable artifacts so the thesis can move
from concept to repeatable experiments.

## Next Build Steps

1. Run the full `configs/polished_core.json` matrix after confirming runtime budget.
2. Review `results/polished_summary/dashboard_metrics.csv` and exported tables for thesis evidence.
3. Use `thesis_assets/polished/figures/*.html` as figure sources.
4. Treat synthetic runs as smoke tests only; do not use them as thesis evidence.
