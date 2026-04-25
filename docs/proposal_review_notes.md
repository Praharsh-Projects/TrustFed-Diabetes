# Proposal And Blueprint Review Notes

## Sources Reviewed

- `annotated-Master_thesis_proposal.pdf`
- `Federated Learning Thesis Blueprint for Reliable Diabetes Prediction With Calibration, Explainabilit.docx`

## Proposal Direction

The proposal frames the project as a federated learning system for reliable
diabetes prediction. Its core implementation intent is:

- Simulate multiple healthcare clients without sharing raw patient records.
- Compare centralized, local/client, and federated behavior.
- Evaluate IID and non-IID client distributions.
- Compare classical machine learning models with a shallow MLP.
- Add SHAP explanations so feature contributions are visible.
- Add probability calibration, with isotonic regression as the proposal default.
- Report classical metrics plus calibration metrics such as Brier score, ECE,
  and log loss.

## Blueprint Upgrades Applied

The blueprint keeps the proposal direction but makes it more defensible for the
thesis review by requiring measurable trustworthiness outputs. The prototype
therefore includes:

- Public-data-ready Pima loading plus an offline synthetic fallback.
- Dirichlet non-IID partitioning with a tunable alpha value.
- Centralized ML/DL baselines.
- A FedAvg-style federated logistic regression simulator.
- A calibration-aware aggregation option.
- Local and federated calibration artifacts.
- Fairness reports for available age and BMI subgroups.
- Communication-cost logging per federated round.
- SHAP summary output and round-to-round feature ranking stability.
- A static dashboard artifact that can be opened without running a web server.

## Current Implementation Boundary

This is now a runnable implementation-focused thesis artifact. The core
federated algorithms are implemented in NumPy instead of Flower/Torch so the
experiments remain reproducible with a small dependency set. Dash is used for
the interactive audit dashboard, while the static HTML report remains available
as a fallback/export artifact.

## Recommended Next Iterations

1. Run the full `configs/core.json` matrix when compute time is available.
2. Inspect the aggregated result tables before drafting thesis claims.
3. Use synthetic results only for smoke testing, never as final evidence.
4. Keep Flower, secure aggregation, and differential privacy as future work
   unless the thesis scope is formally expanded.
