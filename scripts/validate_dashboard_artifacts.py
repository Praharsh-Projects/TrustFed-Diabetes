from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED = [
    "dashboard_metrics.csv",
    "dashboard_calibration.csv",
    "dashboard_fairness.csv",
    "dashboard_rounds.csv",
    "dashboard_shap.csv",
    "dashboard_stability.csv",
    "dashboard_thresholds.csv",
    "dashboard_local_explanations.csv",
    "dashboard_curves.csv",
    "dashboard_confusion.csv",
    "dashboard_showcase_metrics.csv",
    "dashboard_score_ceiling.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate curated dashboard artifacts without mutating results.")
    parser.add_argument("--summary-dir", default="results/summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_dir = Path(args.summary_dir)
    missing = [name for name in REQUIRED if not (summary_dir / name).exists()]
    report: dict[str, object] = {"summary_dir": str(summary_dir), "missing_files": missing, "checks": {}}
    if missing:
        raise SystemExit(json.dumps(report, indent=2))

    metrics = _read_csv(summary_dir / "dashboard_metrics.csv")
    fairness = _read_csv(summary_dir / "dashboard_fairness.csv")
    thresholds = _read_csv(summary_dir / "dashboard_thresholds.csv")
    showcase = _read_csv(summary_dir / "dashboard_showcase_metrics.csv")
    curves = _read_csv(summary_dir / "dashboard_curves.csv")
    confusion = _read_csv(summary_dir / "dashboard_confusion.csv")
    score_ceiling = _read_csv(summary_dir / "dashboard_score_ceiling.csv")

    checks = {
        "metrics_nonempty": bool(not metrics.empty),
        "fairness_nonempty": bool(not fairness.empty),
        "thresholds_nonempty": bool(not thresholds.empty),
        "showcase_nonempty": bool(not showcase.empty),
        "curves_nonempty": bool(not curves.empty),
        "confusion_nonempty": bool(not confusion.empty),
        "score_ceiling_nonempty": bool(not score_ceiling.empty),
        "auroc_in_range": bool(metrics["roc_auc_mean"].between(0, 1, inclusive="both").all()),
        "f1_in_range": bool(metrics["f1_mean"].between(0, 1, inclusive="both").all()),
        "pr_auc_in_range": bool(
            metrics["pr_auc_mean"].between(0, 1, inclusive="both").all() if "pr_auc_mean" in metrics.columns else True
        ),
        "ece_in_range": bool(metrics["ece_mean"].between(0, 1, inclusive="both").all()),
        "selection_rate_in_range": bool(fairness["selection_rate_mean"].dropna().between(0, 1, inclusive="both").all()),
        "fairness_gaps_nonnegative": bool(
            fairness["demographic_parity_difference_mean"].dropna().ge(0).all()
            and fairness["equalized_odds_difference_mean"].dropna().ge(0).all()
        ),
        "human_readable_thresholds": bool(not metrics["threshold_strategy"].astype(str).str.fullmatch(r"__all__").any()),
        "showcase_has_cdc": bool(showcase["dataset"].astype(str).eq("cdc").any()) or bool(not metrics["dataset"].astype(str).eq("cdc").any()),
        "showcase_defaults_include_threshold_tuned": bool(showcase["threshold_strategy"].astype(str).eq("calib_f1_optimal").any()),
        "curve_types_present": bool({"roc", "pr", "score_hist"}.issubset(set(curves["curve_type"].astype(str).unique()))),
        "confusion_cells_present": bool({"TN", "FP", "FN", "TP"}.issubset(set(confusion["cell"].astype(str).unique()))),
        "near99_not_labeled_accuracy": bool(
            not (
                score_ceiling["metric"].astype(str).eq("accuracy")
                & pd.to_numeric(score_ceiling["value"], errors="coerce").ge(0.99)
            ).any()
        ),
    }
    report["checks"] = checks
    report["valid"] = bool(all(checks.values()))
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


if __name__ == "__main__":
    main()
