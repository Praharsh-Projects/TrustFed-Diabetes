from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fl_diabetes.metrics import aggregate_with_confidence_intervals


METRICS = [
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "balanced_accuracy",
    "f1",
    "pr_auc",
    "brier",
    "log_loss",
    "ece",
    "roc_auc",
    "selection_rate",
    "decision_threshold",
]
GROUPS = [
    "dataset",
    "experiment_track",
    "clients",
    "partition",
    "alpha",
    "run_type",
    "algorithm",
    "model",
    "calibration",
    "threshold_strategy",
]
SHOWCASE_RUN_TYPES = {"centralized", "federated"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-run artifacts into thesis-ready tables.")
    parser.add_argument("--results-dir", default="results/runs")
    parser.add_argument("--output-dir", default="results/summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "metrics": _concat_artifact(results_dir, "metrics.csv"),
        "fairness": _concat_artifact(results_dir, "fairness.csv"),
        "communication": _concat_artifact(results_dir, "communication.csv"),
        "calibration": _concat_artifact(results_dir, "calibration_bins.csv"),
        "shap": _concat_artifact(results_dir, "shap_summary.csv"),
        "stability": _concat_artifact(results_dir, "stability.csv"),
        "thresholds": _concat_artifact(results_dir, "threshold_sweeps.csv"),
        "local_explanations": _concat_artifact(results_dir, "local_explanations.csv"),
    }

    _write_raw_tables(output_dir, artifacts)

    metrics = _prepare_dashboard_scope(artifacts["metrics"], dedupe_centralized=True)
    fairness = _prepare_dashboard_scope(artifacts["fairness"], dedupe_centralized=True)
    communication = _prepare_dashboard_scope(artifacts["communication"], dedupe_centralized=False)
    calibration = _prepare_dashboard_scope(artifacts["calibration"], dedupe_centralized=True)
    shap = _prepare_dashboard_scope(artifacts["shap"], dedupe_centralized=True)
    stability = _prepare_dashboard_scope(artifacts["stability"], dedupe_centralized=True)
    thresholds = _prepare_dashboard_scope(artifacts["thresholds"], dedupe_centralized=True)
    local_explanations = _prepare_dashboard_scope(artifacts["local_explanations"], dedupe_centralized=True)

    metrics_summary = aggregate_with_confidence_intervals(metrics, _present_groups(metrics), _present_metrics(metrics))
    metrics_summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    dashboard_metrics = metrics_summary.copy()
    dashboard_metrics.to_csv(output_dir / "dashboard_metrics.csv", index=False)

    communication_summary = _aggregate_rounds(communication)
    communication_summary.to_csv(output_dir / "dashboard_rounds.csv", index=False)

    calibration_summary = _aggregate_calibration(calibration)
    calibration_summary.to_csv(output_dir / "dashboard_calibration.csv", index=False)

    fairness_summary = _aggregate_fairness(fairness)
    fairness_summary.to_csv(output_dir / "dashboard_fairness.csv", index=False)

    shap_summary = _aggregate_shap(shap)
    shap_summary.to_csv(output_dir / "dashboard_shap.csv", index=False)

    stability_summary = _aggregate_stability(stability)
    stability_summary.to_csv(output_dir / "dashboard_stability.csv", index=False)

    threshold_summary = _aggregate_thresholds(thresholds)
    threshold_summary.to_csv(output_dir / "dashboard_thresholds.csv", index=False)

    local_explanation_summary = _aggregate_local_explanations(local_explanations)
    local_explanation_summary.to_csv(output_dir / "dashboard_local_explanations.csv", index=False)

    dashboard_curves, prediction_rows = _aggregate_prediction_curves_from_files(results_dir)
    if dashboard_curves.empty:
        dashboard_curves = _load_showcase_fallback(output_dir, "dashboard_curves.csv")
    dashboard_curves.to_csv(output_dir / "dashboard_curves.csv", index=False)

    dashboard_confusion = _aggregate_confusion_from_files(results_dir)
    if dashboard_confusion.empty:
        dashboard_confusion = _load_showcase_fallback(output_dir, "dashboard_confusion.csv")
    dashboard_confusion.to_csv(output_dir / "dashboard_confusion.csv", index=False)

    dashboard_showcase = _aggregate_showcase_metrics(dashboard_metrics)
    dashboard_showcase.to_csv(output_dir / "dashboard_showcase_metrics.csv", index=False)

    dashboard_score_ceiling = _aggregate_score_ceiling(
        dashboard_metrics,
        threshold_summary,
        dashboard_confusion,
        stability_summary,
    )
    dashboard_score_ceiling.to_csv(output_dir / "dashboard_score_ceiling.csv", index=False)

    if not local_explanations.empty:
        local_explanations.to_csv(output_dir / "all_local_explanations.csv", index=False)

    communication_final = pd.DataFrame()
    if not communication.empty and "cumulative_communication_bytes" in communication.columns:
        final_rounds = communication.sort_values("round").groupby(_present_groups(communication), dropna=False).tail(1)
        communication_final = aggregate_with_confidence_intervals(
            final_rounds,
            _present_groups(final_rounds),
            ["cumulative_communication_bytes"],
        )
        communication_final.to_csv(output_dir / "communication_summary.csv", index=False)

    report = {
        "runs_found": len(list(results_dir.glob("*/summary.json"))),
        "metric_rows": len(artifacts["metrics"]),
        "summary_rows": len(metrics_summary),
        "prediction_rows": int(prediction_rows),
        "dashboard_files": [
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
        ],
        "output_dir": str(output_dir),
    }
    (output_dir / "aggregation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _write_raw_tables(output_dir: Path, artifacts: dict[str, pd.DataFrame]) -> None:
    mapping = {
        "metrics": "all_metrics.csv",
        "fairness": "all_fairness.csv",
        "communication": "all_communication.csv",
        "calibration": "all_calibration_bins.csv",
        "shap": "all_shap_summary.csv",
        "stability": "all_stability.csv",
        "thresholds": "all_threshold_sweeps.csv",
    }
    for key, filename in mapping.items():
        artifacts[key].to_csv(output_dir / filename, index=False)


def _concat_artifact(results_dir: Path, filename: str) -> pd.DataFrame:
    frames = []
    for path in sorted(results_dir.glob(f"*/{filename}")):
        frame = pd.read_csv(path, low_memory=False)
        frame.insert(0, "run_id", path.parent.name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_showcase_fallback(output_dir: Path, filename: str) -> pd.DataFrame:
    fallback = output_dir.parent / "showcase_summary" / filename
    if not fallback.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(fallback, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _prepare_dashboard_scope(frame: pd.DataFrame, dedupe_centralized: bool) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    if dedupe_centralized:
        output = _dedupe_centralized(output)
    if "run_type" in output.columns:
        mask = output["run_type"].astype(str) == "centralized"
        if "clients" in output.columns:
            output["clients"] = output["clients"].astype(str)
            output.loc[mask, "clients"] = "shared"
        if "partition" in output.columns:
            output["partition"] = output["partition"].astype(str)
            output.loc[mask, "partition"] = "shared"
        if "alpha" in output.columns:
            output["alpha"] = output["alpha"].astype(str)
            output.loc[mask, "alpha"] = "shared"
    return output


def _dedupe_centralized(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "run_type" not in frame.columns:
        return frame
    central_mask = frame["run_type"].astype(str) == "centralized"
    central = frame[central_mask].copy()
    other = frame[~central_mask].copy()
    if central.empty:
        return frame
    subset = [
        column
        for column in [
            "dataset",
            "experiment_track",
            "seed",
            "run_type",
            "algorithm",
            "model",
            "calibration",
            "threshold_strategy",
            "decision_threshold",
            "client_id",
            "group_feature",
            "group_value",
            "bin",
            "lower",
            "upper",
            "round",
            "rank",
            "feature",
            "threshold",
            "split",
            "row_id",
        ]
        if column in central.columns
    ]
    central = central.drop_duplicates(subset=subset or None).copy()
    return pd.concat([central, other], ignore_index=True)


def _aggregate_rounds(communication: pd.DataFrame) -> pd.DataFrame:
    if communication.empty:
        return pd.DataFrame()
    groups = [column for column in GROUPS if column in communication.columns] + ["round"]
    metrics = [
        column
        for column in [
            "global_eval_auc",
            "global_eval_f1",
            "global_eval_ece",
            "cumulative_communication_bytes",
            "round_communication_bytes",
            "params_bytes",
        ]
        if column in communication.columns
    ]
    return aggregate_with_confidence_intervals(communication, groups, metrics)


def _aggregate_calibration(calibration: pd.DataFrame) -> pd.DataFrame:
    if calibration.empty:
        return pd.DataFrame()
    groups = [column for column in GROUPS if column in calibration.columns] + [
        column for column in ["bin", "lower", "upper"] if column in calibration.columns
    ]
    metrics = [column for column in ["count", "avg_predicted_risk", "observed_event_rate"] if column in calibration.columns]
    return aggregate_with_confidence_intervals(calibration, groups, metrics)


def _aggregate_fairness(fairness: pd.DataFrame) -> pd.DataFrame:
    if fairness.empty:
        return pd.DataFrame()
    groups = [column for column in GROUPS if column in fairness.columns] + [
        column for column in ["group_feature", "group_value"] if column in fairness.columns
    ]
    metrics = [
        column
        for column in [
            "count",
            "selection_rate",
            "true_positive_rate",
            "false_positive_rate",
            "demographic_parity_difference",
            "equalized_odds_difference",
        ]
        if column in fairness.columns
    ]
    return aggregate_with_confidence_intervals(fairness, groups, metrics)


def _aggregate_shap(shap: pd.DataFrame) -> pd.DataFrame:
    if shap.empty:
        return pd.DataFrame()
    groups = [column for column in GROUPS if column in shap.columns] + [
        column for column in ["client_id", "feature", "method"] if column in shap.columns
    ]
    metrics = [column for column in ["rank", "mean_abs_shap"] if column in shap.columns]
    summary = aggregate_with_confidence_intervals(shap, groups, metrics)
    if "rank_mean" in summary.columns:
        sort_cols = [column for column in ["dataset", "experiment_track", "run_type", "model", "calibration", "rank_mean"] if column in summary.columns]
        summary = summary.sort_values(sort_cols, kind="stable")
    return summary


def _aggregate_local_explanations(local_explanations: pd.DataFrame) -> pd.DataFrame:
    if local_explanations.empty:
        return pd.DataFrame()
    groups = [column for column in GROUPS if column in local_explanations.columns] + [
        column for column in ["client_id", "feature", "method"] if column in local_explanations.columns
    ]
    metrics = [column for column in ["rank", "contribution"] if column in local_explanations.columns]
    summary = aggregate_with_confidence_intervals(local_explanations, groups, metrics)
    if "rank_mean" in summary.columns:
        sort_cols = [column for column in ["dataset", "experiment_track", "run_type", "model", "calibration", "rank_mean"] if column in summary.columns]
        summary = summary.sort_values(sort_cols, kind="stable")
    return summary


def _aggregate_stability(stability: pd.DataFrame) -> pd.DataFrame:
    if stability.empty:
        return pd.DataFrame()
    groups = [column for column in GROUPS if column in stability.columns] + [
        column for column in ["round", "client_left", "client_right", "top_k"] if column in stability.columns
    ]
    metrics = [
        column
        for column in [
            "spearman_top_feature_stability",
            "spearman_rank_correlation",
            "top_k_overlap",
        ]
        if column in stability.columns
    ]
    return aggregate_with_confidence_intervals(stability, groups, metrics)


def _aggregate_thresholds(thresholds: pd.DataFrame) -> pd.DataFrame:
    if thresholds.empty:
        return pd.DataFrame()
    thresholds = _add_derived_threshold_metrics(thresholds)
    groups = [column for column in GROUPS if column in thresholds.columns] + [
        column for column in ["threshold", "group_feature", "group_value"] if column in thresholds.columns
    ]
    metrics = [
        column
        for column in [
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "balanced_accuracy",
            "f1",
            "pr_auc",
            "selection_rate",
            "demographic_parity_difference",
            "equalized_odds_difference",
            "selected_threshold",
            "is_selected_threshold",
        ]
        if column in thresholds.columns
    ]
    return aggregate_with_confidence_intervals(thresholds, groups, metrics)


def _add_derived_threshold_metrics(thresholds: pd.DataFrame) -> pd.DataFrame:
    output = thresholds.copy()
    required = {"accuracy", "precision", "recall", "selection_rate"}
    if not required.issubset(output.columns):
        return output

    accuracy = pd.to_numeric(output["accuracy"], errors="coerce")
    precision = pd.to_numeric(output["precision"], errors="coerce")
    recall = pd.to_numeric(output["recall"], errors="coerce")
    selection = pd.to_numeric(output["selection_rate"], errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        prevalence_estimate = (precision * selection) / recall
    prevalence_estimate = prevalence_estimate.where(np.isfinite(prevalence_estimate))
    prevalence_estimate = prevalence_estimate.clip(lower=1e-6, upper=1.0 - 1e-6)

    prevalence_groups = [
        column
        for column in [
            "run_id",
            "dataset",
            "seed",
            "clients",
            "partition",
            "alpha",
            "run_type",
            "algorithm",
            "model",
            "calibration",
            "threshold_strategy",
            "group_feature",
            "group_value",
        ]
        if column in output.columns
    ]
    output["_prevalence_estimate"] = prevalence_estimate
    if prevalence_groups:
        prevalence = output.groupby(prevalence_groups, dropna=False)["_prevalence_estimate"].transform("median")
    else:
        prevalence = output["_prevalence_estimate"].median()
    prevalence = pd.to_numeric(prevalence, errors="coerce").clip(lower=1e-6, upper=1.0 - 1e-6)

    with np.errstate(divide="ignore", invalid="ignore"):
        specificity = (accuracy - (recall * prevalence)) / (1.0 - prevalence)
    output["specificity"] = pd.Series(specificity, index=output.index).replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0)
    output["balanced_accuracy"] = ((recall + output["specificity"]) / 2.0).clip(0.0, 1.0)
    return output.drop(columns=["_prevalence_estimate"])


def _aggregate_prediction_curves(test_predictions: pd.DataFrame) -> pd.DataFrame:
    if test_predictions.empty:
        return pd.DataFrame()
    per_run = []
    scenario_columns = [column for column in GROUPS if column in test_predictions.columns]
    run_group_columns = scenario_columns + ["seed", "run_id"]
    for _, group in test_predictions.groupby(run_group_columns, dropna=False):
        per_run.append(_curve_rows_for_run(group, scenario_columns))
    curve_rows = pd.concat(per_run, ignore_index=True) if per_run else pd.DataFrame()
    if curve_rows.empty:
        return curve_rows
    groups = scenario_columns + [column for column in ["curve_type", "point_id", "class_label"] if column in curve_rows.columns]
    metrics = [column for column in ["curve_x", "curve_y"] if column in curve_rows.columns]
    summary = aggregate_with_confidence_intervals(curve_rows, groups, metrics)
    return summary


def _aggregate_prediction_curves_from_files(results_dir: Path) -> tuple[pd.DataFrame, int]:
    curve_frames = []
    total_prediction_rows = 0
    for frame in _iter_prediction_frames(results_dir, needs_predicted=False):
        total_prediction_rows += len(frame)
        if frame.empty:
            continue
        scenario_columns = [column for column in GROUPS if column in frame.columns]
        run_group_columns = scenario_columns + [column for column in ["seed", "run_id"] if column in frame.columns]
        for _, group in frame.groupby(run_group_columns, dropna=False):
            rows = _curve_rows_for_run(group, scenario_columns)
            if rows.empty:
                continue
            if "seed" in group.columns:
                rows["seed"] = group["seed"].iloc[0]
            if "run_id" in group.columns:
                rows["run_id"] = group["run_id"].iloc[0]
            curve_frames.append(rows)
    curve_rows = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    if curve_rows.empty:
        return curve_rows, total_prediction_rows
    curve_rows = _normalise_prediction_summary_scope(curve_rows)
    curve_rows = _dedupe_centralized_summary_rows(curve_rows, extra=["curve_type", "point_id", "class_label"])
    groups = [column for column in GROUPS if column in curve_rows.columns] + [
        column for column in ["curve_type", "point_id", "class_label"] if column in curve_rows.columns
    ]
    return aggregate_with_confidence_intervals(curve_rows, groups, ["curve_x", "curve_y"]), total_prediction_rows


def _curve_rows_for_run(group: pd.DataFrame, scenario_columns: list[str]) -> pd.DataFrame:
    context = {column: group[column].iloc[0] for column in scenario_columns}
    y_true = pd.to_numeric(group["true_label"], errors="coerce").astype(int).to_numpy()
    probabilities = pd.to_numeric(group["calibrated_probability"], errors="coerce").astype(float).to_numpy()
    rows = []

    if len(np.unique(y_true)) >= 2:
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        fpr_grid = np.linspace(0.0, 1.0, 51)
        tpr_interp = np.interp(fpr_grid, fpr, tpr)
        rows.extend(
            {
                **context,
                "curve_type": "roc",
                "point_id": point_id,
                "curve_x": float(x_value),
                "curve_y": float(y_value),
            }
            for point_id, (x_value, y_value) in enumerate(zip(fpr_grid, tpr_interp), start=1)
        )

        precision, recall, _ = precision_recall_curve(y_true, probabilities)
        order = np.argsort(recall)
        recall_sorted = recall[order]
        precision_sorted = precision[order]
        recall_grid = np.linspace(0.0, 1.0, 51)
        precision_interp = np.interp(recall_grid, recall_sorted, precision_sorted)
        rows.extend(
            {
                **context,
                "curve_type": "pr",
                "point_id": point_id,
                "curve_x": float(x_value),
                "curve_y": float(y_value),
            }
            for point_id, (x_value, y_value) in enumerate(zip(recall_grid, precision_interp), start=1)
        )

    bins = np.linspace(0.0, 1.0, 21)
    bin_ids = np.digitize(probabilities, bins[1:-1], right=False) + 1
    for class_value in [0, 1]:
        class_mask = y_true == class_value
        class_count = max(int(class_mask.sum()), 1)
        for bin_id in range(1, len(bins)):
            lower = bins[bin_id - 1]
            upper = bins[bin_id]
            mask = class_mask & (bin_ids == bin_id)
            rows.append(
                {
                    **context,
                    "curve_type": "score_hist",
                    "class_label": f"class_{class_value}",
                    "point_id": bin_id,
                    "curve_x": float((lower + upper) / 2.0),
                    "curve_y": float(mask.sum() / class_count),
                }
            )
    return pd.DataFrame(rows)


def _aggregate_confusion(test_predictions: pd.DataFrame) -> pd.DataFrame:
    if test_predictions.empty:
        return pd.DataFrame()
    scenario_columns = [column for column in GROUPS if column in test_predictions.columns]
    run_group_columns = scenario_columns + ["seed", "run_id"]
    rows = []
    for _, group in test_predictions.groupby(run_group_columns, dropna=False):
        context = {column: group[column].iloc[0] for column in scenario_columns}
        y_true = pd.to_numeric(group["true_label"], errors="coerce").astype(int).to_numpy()
        y_pred = pd.to_numeric(group["predicted_label"], errors="coerce").astype(int).to_numpy()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.extend(
            {
                **context,
                "cell": cell,
                "count": float(value),
            }
            for cell, value in {"TN": tn, "FP": fp, "FN": fn, "TP": tp}.items()
        )
    confusion = pd.DataFrame(rows)
    groups = scenario_columns + ["cell"]
    return aggregate_with_confidence_intervals(confusion, groups, ["count"])


def _aggregate_confusion_from_files(results_dir: Path) -> pd.DataFrame:
    rows = []
    for frame in _iter_prediction_frames(results_dir, needs_predicted=True):
        if frame.empty:
            continue
        scenario_columns = [column for column in GROUPS if column in frame.columns]
        run_group_columns = scenario_columns + [column for column in ["seed", "run_id"] if column in frame.columns]
        for _, group in frame.groupby(run_group_columns, dropna=False):
            context = {column: group[column].iloc[0] for column in scenario_columns}
            if "seed" in group.columns:
                context["seed"] = group["seed"].iloc[0]
            if "run_id" in group.columns:
                context["run_id"] = group["run_id"].iloc[0]
            y_true = pd.to_numeric(group["true_label"], errors="coerce").astype(int).to_numpy()
            y_pred = pd.to_numeric(group["predicted_label"], errors="coerce").astype(int).to_numpy()
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            rows.extend(
                {
                    **context,
                    "cell": cell,
                    "count": float(value),
                }
                for cell, value in {"TN": tn, "FP": fp, "FN": fn, "TP": tp}.items()
            )
    confusion = pd.DataFrame(rows)
    if confusion.empty:
        return confusion
    confusion = _normalise_prediction_summary_scope(confusion)
    confusion = _dedupe_centralized_summary_rows(confusion, extra=["cell"])
    groups = [column for column in GROUPS if column in confusion.columns] + ["cell"]
    return aggregate_with_confidence_intervals(confusion, groups, ["count"])


def _iter_prediction_frames(results_dir: Path, needs_predicted: bool):
    base_columns = GROUPS + ["seed", "true_label", "calibrated_probability"]
    if needs_predicted:
        base_columns.append("predicted_label")
    for path in sorted(results_dir.glob("*/test_predictions.csv")):
        header = pd.read_csv(path, nrows=0).columns
        usecols = [column for column in base_columns if column in header]
        if "true_label" not in usecols or "calibrated_probability" not in usecols:
            continue
        frame = pd.read_csv(path, usecols=usecols, low_memory=False)
        frame.insert(0, "run_id", path.parent.name)
        yield frame


def _normalise_prediction_summary_scope(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "run_type" not in output.columns:
        return output
    central_mask = output["run_type"].astype(str) == "centralized"
    for column in ["clients", "partition", "alpha"]:
        if column in output.columns:
            output[column] = output[column].astype(str)
            output.loc[central_mask, column] = "shared"
    return output


def _dedupe_centralized_summary_rows(frame: pd.DataFrame, extra: list[str]) -> pd.DataFrame:
    if frame.empty or "run_type" not in frame.columns:
        return frame
    central = frame[frame["run_type"].astype(str) == "centralized"].copy()
    other = frame[frame["run_type"].astype(str) != "centralized"].copy()
    if central.empty:
        return frame
    subset = [
        column
        for column in [
            "dataset",
            "experiment_track",
            "seed",
            "run_type",
            "algorithm",
            "model",
            "calibration",
            "threshold_strategy",
            *extra,
        ]
        if column in central.columns
    ]
    central = central.drop_duplicates(subset=subset)
    return pd.concat([central, other], ignore_index=True)


def _aggregate_showcase_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    showcase = metrics.copy()
    showcase = showcase[
        showcase["run_type"].astype(str).isin(SHOWCASE_RUN_TYPES)
        & showcase["threshold_strategy"].astype(str).eq("calib_f1_optimal")
    ].copy()
    if showcase.empty:
        return showcase

    rows: list[pd.Series] = []
    for dataset, dataset_frame in showcase.groupby("dataset", dropna=False):
        central = dataset_frame[dataset_frame["run_type"].astype(str) == "centralized"]
        fed = dataset_frame[dataset_frame["run_type"].astype(str) == "federated"]
        if not central.empty:
            row = _select_showcase_row(central, mode="decision").copy()
            row["showcase_role"] = "best_centralized_decision"
            rows.append(row)
        if not fed.empty:
            row = _select_showcase_row(fed, mode="decision").copy()
            row["showcase_role"] = "best_federated_decision"
            rows.append(row)
        for run_type, run_frame in dataset_frame.groupby("run_type", dropna=False):
            row = _select_showcase_row(run_frame, mode="probability_quality").copy()
            row["showcase_role"] = "best_probability_quality"
            row["showcase_run_type"] = run_type
            rows.append(row)
    output = pd.DataFrame(rows).reset_index(drop=True)
    sort_cols = [column for column in ["dataset", "showcase_role", "showcase_run_type"] if column in output.columns]
    return output.sort_values(sort_cols, kind="stable") if sort_cols else output


def _aggregate_score_ceiling(
    metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
    confusion: pd.DataFrame,
    stability: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    datasets = sorted(
        {
            str(value)
            for frame in [metrics, thresholds, confusion, stability]
            if not frame.empty and "dataset" in frame.columns
            for value in frame["dataset"].dropna().astype(str).unique()
        }
    )
    for dataset in datasets:
        metric_rows = metrics[metrics["dataset"].astype(str) == dataset].copy() if not metrics.empty else pd.DataFrame()
        threshold_rows = thresholds[thresholds["dataset"].astype(str) == dataset].copy() if not thresholds.empty else pd.DataFrame()
        confusion_rows = confusion[confusion["dataset"].astype(str) == dataset].copy() if not confusion.empty else pd.DataFrame()
        stability_rows = stability[stability["dataset"].astype(str) == dataset].copy() if not stability.empty else pd.DataFrame()

        threshold_rows = _dedupe_threshold_aggregate_rows(threshold_rows)
        for score_card, metric, metric_label, caveat, interpretation in [
            (
                "Best Overall Accuracy",
                "accuracy",
                "Accuracy",
                "Overall accuracy is useful, but can look high on imbalanced data because the majority class can dominate.",
                "This is the highest legitimate thresholded accuracy found in the existing sweep.",
            ),
            (
                "Best F1 Operating Point",
                "f1",
                "F1",
                "F1 balances precision and recall, so it is usually more informative than raw accuracy under imbalance.",
                "This is the strongest decision threshold for catching positives without ignoring false alarms.",
            ),
            (
                "Best Balanced Accuracy",
                "balanced_accuracy",
                "Balanced Accuracy",
                "Balanced accuracy averages sensitivity and specificity so the negative class cannot dominate the score.",
                "This is a fairer accuracy-style score for imbalanced diabetes prediction.",
            ),
            (
                "High Recall View",
                "recall",
                "Recall",
                "Near-99 recall means most diabetic cases are caught, but it can create many false positives.",
                "Use this as a safety-oriented operating view, not as overall accuracy.",
            ),
            (
                "High Specificity View",
                "specificity",
                "Specificity",
                "High specificity means many non-diabetic cases are correctly left negative; it does not mean diabetic recall is high.",
                "Use this as a low-false-alarm operating view, not as overall accuracy.",
            ),
            (
                "High Precision View",
                "precision",
                "Precision",
                "High precision can occur at strict thresholds that predict fewer positives.",
                "Use this to understand how confident positive predictions can be at a chosen threshold.",
            ),
        ]:
            row = _best_summary_row(threshold_rows, f"{metric}_mean", higher=True)
            if row is not None:
                rows.append(
                    _score_ceiling_row(
                        row,
                        score_card=score_card,
                        metric=metric,
                        metric_label=metric_label,
                        value_column=f"{metric}_mean",
                        source_table="dashboard_thresholds.csv",
                        caveat=caveat,
                        interpretation=interpretation,
                    )
                )

        for score_card, metric, metric_label, higher, caveat, interpretation in [
            (
                "Best ROC AUC",
                "roc_auc",
                "ROC AUC",
                True,
                "ROC AUC is threshold-independent ranking quality, not percent-correct accuracy.",
                "This is the best ranking result across the saved current-dataset runs.",
            ),
            (
                "Best PR AUC",
                "pr_auc",
                "PR AUC",
                True,
                "PR AUC is usually harder than ROC AUC on imbalanced diabetes data.",
                "This is the best precision-recall ranking result across the saved runs.",
            ),
            (
                "Best Calibration Error",
                "ece",
                "ECE",
                False,
                "For ECE, lower is better. A very small value means probability estimates match observed rates well.",
                "This is a probability-quality score, not a classification accuracy score.",
            ),
            (
                "Best Brier Score",
                "brier",
                "Brier",
                False,
                "For Brier score, lower is better because it measures probability error.",
                "This shows the best probability-estimation slice among the current runs.",
            ),
        ]:
            row = _best_summary_row(metric_rows, f"{metric}_mean", higher=higher)
            if row is not None:
                rows.append(
                    _score_ceiling_row(
                        row,
                        score_card=score_card,
                        metric=metric,
                        metric_label=metric_label,
                        value_column=f"{metric}_mean",
                        source_table="dashboard_metrics.csv",
                        caveat=caveat,
                        interpretation=interpretation,
                        higher_is_better=higher,
                    )
                )

        selected_specificity = _selected_confusion_specificity(confusion_rows, metric_rows)
        row = _best_summary_row(selected_specificity, "specificity_mean", higher=True)
        if row is not None:
            rows.append(
                _score_ceiling_row(
                    row,
                    score_card="Best Selected-Threshold Specificity",
                    metric="specificity",
                    metric_label="Specificity",
                    value_column="specificity_mean",
                    source_table="dashboard_confusion.csv",
                    caveat="This specificity is computed from the saved selected-threshold confusion matrix.",
                    interpretation="It confirms whether the deployed threshold avoids false positives for non-diabetic cases.",
                )
            )

        for score_card, metric, metric_label, caveat, interpretation in [
            (
                "Explanation Stability",
                "spearman_top_feature_stability",
                "Explanation Stability",
                "Near-99 stability means feature rankings are consistent; it is not prediction accuracy.",
                "This supports the interpretability part of the trustworthiness claim.",
            ),
            (
                "Cross-Client Feature Overlap",
                "top_k_overlap",
                "Top-k Overlap",
                "Near-99 overlap means clients emphasize similar top features; it is not prediction accuracy.",
                "This shows whether explanations agree across federated clients.",
            ),
        ]:
            row = _best_summary_row(stability_rows, f"{metric}_mean", higher=True)
            if row is not None:
                rows.append(
                    _score_ceiling_row(
                        row,
                        score_card=score_card,
                        metric=metric,
                        metric_label=metric_label,
                        value_column=f"{metric}_mean",
                        source_table="dashboard_stability.csv",
                        caveat=caveat,
                        interpretation=interpretation,
                    )
                )
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["is_near_99"] = pd.to_numeric(output["value"], errors="coerce").ge(0.99)
    sort_columns = [column for column in ["dataset", "score_card"] if column in output.columns]
    return output.sort_values(sort_columns, kind="stable")


def _dedupe_threshold_aggregate_rows(thresholds: pd.DataFrame) -> pd.DataFrame:
    if thresholds.empty:
        return thresholds
    output = thresholds.copy()
    if "group_value" in output.columns:
        output = output[output["group_value"].astype(str) == "aggregate"].copy()
    subset = [
        column
        for column in [
            *GROUPS,
            "threshold",
        ]
        if column in output.columns
    ]
    return output.drop_duplicates(subset=subset)


def _best_summary_row(frame: pd.DataFrame, value_column: str, higher: bool) -> pd.Series | None:
    if frame.empty or value_column not in frame.columns:
        return None
    working = frame.copy()
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce")
    working = working.dropna(subset=[value_column])
    if working.empty:
        return None
    tie_columns = [column for column in ["f1_mean", "roc_auc_mean", "pr_auc_mean", "ece_mean"] if column in working.columns and column != value_column]
    ascending = [not higher]
    for column in tie_columns:
        ascending.append(column == "ece_mean")
    return working.sort_values([value_column, *tie_columns], ascending=ascending).iloc[0]


def _score_ceiling_row(
    row: pd.Series,
    score_card: str,
    metric: str,
    metric_label: str,
    value_column: str,
    source_table: str,
    caveat: str,
    interpretation: str,
    higher_is_better: bool = True,
) -> dict[str, object]:
    result: dict[str, object] = {
        "score_card": score_card,
        "metric": metric,
        "metric_label": metric_label,
        "value": row.get(value_column),
        "ci95_low": row.get(value_column.replace("_mean", "_ci95_low")),
        "ci95_high": row.get(value_column.replace("_mean", "_ci95_high")),
        "n": row.get(value_column.replace("_mean", "_n")),
        "higher_is_better": higher_is_better,
        "source_table": source_table,
        "caveat": caveat,
        "interpretation": interpretation,
        "threshold": row.get("threshold", row.get("decision_threshold_mean", row.get("selected_threshold_mean"))),
    }
    for column in GROUPS:
        if column in row.index:
            result[column] = row.get(column)
    for column in ["showcase_role", "showcase_run_type"]:
        if column in row.index:
            result[column] = row.get(column)
    return result


def _selected_confusion_specificity(confusion: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    if confusion.empty:
        return pd.DataFrame()
    groups = [column for column in GROUPS if column in confusion.columns]
    pivot = confusion.pivot_table(index=groups, columns="cell", values="count_mean", aggfunc="sum").reset_index()
    if not {"TN", "FP", "FN", "TP"}.issubset(pivot.columns):
        return pd.DataFrame()
    negatives = pivot["TN"] + pivot["FP"]
    positives = pivot["TP"] + pivot["FN"]
    pivot["specificity_mean"] = np.where(negatives > 0, pivot["TN"] / negatives, np.nan)
    pivot["recall_mean"] = np.where(positives > 0, pivot["TP"] / positives, np.nan)
    pivot["balanced_accuracy_mean"] = (pivot["specificity_mean"] + pivot["recall_mean"]) / 2.0
    if not metrics.empty:
        merge_cols = [column for column in groups if column in metrics.columns]
        keep = merge_cols + [
            column
            for column in ["f1_mean", "roc_auc_mean", "pr_auc_mean", "ece_mean", "decision_threshold_mean"]
            if column in metrics.columns
        ]
        pivot = pivot.merge(metrics[keep].drop_duplicates(subset=merge_cols), on=merge_cols, how="left")
    return pivot


def _select_showcase_row(frame: pd.DataFrame, mode: str) -> pd.Series:
    working = frame.copy()
    if mode == "probability_quality":
        best_auc = pd.to_numeric(working["roc_auc_mean"], errors="coerce").max()
        working = working[pd.to_numeric(working["roc_auc_mean"], errors="coerce") >= best_auc - 0.01].copy()
        return working.sort_values(["ece_mean", "brier_mean", "roc_auc_mean"], ascending=[True, True, False]).iloc[0]
    return working.sort_values(["f1_mean", "roc_auc_mean", "ece_mean"], ascending=[False, False, True]).iloc[0]


def _present_groups(frame: pd.DataFrame) -> list[str]:
    return [column for column in GROUPS if column in frame.columns]


def _present_metrics(frame: pd.DataFrame) -> list[str]:
    return [column for column in METRICS if column in frame.columns]


if __name__ == "__main__":
    main()
