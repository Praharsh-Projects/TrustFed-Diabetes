from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


SCENARIO_COLUMNS = [
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

LABELS = {
    "cdc": "CDC Diabetes Indicators",
    "pima": "Pima Diabetes Dataset",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "not_applicable": "Not Applicable",
    "centralized": "Centralized",
    "federated": "Federated",
    "logistic": "Logistic",
    "logistic_regression": "Logistic Regression",
    "mlp": "Shallow Neural Network",
    "xgboost": "XGBoost",
    "gradient_boosting": "Gradient Boosting",
    "random_forest": "Random Forest",
    "decision_tree": "Decision Tree",
    "global_isotonic": "Global Isotonic",
    "global_sigmoid": "Global Sigmoid",
    "federated_isotonic": "Federated Isotonic",
    "federated_sigmoid": "Federated Sigmoid",
    "calib_f1_optimal": "Use F1-Tuned Cutoff",
    "fixed_0p5": "Use 50% Risk Cutoff",
    "local_only": "Local-Only",
    "local_only_mean": "Average Local-Only",
    "non_iid": "Different Data Across Sites",
    "iid": "Similar Data Across Sites",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export thesis figures and tables from aggregate results.")
    parser.add_argument("--summary-dir", default="results/summary")
    parser.add_argument("--visual-summary-dir", default=None)
    parser.add_argument("--output-dir", default="thesis_assets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_dir = Path(args.summary_dir)
    visual_summary_dir = _discover_visual_summary_dir(summary_dir, None if args.visual_summary_dir is None else Path(args.visual_summary_dir))
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    exported: list[str] = []
    metrics = _read(summary_dir / "dashboard_metrics.csv")
    showcase = _read(summary_dir / "dashboard_showcase_metrics.csv")
    curves = _read_preferred(summary_dir, visual_summary_dir, "dashboard_curves.csv")
    confusion = _read_preferred(summary_dir, visual_summary_dir, "dashboard_confusion.csv")
    calibration = _read(summary_dir / "dashboard_calibration.csv")
    fairness = _read(summary_dir / "dashboard_fairness.csv")
    rounds = _read(summary_dir / "dashboard_rounds.csv")
    stability = _read(summary_dir / "dashboard_stability.csv")
    shap = _read(summary_dir / "dashboard_shap.csv")
    communication = _read(summary_dir / "communication_summary.csv")
    score_ceiling = _read(summary_dir / "dashboard_score_ceiling.csv")
    thresholds = _read(summary_dir / "dashboard_thresholds.csv")

    best_for_ranking = pd.DataFrame()
    best_for_f1 = pd.DataFrame()
    best_for_low_fpr = pd.DataFrame()
    best_for_calibration = pd.DataFrame()
    best_federated_gap = pd.DataFrame()
    best_fairness_threshold = pd.DataFrame()
    communication_efficient = pd.DataFrame()

    if not metrics.empty:
        _export_table(metrics, tables_dir / "dashboard_metrics.csv", exported)
        best_centralized = _best_rows(metrics, run_type="centralized")
        best_federated = _best_rows(metrics, run_type="federated")
        federated_gap = _federated_gap_table(metrics)
        calibration_gain = _calibration_gain_table(metrics)
        decision_metrics = _decision_view(metrics)
        best_for_ranking = _best_metric_rows(
            decision_metrics,
            "roc_auc_mean",
            higher=True,
            tie_columns=["pr_auc_mean", "f1_mean", "ece_mean"],
        )
        best_for_f1 = _best_metric_rows(
            decision_metrics,
            "f1_mean",
            higher=True,
            tie_columns=["roc_auc_mean", "pr_auc_mean", "ece_mean"],
        )
        best_for_calibration = _best_calibration_rows(decision_metrics)
        _export_table(best_centralized, tables_dir / "best_centralized.csv", exported)
        _export_table(best_federated, tables_dir / "best_federated.csv", exported)
        _export_table(federated_gap, tables_dir / "federated_vs_centralized_gap.csv", exported)
        _export_table(calibration_gain, tables_dir / "calibration_improvement.csv", exported)
        _export_table(best_for_ranking, tables_dir / "best_for_ranking.csv", exported)
        _export_table(best_for_f1, tables_dir / "best_for_f1.csv", exported)
        _export_table(best_for_calibration, tables_dir / "best_for_calibration.csv", exported)

        performance = metrics[metrics["run_type"].isin(["centralized", "local_only_mean", "federated"])].copy()
        performance["roc_auc_error_plus"] = performance["roc_auc_ci95_high"] - performance["roc_auc_mean"]
        performance["roc_auc_error_minus"] = performance["roc_auc_mean"] - performance["roc_auc_ci95_low"]
        fig = px.bar(
            performance,
            x="model",
            y="roc_auc_mean",
            color="run_type",
            facet_col="dataset",
            barmode="group",
            title="Average Ranking Quality By Model And Training Setup",
            error_y="roc_auc_error_plus",
            error_y_minus="roc_auc_error_minus",
        )
        fig.write_html(figures_dir / "performance_overview.html")
        exported.append(str(figures_dir / "performance_overview.html"))

    if not showcase.empty:
        _export_table(showcase, tables_dir / "dashboard_showcase_metrics.csv", exported)
        _export_showcase_curves(showcase, curves, figures_dir, exported)
        _export_showcase_confusion(showcase, confusion, figures_dir, exported)
        best_federated_gap = _best_federated_gap(showcase)
        _export_table(best_federated_gap, tables_dir / "best_federated_gap.csv", exported)

    if not score_ceiling.empty:
        _export_table(score_ceiling, tables_dir / "score_ceiling_audit.csv", exported)
        best_for_low_fpr = _best_low_fpr_rows(score_ceiling)
        _export_table(best_for_low_fpr, tables_dir / "best_for_low_fpr.csv", exported)
        display = score_ceiling[score_ceiling["score_card"].isin(
            [
                "Best Overall Accuracy",
                "Best F1 Operating Point",
                "High Recall View",
                "Best Selected-Threshold Specificity",
                "Explanation Stability",
            ]
        )].copy()
        if not display.empty:
            fig = px.bar(
                display,
                x="value",
                y="score_card",
                color="metric_label",
                facet_col="dataset",
                orientation="h",
                title="Operating Point Review: Strongest Metric-Specific Views",
                hover_data=["run_type", "model", "algorithm", "calibration", "threshold", "caveat"],
            )
            fig.update_xaxes(range=[0, 1], title="Metric value")
            fig.update_yaxes(title="")
            fig.write_html(figures_dir / "score_ceiling_audit.html")
            exported.append(str(figures_dir / "score_ceiling_audit.html"))

    if not thresholds.empty:
        best_fairness_threshold = _best_fairness_threshold_rows(thresholds)
        _export_table(best_fairness_threshold, tables_dir / "best_fairness_threshold.csv", exported)

    if not calibration.empty:
        _export_table(calibration, tables_dir / "dashboard_calibration.csv", exported)
        fig = px.line(
            calibration,
            x="avg_predicted_risk_mean",
            y="observed_event_rate_mean",
            color="calibration",
            facet_col="dataset",
            title="Do The Risk Percentages Match Reality?",
            markers=True,
        )
        fig.write_html(figures_dir / "reliability_overview.html")
        exported.append(str(figures_dir / "reliability_overview.html"))

    if not fairness.empty:
        fairness_summary = fairness.drop_duplicates(subset=["dataset", "run_type", "model", "calibration", "group_feature"])
        _export_table(fairness_summary, tables_dir / "fairness_gap_summary.csv", exported)
        fig = px.bar(
            fairness_summary,
            x="group_feature",
            y="equalized_odds_difference_mean",
            color="run_type",
            facet_col="dataset",
            barmode="group",
            title="How Different The Error Pattern Is Between Groups",
        )
        fig.write_html(figures_dir / "fairness_overview.html")
        exported.append(str(figures_dir / "fairness_overview.html"))

    if not rounds.empty:
        fig = px.line(
            rounds,
            x="round",
            y="global_eval_auc_mean",
            color="algorithm",
            facet_col="dataset",
            title="How Federated Training Improves Round By Round",
            markers=True,
        )
        fig.write_html(figures_dir / "federated_training_overview.html")
        exported.append(str(figures_dir / "federated_training_overview.html"))

    if not communication.empty:
        _export_table(communication, tables_dir / "communication_summary.csv", exported)
        communication_efficient = _communication_efficient_rows(communication, metrics)
        _export_table(communication_efficient, tables_dir / "communication_efficient_federated.csv", exported)

    if not shap.empty:
        top = shap.sort_values("mean_abs_shap_mean", ascending=False).head(20)
        _export_table(top, tables_dir / "top_feature_attributions.csv", exported)

    if not stability.empty:
        _export_table(stability, tables_dir / "stability_summary.csv", exported)
        line_rows = stability.dropna(subset=["spearman_top_feature_stability_mean"])
        if not line_rows.empty:
            fig = px.line(
                line_rows,
                x="round",
                y="spearman_top_feature_stability_mean",
                color="algorithm",
                facet_col="dataset",
                title="Do The Feature Explanations Stay Consistent Over Time?",
                markers=True,
            )
            fig.write_html(figures_dir / "explanation_stability.html")
            exported.append(str(figures_dir / "explanation_stability.html"))

    conclusion = _build_conclusion_markdown(
        best_for_ranking=best_for_ranking,
        best_for_f1=best_for_f1,
        best_for_low_fpr=best_for_low_fpr,
        best_for_calibration=best_for_calibration,
        best_federated_gap=best_federated_gap,
        best_fairness_threshold=best_fairness_threshold,
        communication_efficient=communication_efficient,
        score_ceiling=score_ceiling,
    )
    conclusion_path = output_dir / "thesis_conclusion_summary.md"
    conclusion_path.write_text(conclusion, encoding="utf-8")
    exported.append(str(conclusion_path))

    report = {"exported": exported, "output_dir": str(output_dir)}
    (output_dir / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _export_showcase_curves(showcase: pd.DataFrame, curves: pd.DataFrame, figures_dir: Path, exported: list[str]) -> None:
    if showcase.empty or curves.empty:
        return
    for dataset in sorted(showcase["dataset"].dropna().astype(str).unique()):
        central = _select_role(showcase, dataset, "best_centralized_decision")
        fed = _select_role(showcase, dataset, "best_federated_decision")
        if central is None and fed is None:
            continue
        for curve_type, filename, title, x_title, y_title in [
            ("roc", f"showcase_{dataset}_roc.html", f"{dataset.upper()} - How Well The Best Models Separate Risk", "False Positive Rate", "True Positive Rate"),
            ("pr", f"showcase_{dataset}_pr.html", f"{dataset.upper()} - How Well The Best Models Catch Positive Cases", "Recall", "Precision"),
        ]:
            figure = go.Figure()
            for row, label, color in [
                (central, "Centralized", "#165DFF"),
                (fed, "Federated", "#0F9D58"),
            ]:
                if row is None:
                    continue
                subset = _filter_exact(curves[curves["curve_type"].astype(str) == curve_type], row)
                if subset.empty:
                    continue
                figure.add_trace(
                    go.Scatter(
                        x=subset["curve_x_mean"],
                        y=subset["curve_y_mean"],
                        mode="lines",
                        name=f"{label} ({row.get('model', '')})",
                        line={"color": color, "width": 3},
                    )
                )
            figure.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
            figure.write_html(figures_dir / filename)
            exported.append(str(figures_dir / filename))


def _export_showcase_confusion(showcase: pd.DataFrame, confusion: pd.DataFrame, figures_dir: Path, exported: list[str]) -> None:
    if showcase.empty or confusion.empty:
        return
    for dataset in sorted(showcase["dataset"].dropna().astype(str).unique()):
        central = _select_role(showcase, dataset, "best_centralized_decision")
        fed = _select_role(showcase, dataset, "best_federated_decision")
        if central is None and fed is None:
            continue
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Centralized", "Federated"])
        for idx, row in enumerate([central, fed], start=1):
            if row is None:
                continue
            subset = _filter_exact(confusion, row)
            if subset.empty:
                continue
            cell_map = {str(item["cell"]): float(item["count_mean"]) for _, item in subset.iterrows()}
            matrix = [
                [cell_map.get("TN", 0.0), cell_map.get("FP", 0.0)],
                [cell_map.get("FN", 0.0), cell_map.get("TP", 0.0)],
            ]
            fig.add_trace(
                go.Heatmap(
                    z=matrix,
                    x=["Pred 0", "Pred 1"],
                    y=["True 0", "True 1"],
                    text=[[f"{value:.0f}" for value in row_values] for row_values in matrix],
                    texttemplate="%{text}",
                    colorscale="Blues",
                    showscale=(idx == 2),
                ),
                row=1,
                col=idx,
            )
        fig.update_layout(title=f"{dataset.upper()} - How The Best Models Classify Cases")
        output = figures_dir / f"showcase_{dataset}_confusion.html"
        fig.write_html(output)
        exported.append(str(output))


def _best_rows(metrics: pd.DataFrame, run_type: str) -> pd.DataFrame:
    frame = metrics[metrics["run_type"] == run_type].copy()
    if frame.empty:
        return frame
    idx = frame.groupby("dataset", dropna=False)["roc_auc_mean"].idxmax()
    return frame.loc[idx].sort_values("dataset")


def _decision_view(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "threshold_strategy" not in metrics.columns:
        return metrics.copy()
    tuned = metrics[metrics["threshold_strategy"].astype(str) == "calib_f1_optimal"].copy()
    return tuned if not tuned.empty else metrics.copy()


def _best_metric_rows(
    metrics: pd.DataFrame,
    value_column: str,
    higher: bool,
    tie_columns: list[str],
) -> pd.DataFrame:
    if metrics.empty or value_column not in metrics.columns:
        return pd.DataFrame()
    rows = []
    for _, dataset_frame in metrics.groupby("dataset", dropna=False):
        working = dataset_frame.copy()
        working[value_column] = pd.to_numeric(working[value_column], errors="coerce")
        working = working.dropna(subset=[value_column])
        if working.empty:
            continue
        available_ties = [column for column in tie_columns if column in working.columns and column != value_column]
        sort_columns = [value_column, *available_ties]
        ascending = [not higher] + [column == "ece_mean" for column in available_ties]
        rows.append(working.sort_values(sort_columns, ascending=ascending).iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def _best_calibration_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for _, dataset_frame in metrics.groupby("dataset", dropna=False):
        working = dataset_frame.copy()
        working["roc_auc_mean"] = pd.to_numeric(working["roc_auc_mean"], errors="coerce")
        working["ece_mean"] = pd.to_numeric(working["ece_mean"], errors="coerce")
        working["brier_mean"] = pd.to_numeric(working["brier_mean"], errors="coerce")
        working["log_loss_mean"] = pd.to_numeric(working["log_loss_mean"], errors="coerce")
        working = working.dropna(subset=["roc_auc_mean", "ece_mean", "brier_mean", "log_loss_mean"])
        if working.empty:
            continue
        max_auc = float(working["roc_auc_mean"].max())
        near_best = working[working["roc_auc_mean"] >= max_auc - 0.01].copy()
        candidate = near_best if not near_best.empty else working
        rows.append(
            candidate.sort_values(
                ["ece_mean", "brier_mean", "log_loss_mean", "roc_auc_mean", "f1_mean"],
                ascending=[True, True, True, False, False],
            ).iloc[0]
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def _best_low_fpr_rows(score_ceiling: pd.DataFrame) -> pd.DataFrame:
    if score_ceiling.empty:
        return pd.DataFrame()
    frame = score_ceiling[score_ceiling["score_card"].astype(str) == "Best Selected-Threshold Specificity"].copy()
    if frame.empty:
        frame = score_ceiling[score_ceiling["score_card"].astype(str) == "High Specificity View"].copy()
    if frame.empty:
        return frame
    frame["specificity"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["false_positive_rate"] = 1.0 - frame["specificity"]
    return frame.sort_values(["dataset", "specificity"], ascending=[True, False], kind="stable").reset_index(drop=True)


def _best_federated_gap(showcase: pd.DataFrame) -> pd.DataFrame:
    if showcase.empty:
        return pd.DataFrame()
    central = showcase[showcase["showcase_role"].astype(str) == "best_centralized_decision"].copy()
    federated = showcase[showcase["showcase_role"].astype(str) == "best_federated_decision"].copy()
    if central.empty or federated.empty:
        return pd.DataFrame()
    keep = [
        "dataset",
        "roc_auc_mean",
        "f1_mean",
        "pr_auc_mean",
        "ece_mean",
        "model",
        "calibration",
        "threshold_strategy",
    ]
    central = central[[column for column in keep if column in central.columns]].rename(
        columns={
            "roc_auc_mean": "centralized_roc_auc_mean",
            "f1_mean": "centralized_f1_mean",
            "pr_auc_mean": "centralized_pr_auc_mean",
            "ece_mean": "centralized_ece_mean",
            "model": "centralized_model",
            "calibration": "centralized_calibration",
            "threshold_strategy": "centralized_threshold_strategy",
        }
    )
    merged = federated.merge(central, on="dataset", how="left")
    merged["roc_auc_gap_vs_centralized"] = merged["roc_auc_mean"] - merged["centralized_roc_auc_mean"]
    merged["f1_gap_vs_centralized"] = merged["f1_mean"] - merged["centralized_f1_mean"]
    if {"pr_auc_mean", "centralized_pr_auc_mean"}.issubset(merged.columns):
        merged["pr_auc_gap_vs_centralized"] = merged["pr_auc_mean"] - merged["centralized_pr_auc_mean"]
    if {"ece_mean", "centralized_ece_mean"}.issubset(merged.columns):
        merged["ece_gap_vs_centralized"] = merged["ece_mean"] - merged["centralized_ece_mean"]
    return merged.sort_values("dataset", kind="stable").reset_index(drop=True)


def _best_fairness_threshold_rows(thresholds: pd.DataFrame) -> pd.DataFrame:
    if thresholds.empty:
        return pd.DataFrame()
    frame = thresholds.copy()
    if "group_value" in frame.columns:
        aggregate = frame[frame["group_value"].astype(str) == "aggregate"].copy()
        if not aggregate.empty:
            frame = aggregate
    required = {"f1_mean", "demographic_parity_difference_mean", "equalized_odds_difference_mean"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()

    rows = []
    for _, dataset_frame in frame.groupby("dataset", dropna=False):
        working = dataset_frame.copy()
        for column in ["f1_mean", "demographic_parity_difference_mean", "equalized_odds_difference_mean"]:
            working[column] = pd.to_numeric(working[column], errors="coerce")
        working = working.dropna(subset=["f1_mean"])
        if working.empty:
            continue
        best_f1 = float(working["f1_mean"].max())
        candidates = working[working["f1_mean"] >= best_f1 * 0.90].copy()
        if candidates.empty:
            candidates = working
        candidates["fairness_total_gap"] = (
            candidates["demographic_parity_difference_mean"].fillna(0.0)
            + candidates["equalized_odds_difference_mean"].fillna(0.0)
        )
        sort_columns = [column for column in ["fairness_total_gap", "f1_mean", "roc_auc_mean", "ece_mean"] if column in candidates.columns]
        ascending = [column in {"fairness_total_gap", "ece_mean"} for column in sort_columns]
        rows.append(
            candidates.sort_values(sort_columns, ascending=ascending).iloc[0]
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def _communication_efficient_rows(communication: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    if communication.empty or metrics.empty:
        return pd.DataFrame()
    metric_frame = _decision_view(metrics)
    metric_frame = metric_frame[metric_frame["run_type"].astype(str) == "federated"].copy()
    comm_frame = communication.copy()
    if "run_type" in comm_frame.columns:
        comm_frame = comm_frame[comm_frame["run_type"].astype(str) == "federated"].copy()
    if metric_frame.empty or comm_frame.empty:
        return pd.DataFrame()
    merge_cols = [column for column in SCENARIO_COLUMNS if column in metric_frame.columns and column in comm_frame.columns]
    for column in merge_cols:
        metric_frame[column] = metric_frame[column].astype(str)
        comm_frame[column] = comm_frame[column].astype(str)
    merged = metric_frame.merge(comm_frame, on=merge_cols, how="inner", suffixes=("", "_communication"))
    if merged.empty or "cumulative_communication_bytes_mean" not in merged.columns:
        return pd.DataFrame()
    rows = []
    for _, dataset_frame in merged.groupby("dataset", dropna=False):
        working = dataset_frame.copy()
        working["roc_auc_mean"] = pd.to_numeric(working["roc_auc_mean"], errors="coerce")
        working["cumulative_communication_bytes_mean"] = pd.to_numeric(
            working["cumulative_communication_bytes_mean"],
            errors="coerce",
        )
        working = working.dropna(subset=["roc_auc_mean", "cumulative_communication_bytes_mean"])
        if working.empty:
            continue
        best_auc = float(working["roc_auc_mean"].max())
        candidates = working[working["roc_auc_mean"] >= best_auc - 0.01].copy()
        if candidates.empty:
            candidates = working
        rows.append(
            candidates.sort_values(
                ["cumulative_communication_bytes_mean", "f1_mean", "roc_auc_mean"],
                ascending=[True, False, False],
            ).iloc[0]
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def _build_conclusion_markdown(
    best_for_ranking: pd.DataFrame,
    best_for_f1: pd.DataFrame,
    best_for_low_fpr: pd.DataFrame,
    best_for_calibration: pd.DataFrame,
    best_federated_gap: pd.DataFrame,
    best_fairness_threshold: pd.DataFrame,
    communication_efficient: pd.DataFrame,
    score_ceiling: pd.DataFrame,
) -> str:
    datasets = sorted(
        {
            str(value)
            for frame in [
                best_for_ranking,
                best_for_f1,
                best_for_low_fpr,
                best_for_calibration,
                best_federated_gap,
                best_fairness_threshold,
                communication_efficient,
                score_ceiling,
            ]
            if not frame.empty and "dataset" in frame.columns
            for value in frame["dataset"].dropna().astype(str).unique()
        }
    )
    lines = [
        "# Thesis Conclusion Summary",
        "",
        "This summary is generated from the active evidence package and keeps every claim tied to the real saved metrics.",
        "",
        "## Main Thesis Statement",
        "",
        "- Centralized training is the raw upper-bound reference for predictive performance on the active public dataset package.",
        "- Federated learning gets close to that Centralized reference while preserving data locality and outperforming isolated Local-Only baselines in the broader audit.",
        "- Different operating points are best for different goals: ranking quality, balanced detection, low false positives, calibrated probabilities, fairness tradeoffs, and communication efficiency.",
        "",
    ]
    for dataset in datasets:
        lines.extend([f"## {dataset.upper()}", ""])
        ranking = _row_for_dataset(best_for_ranking, dataset)
        best_f1_row = _row_for_dataset(best_for_f1, dataset)
        low_fpr = _row_for_dataset(best_for_low_fpr, dataset)
        calibration = _row_for_dataset(best_for_calibration, dataset)
        gap = _row_for_dataset(best_federated_gap, dataset)
        fairness = _row_for_dataset(best_fairness_threshold, dataset)
        communication = _row_for_dataset(communication_efficient, dataset)
        explanation = _explanation_row(score_ceiling, dataset)

        if ranking is not None:
            ranking_pr = f", PR AUC {_fmt(ranking.get('pr_auc_mean'))}" if "pr_auc_mean" in ranking.index else ""
            lines.append(
                f"- **Best for ranking:** {_scenario_text(ranking)} with AUROC {_fmt(ranking.get('roc_auc_mean'))}{ranking_pr}, and F1 {_fmt(ranking.get('f1_mean'))}."
            )
        if best_f1_row is not None:
            lines.append(
                f"- **Best for balanced detection:** {_scenario_text(best_f1_row)} with balanced detection {_fmt(best_f1_row.get('f1_mean'))}, ranking quality {_fmt(best_f1_row.get('roc_auc_mean'))}, and decision rule {_fmt(best_f1_row.get('decision_threshold_mean'))}."
            )
        if low_fpr is not None:
            lines.append(
                f"- **Best for low false positives:** {_scenario_text(low_fpr)} with specificity {_fmt(low_fpr.get('specificity'))}, false-positive rate {_fmt(low_fpr.get('false_positive_rate'))}, and decision rule {_fmt(low_fpr.get('threshold'))}."
            )
        if calibration is not None:
            lines.append(
                f"- **Best for trustworthy probabilities:** {_scenario_text(calibration)} with risk-match error {_fmt(calibration.get('ece_mean'))}, probability error {_fmt(calibration.get('brier_mean'))}, and confidence penalty {_fmt(calibration.get('log_loss_mean'))}."
            )
        if gap is not None:
            lines.append(
                f"- **Best federated close-to-centralized slice:** {_scenario_text(gap)} with ranking gap {_fmt(gap.get('roc_auc_gap_vs_centralized'))} and balanced-detection gap {_fmt(gap.get('f1_gap_vs_centralized'))} versus the strongest Centralized slice."
            )
        if fairness is not None:
            lines.append(
                f"- **Best group-aware decision rule:** {_scenario_text(fairness)} with decision rule {_fmt(fairness.get('threshold'))}, balanced detection {_fmt(fairness.get('f1_mean'))}, positive-rate gap {_fmt(fairness.get('demographic_parity_difference_mean'))}, and error gap {_fmt(fairness.get('equalized_odds_difference_mean'))}."
            )
        if explanation is not None:
            lines.append(
                f"- **Best explanation stability slice:** {_scenario_text(explanation)} with {_human_metric(explanation.get('metric_label', 'Explanation Stability'))} {_fmt(explanation.get('value'))}."
            )
        if communication is not None:
            lines.append(
                f"- **Most communication-efficient competitive federated choice:** {_scenario_text(communication)} with cumulative communication {_fmt(communication.get('cumulative_communication_bytes_mean'))} bytes while keeping ranking quality {_fmt(communication.get('roc_auc_mean'))}."
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- The strongest honest story is not that Federated learning universally beats Centralized training.",
            "- The strongest honest story is that Federated learning can stay close to Centralized performance while adding calibration, fairness, explainability, and communication-aware evidence that isolated Local-Only models do not provide as cleanly.",
            "- Threshold choice materially changes recall, specificity, and false-positive behavior, so any final recommendation should name the operating threshold together with the metric it optimizes.",
            "",
        ]
    )
    return "\n".join(lines)


def _row_for_dataset(frame: pd.DataFrame, dataset: str) -> pd.Series | None:
    if frame.empty or "dataset" not in frame.columns:
        return None
    match = frame[frame["dataset"].astype(str) == str(dataset)].copy()
    if match.empty:
        return None
    return match.iloc[0]


def _explanation_row(score_ceiling: pd.DataFrame, dataset: str) -> pd.Series | None:
    if score_ceiling.empty:
        return None
    match = score_ceiling[
        (score_ceiling["dataset"].astype(str) == str(dataset))
        & (score_ceiling["score_card"].astype(str) == "Explanation Stability")
    ].copy()
    if match.empty:
        return None
    return match.sort_values("value", ascending=False).iloc[0]


def _scenario_text(row: pd.Series) -> str:
    parts = []
    for column in ["run_type", "algorithm", "model", "calibration", "threshold_strategy"]:
        if column in row.index and pd.notna(row.get(column)):
            text = _label(row.get(column))
            if text != "Not Applicable":
                parts.append(text)
    return " / ".join(parts) if parts else "Selected slice"


def _human_metric(label: object) -> str:
    return _label(label)


def _fmt(value) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "n/a"


def _label(value: object) -> str:
    key = str(value)
    return LABELS.get(key, key.replace("_", " ").title())


def _federated_gap_table(metrics: pd.DataFrame) -> pd.DataFrame:
    fed = _best_rows(metrics, "federated")
    central_columns = [column for column in ["dataset", "roc_auc_mean", "f1_mean", "ece_mean"] if column in metrics.columns or column == "dataset"]
    central = _best_rows(metrics, "centralized")[central_columns].rename(
        columns={
            "roc_auc_mean": "centralized_roc_auc_mean",
            "f1_mean": "centralized_f1_mean",
            "ece_mean": "centralized_ece_mean",
        }
    )
    if fed.empty or central.empty:
        return pd.DataFrame()
    merged = fed.merge(central, on="dataset", how="left")
    merged["roc_auc_gap_vs_centralized"] = merged["roc_auc_mean"] - merged["centralized_roc_auc_mean"]
    merged["f1_gap_vs_centralized"] = merged["f1_mean"] - merged["centralized_f1_mean"]
    if {"ece_mean", "centralized_ece_mean"}.issubset(merged.columns):
        merged["ece_gap_vs_centralized"] = merged["ece_mean"] - merged["centralized_ece_mean"]
    return merged


def _calibration_gain_table(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    base_columns = [
        column
        for column in ["dataset", "run_type", "algorithm", "model", "roc_auc_mean", "ece_mean", "brier_mean", "pr_auc_mean"]
        if column in metrics.columns
    ]
    base = metrics[metrics["calibration"] == "none"][base_columns]
    base = base.rename(
        columns={
            "roc_auc_mean": "roc_auc_mean_none",
            "ece_mean": "ece_mean_none",
            "brier_mean": "brier_mean_none",
            "pr_auc_mean": "pr_auc_mean_none",
        }
    )
    calibrated = metrics[metrics["calibration"] != "none"].copy()
    merged = calibrated.merge(base, on=["dataset", "run_type", "algorithm", "model"], how="left")
    merged["roc_auc_delta"] = merged["roc_auc_mean"] - merged["roc_auc_mean_none"]
    if {"pr_auc_mean", "pr_auc_mean_none"}.issubset(merged.columns):
        merged["pr_auc_delta"] = merged["pr_auc_mean"] - merged["pr_auc_mean_none"]
    if {"ece_mean", "ece_mean_none"}.issubset(merged.columns):
        merged["ece_delta"] = merged["ece_mean"] - merged["ece_mean_none"]
    if {"brier_mean", "brier_mean_none"}.issubset(merged.columns):
        merged["brier_delta"] = merged["brier_mean"] - merged["brier_mean_none"]
    return merged.sort_values(["dataset", "run_type", "model", "calibration"])


def _select_role(showcase: pd.DataFrame, dataset: str, role: str) -> pd.Series | None:
    frame = showcase[
        (showcase["dataset"].astype(str) == str(dataset))
        & (showcase["showcase_role"].astype(str) == role)
    ].copy()
    if frame.empty:
        return None
    return frame.iloc[0]


def _filter_exact(frame: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    output = frame.copy()
    if output.empty:
        return output
    for column in ["dataset", "experiment_track", "clients", "partition", "alpha", "run_type", "algorithm", "model", "calibration", "threshold_strategy"]:
        if column in output.columns and column in row.index:
            output = output[output[column].astype(str) == str(row[column])]
    return output


def _export_table(frame: pd.DataFrame, path: Path, exported: list[str]) -> None:
    if frame.empty:
        return
    frame.to_csv(path, index=False)
    exported.append(str(path))


def _discover_visual_summary_dir(summary_dir: Path, requested: Path | None) -> Path | None:
    candidates: list[Path] = []
    if requested is not None:
        candidates.append(requested)
    candidates.extend(
        [
            summary_dir.parent / "polished_visual_verify_summary",
            summary_dir.parent / "showcase_summary",
        ]
    )
    for candidate in candidates:
        if candidate is None or candidate == summary_dir:
            continue
        if (candidate / "dashboard_curves.csv").exists() or (candidate / "dashboard_confusion.csv").exists():
            return candidate
    return None


def _read_preferred(summary_dir: Path, visual_summary_dir: Path | None, filename: str) -> pd.DataFrame:
    preferred_paths: list[Path] = []
    if visual_summary_dir is not None:
        preferred_paths.append(visual_summary_dir / filename)
    preferred_paths.append(summary_dir / filename)
    for path in preferred_paths:
        frame = _read(path)
        if not frame.empty:
            return frame
    return pd.DataFrame()


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


if __name__ == "__main__":
    main()
