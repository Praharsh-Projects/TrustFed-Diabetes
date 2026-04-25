"""Interactive Dash dashboard for audit and showcase experiment summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DashECharts = None


LABELS = {
    "cdc": "CDC Diabetes Indicators",
    "pima": "Pima Diabetes Dataset",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "not_applicable": "Not Applicable",
    "local_only_mean": "Average Local-Only",
    "local_only": "Local-Only",
    "centralized": "Centralized",
    "federated": "Federated",
    "logistic": "Logistic",
    "global_isotonic": "Global Isotonic",
    "global_sigmoid": "Global Sigmoid",
    "federated_isotonic": "Federated Isotonic",
    "federated_sigmoid": "Federated Sigmoid",
    "fixed_0p5": "Use 50% Risk Cutoff",
    "calib_f1_optimal": "Use F1-Tuned Cutoff",
    "calib_youden_j": "Use Balanced Cutoff",
    "iid": "Similar Data Across Sites",
    "non_iid": "Different Data Across Sites",
    "shared": "Shared Centralized Reference",
    "showcase": "Key Findings",
    "audit": "Full Audit",
    "pr": "PR Curve",
    "roc": "ROC Curve",
    "score_hist": "Score Distribution",
    "roc_auc": "Ranking Quality (AUROC)",
    "pr_auc": "Positive-Case Ranking (PR AUC)",
    "f1": "Balanced Detection Score (F1)",
    "ece": "Risk Match Error (ECE)",
    "brier": "Probability Error (Brier)",
    "balanced_accuracy": "Balanced Accuracy",
    "spearman_top_feature_stability": "Explanation Stability",
    "top_k_overlap": "Cross-Site Feature Agreement",
    "xgboost": "XGBoost",
    "mlp": "Shallow Neural Network",
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "selection_rate": "Positive Prediction Rate",
    "demographic_parity_difference": "Positive Rate Gap",
    "equalized_odds_difference": "Error Gap Between Groups",
}

FILTER_SPECS = [
    ("dataset", "Data Source", False),
    ("run_type", "Run Type", True),
    ("model", "Model Family", True),
    ("algorithm", "Algorithm", True),
    ("calibration", "Probability Adjustment", True),
    ("clients", "Number Of Sites", True),
    ("partition", "Data Split Type", True),
    ("alpha", "Data Difference Level", True),
    ("threshold_strategy", "Decision Rule", True),
]

RUN_TYPE_COLORS = {
    "Centralized": "#165DFF",
    "Federated": "#0F9D58",
    "Local-Only": "#F08C00",
    "Average Local-Only": "#7C4DFF",
}
ACCENT = {
    "primary": "#165DFF",
    "secondary": "#0F9D58",
    "warning": "#C75C00",
    "muted": "#5F6C7B",
    "ink": "#10243E",
    "border": "#D6DEEA",
    "bg": "#F4F7FB",
    "panel": "#FFFFFF",
}


def run_dashboard(
    results_dir: str | Path = "results/summary",
    visual_results_dir: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8050,
) -> None:
    global DashECharts
    try:
        from dash import Dash, Input, Output, dcc, html
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit(
            "Dash is not installed. Install optional dashboard dependencies with: py -m pip install dash"
        ) from exc
    try:
        from dash_echarts import DashECharts as _DashECharts
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit(
            "dash-echarts is not installed. Install optional dashboard dependencies with: py -m pip install dash-echarts==0.0.12.9"
        ) from exc
    DashECharts = _DashECharts

    data = _load_dashboard_data(Path(results_dir), None if visual_results_dir is None else Path(visual_results_dir))
    app = Dash(__name__)
    app.title = "Federated Diabetes Trustworthiness Audit"

    app.layout = html.Div(
        [
            html.Header(
                [
                    html.Div(
                        [
                            html.Div("Thesis-Grade Audit Console", className="eyebrow"),
                            html.H1("Federated Diabetes Trustworthiness Audit"),
                            html.P(
                                "Public-data results for model quality, understandable risk percentages, group checks, feature explanations, and federated communication cost."
                            ),
                        ]
                    )
                ],
                className="header",
            ),
            html.Div(
                [
                    _filter_block("Data Source", _options(data.metrics, "dataset"), "dataset", default=_default_value(data.metrics, "dataset", "cdc")),
                    _filter_block("Run Type", _options(data.metrics, "run_type", add_all=True), "run_type", default="__all__"),
                    _filter_block("Model Family", _options(data.metrics, "model", add_all=True), "model", default="__all__"),
                    _filter_block("Algorithm", _options(data.metrics, "algorithm", add_all=True), "algorithm", default="__all__"),
                    _filter_block("Probability Adjustment", _options(data.metrics, "calibration", add_all=True), "calibration", default="__all__"),
                    _filter_block("Number Of Sites", _options(data.metrics, "clients", add_all=True), "clients", default="__all__"),
                    _filter_block("Data Split Type", _options(data.metrics, "partition", add_all=True), "partition", default="__all__"),
                    _filter_block("Data Difference Level", _options(data.metrics, "alpha", add_all=True), "alpha", default="__all__"),
                    _filter_block("Decision Rule", _options(data.metrics, "threshold_strategy", add_all=True), "threshold_strategy", default=_default_value(data.metrics, "threshold_strategy", "calib_f1_optimal", fallback="__all__")),
                ],
                className="filters",
            ),
            dcc.Tabs(
                [
                    dcc.Tab(
                        label="Key Findings",
                        children=[
                            html.Div(id="showcase-cards", className="hero-cards"),
                            html.Div(id="showcase-leaderboard", className="leaderboard-grid"),
                            html.Div(id="score-ceiling-panel", className="score-audit-grid"),
                            html.Div(
                                className="chart-grid two-up",
                                children=[
                                    _graph_panel(
                                        "showcase-roc",
                                        "This graph shows how well the selected Centralized and Federated models separate positive and negative cases across many possible cutoffs.",
                                        "It matters because it tells us whether Federated stays close to Centralized for overall ranking quality.",
                                    ),
                                    _graph_panel(
                                        "showcase-pr",
                                        "This graph shows how well the selected Centralized and Federated models catch positive cases while controlling precision.",
                                        "It matters because positive-case ranking is especially important when the dataset is imbalanced.",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="chart-grid two-up",
                                children=[
                                    _graph_panel(
                                        "showcase-confusion",
                                        "This graph shows the counts of correct and incorrect predictions in the held-out test fold at the chosen decision rule.",
                                        "It matters because it reveals whether the current operating point favors recall, specificity, or a balanced tradeoff on full-data evaluation.",
                                    ),
                                    _graph_panel(
                                        "showcase-distribution",
                                        "This graph shows the normalized spread of predicted risk scores for the positive and negative classes.",
                                        "It matters because clearer separation means the model is better at assigning higher risk to the right people, even when the class counts differ.",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="chart-grid two-up",
                                children=[
                                    _graph_panel(
                                        "showcase-threshold",
                                        "This graph shows how balanced detection, precision, recall, and group-gap measures change as the decision rule moves.",
                                        "It matters because it explains why one threshold is better for balanced detection while another is better for low false positives.",
                                    ),
                                    html.Div(className="summary-card", children=[html.Div(id="showcase-summary", className="summary-markdown")]),
                                ],
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Model Comparison",
                        children=[
                            html.Div(id="performance-cards", className="cards"),
                            html.Div(
                                className="chart-card",
                                children=[
                                    DashECharts(id="performance-echart", option=_empty_echart("Loading chart..."), style={"height": "470px", "width": "100%"}),
                                    _graph_help(
                                        "This graph shows how Centralized, Local-Only, Average Local-Only, and Federated setups compare on the selected metric view.",
                                        "It matters because it tells us whether Federated remains close to Centralized and whether it improves on isolated local baselines.",
                                    ),
                                    html.P(id="performance-note", className="caption"),
                                ],
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Risk Percentages",
                        children=[
                            _echart_panel(
                                "calibration-curve-echart",
                                "This graph shows whether the predicted risk percentages line up with the outcomes that actually happened.",
                                "It matters because a model can rank well but still give unreliable probabilities, and this view checks that directly.",
                            ),
                            html.Div(
                                className="chart-grid two-up",
                                children=[
                                    _echart_panel(
                                        "calibration-hist-echart",
                                        "This graph shows how many cases fall into each predicted risk range.",
                                        "It matters because it tells us whether the calibration pattern is backed by enough data in each part of the risk scale.",
                                    ),
                                    _echart_panel(
                                        "calibration-compare-echart",
                                        "This graph shows summary error measures for each probability-adjustment method.",
                                        "It matters because lower calibration errors mean the risk percentages are more trustworthy for decision support.",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Feature Explanations",
                        children=[
                            html.Div(
                                className="chart-grid two-up",
                                children=[
                                    _graph_panel(
                                        "xai-bar",
                                        "This graph shows which features have the strongest overall influence on the model's predictions.",
                                        "It matters because it tells us whether the model is relying on understandable clinical or lifestyle signals.",
                                    ),
                                    _graph_panel(
                                        "local-waterfall",
                                        "This graph shows which features pushed one representative case toward higher or lower diabetes risk.",
                                        "It matters because it gives a case-level explanation that a reader can inspect directly.",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="chart-grid two-up",
                                children=[
                                    _graph_panel(
                                        "stability-line",
                                        "This graph shows whether the explanation pattern stays consistent as Federated training moves from round to round.",
                                        "It matters because stable explanations make the Federated model easier to trust and describe.",
                                    ),
                                    _graph_panel(
                                        "stability-heatmap",
                                        "This graph shows whether different sites emphasize similar important features.",
                                        "It matters because strong cross-site agreement suggests the Federated explanation story is not wildly inconsistent across clients.",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Group Checks",
                        children=[
                            html.Div(
                                className="chart-grid two-up",
                                children=[
                                    _echart_panel(
                                        "fairness-selection-echart",
                                        "This graph shows how often each subgroup is predicted positive at the current decision rule.",
                                        "It matters because large differences can signal subgroup disparities that deserve discussion.",
                                    ),
                                    _echart_panel(
                                        "fairness-gap-echart",
                                        "This graph shows summary gap measures for how results differ between groups.",
                                        "It matters because it tells us whether the chosen setup introduces meaningful subgroup disparities.",
                                    ),
                                ],
                            ),
                            _echart_panel(
                                "fairness-threshold-echart",
                                "This graph shows how balanced detection and group-gap measures move as the decision rule changes.",
                                "It matters because it reveals where the model's fairness tradeoffs become better or worse.",
                                height=700,
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Learning Process",
                        children=[
                            html.Div(
                                className="chart-grid two-up",
                                children=[
                                    _echart_panel(
                                        "training-metrics-echart",
                                        "This graph shows how the Federated model's quality changes from round to round during training.",
                                        "It matters because it shows whether the global model improves steadily or becomes unstable as communication continues.",
                                        height=560,
                                    ),
                                    _echart_panel(
                                        "communication-echart",
                                        "This graph shows how many bytes are exchanged over Federated training rounds.",
                                        "It matters because it makes the communication cost of keeping raw data local visible.",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Full Study Comparison",
                        children=[
                            _echart_panel(
                                "study-comparison-echart",
                                "This graph shows how Centralized, Local-Only, and Federated setups compare across the selected slice of the study.",
                                "It matters because it summarizes whether Federated remains close to Centralized and improves on isolated local baselines.",
                            ),
                            html.Div(className="summary-card", children=[html.Div(id="study-note", className="summary-markdown")]),
                        ],
                    ),
                    dcc.Tab(label="Help / Glossary", children=[dcc.Markdown(_guide_markdown(), className="guide")]),
                ]
            ),
        ],
        className="shell",
    )

    app.index_string = _index_string()

    filter_outputs = []
    filter_inputs = []
    for column, _label, _allow_all in FILTER_SPECS:
        filter_outputs.extend([Output(column, "options"), Output(column, "value")])
        filter_inputs.append(Input(column, "value"))

    @app.callback(*filter_outputs, *filter_inputs)
    def sync_filter_state(*current_values):
        requested = {column: value for (column, _label, _allow_all), value in zip(FILTER_SPECS, current_values)}
        option_frames, resolved = _resolve_filter_state(data.metrics, requested)
        response: list[object] = []
        for column, _label, allow_all in FILTER_SPECS:
            response.append(_options(option_frames[column], column, add_all=allow_all))
            response.append(resolved[column])
        return tuple(response)

    @app.callback(
        Output("showcase-cards", "children"),
        Output("showcase-leaderboard", "children"),
        Output("score-ceiling-panel", "children"),
        Output("showcase-roc", "figure"),
        Output("showcase-pr", "figure"),
        Output("showcase-confusion", "figure"),
        Output("showcase-distribution", "figure"),
        Output("showcase-threshold", "figure"),
        Output("showcase-summary", "children"),
        Output("performance-cards", "children"),
        Output("performance-echart", "option"),
        Output("performance-note", "children"),
        Output("calibration-curve-echart", "option"),
        Output("calibration-hist-echart", "option"),
        Output("calibration-compare-echart", "option"),
        Output("xai-bar", "figure"),
        Output("local-waterfall", "figure"),
        Output("stability-line", "figure"),
        Output("stability-heatmap", "figure"),
        Output("fairness-selection-echart", "option"),
        Output("fairness-gap-echart", "option"),
        Output("fairness-threshold-echart", "option"),
        Output("training-metrics-echart", "option"),
        Output("communication-echart", "option"),
        Output("study-comparison-echart", "option"),
        Output("study-note", "children"),
        Input("dataset", "value"),
        Input("run_type", "value"),
        Input("model", "value"),
        Input("algorithm", "value"),
        Input("calibration", "value"),
        Input("clients", "value"),
        Input("partition", "value"),
        Input("alpha", "value"),
        Input("threshold_strategy", "value"),
    )
    def update(dataset: str, run_type: str, model: str, algorithm: str, calibration: str, clients: str, partition: str, alpha: str, threshold_strategy: str):
        criteria = {
            "dataset": dataset,
            "run_type": _coerce_filter(run_type),
            "model": _coerce_filter(model),
            "algorithm": _coerce_filter(algorithm),
            "calibration": _coerce_filter(calibration),
            "clients": _coerce_filter(clients),
            "partition": _coerce_filter(partition),
            "alpha": _coerce_filter(alpha),
            "threshold_strategy": _coerce_filter(threshold_strategy),
        }
        showcase_cards, leaderboard, roc_fig, pr_fig, confusion_fig, dist_fig, threshold_fig, summary_text = _showcase_panel(data, criteria)
        score_ceiling_panel = _score_ceiling_panel(data.score_ceiling, criteria)
        performance_cards = _performance_cards(data.metrics, criteria)
        performance_fig = _performance_echart(data.metrics, criteria)
        performance_note = _performance_note(data.metrics, criteria)
        calibration_curve = _calibration_curve_echart(data.calibration, criteria)
        calibration_hist = _calibration_hist_echart(data.calibration, criteria)
        calibration_compare = _calibration_compare_echart(data.metrics, criteria)
        xai_bar = _xai_bar(data.shap, criteria)
        local_waterfall = _local_explanation_figure(data.local_explanations, criteria)
        stability_line = _stability_line(data.stability, criteria)
        stability_heatmap = _stability_heatmap(data.stability, criteria)
        fairness_selection = _fairness_selection_echart(data.fairness, criteria)
        fairness_gap = _fairness_gap_echart(data.fairness, criteria)
        fairness_threshold = _fairness_threshold_echart(data.thresholds, criteria)
        training_metrics = _training_metrics_echart(data.rounds, criteria)
        communication_figure = _communication_echart(data.rounds, criteria)
        study_comparison = _study_comparison_echart(data.metrics, criteria)
        study_note = _study_note(data.metrics, data.showcase, criteria)
        return (
            showcase_cards,
            leaderboard,
            score_ceiling_panel,
            roc_fig,
            pr_fig,
            confusion_fig,
            dist_fig,
            threshold_fig,
            summary_text,
            performance_cards,
            performance_fig,
            performance_note,
            calibration_curve,
            calibration_hist,
            calibration_compare,
            xai_bar,
            local_waterfall,
            stability_line,
            stability_heatmap,
            fairness_selection,
            fairness_gap,
            fairness_threshold,
            training_metrics,
            communication_figure,
            study_comparison,
            study_note,
        )

    print(f"Dashboard running at http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


def _resolve_filter_state(frame: pd.DataFrame, requested: dict[str, str | None]) -> tuple[dict[str, pd.DataFrame], dict[str, str | None]]:
    option_frames: dict[str, pd.DataFrame] = {}
    resolved_values: dict[str, str | None] = {}
    applied: dict[str, str | None] = {}
    for column, _label, allow_all in FILTER_SPECS:
        current_frame = _apply_filters(frame, applied)
        option_frames[column] = current_frame
        resolved = _choose_filter_value(current_frame, column, requested.get(column), allow_all)
        resolved_values[column] = resolved
        applied[column] = None if resolved in {None, "__all__"} else resolved
    return option_frames, resolved_values


def _choose_filter_value(frame: pd.DataFrame, column: str, requested: str | None, allow_all: bool) -> str | None:
    if frame.empty or column not in frame.columns:
        return "__all__" if allow_all else None
    values = sorted(str(value) for value in frame[column].dropna().astype(str).unique())
    if requested in values:
        return requested
    if requested == "__all__" and allow_all:
        return "__all__"
    if column == "dataset" and "cdc" in values:
        return "cdc"
    if column == "threshold_strategy" and "calib_f1_optimal" in values:
        return "calib_f1_optimal"
    if allow_all:
        return "__all__"
    return values[0] if values else None


class DashboardData:
    def __init__(self, results_dir: Path, visual_results_dir: Path | None = None):
        self.results_dir = results_dir
        self.visual_results_dir = visual_results_dir
        self.metrics = _decorate(_read(results_dir / "dashboard_metrics.csv"))
        self.calibration = _decorate(_read(results_dir / "dashboard_calibration.csv"))
        self.fairness = _decorate(_read(results_dir / "dashboard_fairness.csv"))
        self.rounds = _decorate(_read(results_dir / "dashboard_rounds.csv"))
        self.shap = _decorate(_read(results_dir / "dashboard_shap.csv"))
        self.stability = _decorate(_read(results_dir / "dashboard_stability.csv"))
        self.thresholds = _decorate(_read(results_dir / "dashboard_thresholds.csv"))
        self.local_explanations = _decorate(_read(results_dir / "dashboard_local_explanations.csv"))
        self.curves = _decorate(_read_preferred(results_dir, visual_results_dir, "dashboard_curves.csv"))
        self.confusion = _decorate(_read_preferred(results_dir, visual_results_dir, "dashboard_confusion.csv"))
        self.showcase = _decorate(_read(results_dir / "dashboard_showcase_metrics.csv"))
        self.score_ceiling = _decorate(_read(results_dir / "dashboard_score_ceiling.csv"))


def _load_dashboard_data(results_dir: Path, visual_results_dir: Path | None = None) -> DashboardData:
    return DashboardData(results_dir, _discover_visual_results_dir(results_dir, visual_results_dir))


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_preferred(results_dir: Path, visual_results_dir: Path | None, filename: str) -> pd.DataFrame:
    preferred_paths: list[Path] = []
    if visual_results_dir is not None:
        preferred_paths.append(visual_results_dir / filename)
    preferred_paths.append(results_dir / filename)
    if visual_results_dir is None:
        for sibling in [
            results_dir.parent / "polished_visual_verify_summary",
            results_dir.parent / "showcase_summary",
        ]:
            if sibling != results_dir:
                preferred_paths.append(sibling / filename)
    for path in preferred_paths:
        frame = _read(path)
        if not frame.empty:
            return frame
    return pd.DataFrame()


def _discover_visual_results_dir(results_dir: Path, requested: Path | None) -> Path | None:
    candidates: list[Path] = []
    if requested is not None:
        candidates.append(requested)
    candidates.extend(
        [
            results_dir.parent / "polished_visual_verify_summary",
            results_dir.parent / "showcase_summary",
        ]
    )
    for candidate in candidates:
        if candidate is None or candidate == results_dir:
            continue
        if (candidate / "dashboard_curves.csv").exists() or (candidate / "dashboard_confusion.csv").exists():
            return candidate
    return None


def _decorate(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    for column in [
        "dataset",
        "experiment_track",
        "run_type",
        "model",
        "algorithm",
        "calibration",
        "partition",
        "threshold_strategy",
        "showcase_role",
        "curve_type",
        "cell",
    ]:
        if column in output.columns:
            output[f"{column}_label"] = output[column].astype(str).map(_humanize)
    return output


def _filter_block(label: str, options: list[dict[str, str]], component_id: str, default: str | None = None):
    from dash import dcc, html

    value = default if default is not None else (options[0]["value"] if options else None)
    return html.Div([html.Label(label), dcc.Dropdown(options, value, id=component_id, clearable=False)], className="filter")


def _graph_panel(graph_id: str, what_text: str, why_text: str):
    from dash import dcc, html

    return html.Div(
        className="chart-card",
        children=[
            dcc.Graph(id=graph_id),
            _graph_help(what_text, why_text),
        ],
    )


def _echart_panel(echart_id: str, what_text: str, why_text: str, *, height: int = 430):
    from dash import html

    if DashECharts is None:  # pragma: no cover
        raise RuntimeError("DashECharts is not available. Install dash-echarts==0.0.12.9 before running the dashboard.")

    return html.Div(
        className="chart-card",
        children=[
            DashECharts(
                id=echart_id,
                option=_empty_echart("Loading chart..."),
                style={"height": f"{height}px", "width": "100%"},
            ),
            _graph_help(what_text, why_text),
        ],
    )


def _graph_help(what_text: str, why_text: str):
    from dash import html

    return html.Div(
        [
            html.P([html.Span("What this graph shows: ", className="graph-help-label"), what_text], className="graph-help-line"),
            html.P([html.Span("Why it matters: ", className="graph-help-label"), why_text], className="graph-help-line"),
        ],
        className="graph-help",
    )


def _options(frame: pd.DataFrame, column: str, add_all: bool = False) -> list[dict[str, str]]:
    if frame.empty or column not in frame.columns:
        return []
    values = sorted(str(value) for value in frame[column].dropna().astype(str).unique())
    options = [{"label": _humanize(value), "value": value} for value in values]
    if add_all:
        return [{"label": "All trained options", "value": "__all__"}] + options
    return options


def _default_value(frame: pd.DataFrame, column: str, preferred: str, fallback: str | None = None) -> str | None:
    if frame.empty or column not in frame.columns:
        return fallback
    values = {str(value) for value in frame[column].dropna().astype(str).unique()}
    if preferred in values:
        return preferred
    return fallback if fallback is not None else (sorted(values)[0] if values else None)


def _coerce_filter(value: str | None) -> str | None:
    if value in {None, "__all__"}:
        return None
    return value


def _apply_filters(frame: pd.DataFrame, criteria: dict[str, str | None], allow_shared_baseline: bool = True) -> pd.DataFrame:
    output = frame.copy()
    if output.empty:
        return output
    for column, value in criteria.items():
        if value is None or column not in output.columns:
            continue
        series = output[column].astype(str)
        if column == "algorithm":
            mask = series.eq(str(value))
            if "run_type" in output.columns:
                mask = mask | output["algorithm"].astype(str).eq("not_applicable")
            output = output[mask]
            continue
        if allow_shared_baseline and column in {"clients", "partition", "alpha"} and "run_type" in output.columns:
            shared_mask = output["run_type"].astype(str).eq("centralized") & series.eq("shared")
            output = output[series.eq(str(value)) | shared_mask]
            continue
        output = output[series.eq(str(value))]
    return output


def _row_matches(row: pd.Series, criteria: dict[str, str | None]) -> bool:
    for column, value in criteria.items():
        if value is None or column not in row.index:
            continue
        row_value = str(row[column])
        if column in {"clients", "partition", "alpha"} and str(row.get("run_type", "")) == "centralized" and row_value == "shared":
            continue
        if column == "algorithm" and row_value == "not_applicable":
            continue
        if row_value != str(value):
            return False
    return True


def _showcase_panel(data: DashboardData, criteria: dict[str, str | None]):
    from dash import html

    showcase = data.showcase.copy()
    if criteria["dataset"] is not None and "dataset" in showcase.columns:
        showcase = showcase[showcase["dataset"].astype(str) == str(criteria["dataset"])]
    showcase = showcase[showcase["threshold_strategy"].astype(str) == "calib_f1_optimal"] if "threshold_strategy" in showcase.columns else showcase

    central = _pick_showcase_row(showcase, "best_centralized_decision", criteria)
    federated = _pick_showcase_row(showcase, "best_federated_decision", criteria)
    probability_quality = _pick_probability_row(showcase, criteria)
    low_false_positive = _pick_score_row(data.score_ceiling, criteria, ["Best Selected-Threshold Specificity", "High Specificity View"])

    cards = []
    cards.append(
        _hero_card(
            "Best Centralized Model",
            _metric_line(central, "roc_auc_mean", label="Ranking Quality (AUROC)"),
            _sub_line(central, "f1_mean", "Balanced Detection Score (F1)"),
            "This is the strongest Centralized reference for the current selection.",
        )
    )
    cards.append(
        _hero_card(
            "Best Federated Model",
            _metric_line(federated, "roc_auc_mean", label="Ranking Quality (AUROC)"),
            _sub_line(federated, "f1_mean", "Balanced Detection Score (F1)"),
            "This is the strongest Federated result for the current selection.",
        )
    )
    cards.append(
        _hero_card(
            "How Close Federated Gets",
            _gap_line(central, federated, "roc_auc_mean", prefix="Ranking gap"),
            _gap_line(central, federated, "f1_mean", prefix="Balanced detection gap"),
            "Smaller gaps mean the Federated model stays close to the Centralized reference.",
        )
    )
    cards.append(
        _hero_card(
            "Best Probability-Quality Setup",
            _metric_line(probability_quality, "ece_mean", label="Risk Match Error (ECE)"),
            _metric_line(probability_quality, "brier_mean", label="Probability Error (Brier)"),
            "Lower values mean the risk percentages are more trustworthy.",
        )
    )
    cards.append(
        _hero_card(
            "Best Low False-Positive Setup",
            _score_metric_line(low_false_positive, "Specificity"),
            _score_metric_line(low_false_positive, "False-Positive Rate", field="false_positive_rate"),
            "This view is useful when avoiding false alarms matters most.",
        )
    )

    leaderboard = [
        _leaderboard_card("Best Centralized Model", _row_badges(central)),
        _leaderboard_card("Best Federated Model", _row_badges(federated)),
        _leaderboard_card("Best Probability Adjustment", _row_badges(probability_quality)),
        _leaderboard_card("Best Overall Evidence Slice", _row_badges(_best_trustworthiness(central, federated, probability_quality))),
    ]

    visual_source = data.curves if not data.curves.empty else data.confusion
    central_visual, central_visual_fallback = _resolve_visual_row(visual_source, central)
    federated_visual, federated_visual_fallback = _resolve_visual_row(visual_source, federated)

    roc_fig = _curve_compare(data.curves, [central_visual, federated_visual], "roc", "False Positive Rate", "True Positive Rate", "How Well The Best Models Separate Risk")
    pr_fig = _curve_compare(data.curves, [central_visual, federated_visual], "pr", "Recall", "Precision", "How Well The Best Models Catch Positive Cases")
    confusion_fig = _confusion_compare(data.confusion, [central_visual, federated_visual])
    distribution_fig = _distribution_compare(data.curves, [central_visual, federated_visual])
    threshold_target = central if central is not None else federated
    threshold_fig = _showcase_threshold_figure(data.thresholds, threshold_target)
    visual_note_bits = []
    if central_visual_fallback and central_visual is not None:
        visual_note_bits.append(f"Centralized visual panels use {_humanize(central_visual.get('model', ''))} as the closest saved prediction slice")
    if federated_visual_fallback and federated_visual is not None:
        visual_note_bits.append(f"Federated visual panels use {_humanize(federated_visual.get('model', ''))} as the closest saved prediction slice")
    test_total = _confusion_total(data.confusion, central_visual) or _confusion_total(data.confusion, federated_visual)
    summary = _showcase_summary_markdown(central, federated, probability_quality, visual_note_bits, test_total)
    return cards, leaderboard, roc_fig, pr_fig, confusion_fig, distribution_fig, threshold_fig, summary


def _score_ceiling_panel(score_ceiling: pd.DataFrame, criteria: dict[str, str | None]):
    from dash import html

    if score_ceiling.empty:
        return [
            html.Div(
                [
            html.Div("Decision Review", className="score-title"),
            html.Div("No decision-review summary is available yet.", className="score-note"),
                ],
                className="score-card",
            )
        ]
    frame = score_ceiling.copy()
    if criteria.get("dataset") is not None and "dataset" in frame.columns:
        frame = frame[frame["dataset"].astype(str) == str(criteria["dataset"])]
    if frame.empty:
        return [
            html.Div(
                [
                    html.Div("Decision Review", className="score-title"),
                    html.Div("No decision-review rows match this data source.", className="score-note"),
                ],
                className="score-card",
            )
        ]

    desired = [
        "Best Overall Accuracy",
        "Best F1 Operating Point",
        "High Recall View",
        "Best Selected-Threshold Specificity",
        "High Specificity View",
        "Best Calibration Error",
        "Explanation Stability",
    ]
    cards = []
    for score_card in desired:
        subset = frame[frame["score_card"].astype(str) == score_card].copy()
        if subset.empty:
            continue
        row = subset.iloc[0]
        value = float(pd.to_numeric(row.get("value"), errors="coerce"))
        metric = str(row.get("metric_label", row.get("metric", "")))
        cards.append(
            html.Div(
                [
                    html.Div(_decision_review_kicker(row), className="score-kicker"),
                    html.Div(_decision_review_title(str(row.get("score_card", ""))), className="score-title"),
                    html.Div(f"{metric}: {_fmt(value)}", className="score-value"),
                    html.Div(_score_context(row), className="score-context"),
                    html.Div(str(row.get("caveat", "")), className="score-note"),
                ],
                className="score-card",
            )
        )
    cards.append(
        html.Div(
            [
                html.Div("How To Read These Decision Views", className="score-title"),
                html.Div(
                    "Each card shows a different operating goal, such as stronger recall, fewer false positives, better probability quality, or more stable explanations. "
                    "The metric names stay exact, so the evidence remains easy to defend.",
                    className="score-note",
                ),
            ],
            className="score-card truth",
        )
    )
    return cards


def _score_context(row: pd.Series) -> str:
    pieces = [
        _humanize(row.get("run_type", "")),
        _humanize(row.get("model", "")),
        _humanize(row.get("algorithm", "")),
        _humanize(row.get("calibration", "")),
    ]
    threshold = row.get("threshold")
    if pd.notna(threshold):
        pieces.append(f"Decision rule {_fmt(threshold)}")
    return " / ".join(piece for piece in pieces if piece and piece != "Not Applicable")


def _pick_score_row(score_ceiling: pd.DataFrame, criteria: dict[str, str | None], score_cards: list[str]) -> pd.Series | None:
    if score_ceiling.empty:
        return None
    frame = score_ceiling.copy()
    if criteria.get("dataset") is not None and "dataset" in frame.columns:
        frame = frame[frame["dataset"].astype(str) == str(criteria["dataset"])]
    frame = frame[frame["score_card"].astype(str).isin(score_cards)].copy()
    if frame.empty:
        return None
    filtered = frame[frame.apply(lambda row: _row_matches(row, criteria), axis=1)]
    if not filtered.empty:
        frame = filtered
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value"])
    if frame.empty:
        return None
    return frame.sort_values("value", ascending=False).iloc[0]


def _score_metric_line(row: pd.Series | None, label: str, field: str = "value") -> str:
    if row is None:
        return f"{label}: not available in this run version"
    return f"{label}: {_fmt(row.get(field))}"


def _decision_review_kicker(row: pd.Series) -> str:
    title = str(row.get("score_card", "")).lower()
    if "recall" in title:
        return "High-catch setting"
    if "specificity" in title:
        return "Low false-positive setting"
    if "accuracy" in title:
        return "Overall accuracy setting"
    if "calibration" in title:
        return "Probability-quality setting"
    if "stability" in title:
        return "Explanation setting"
    return "Decision setting"


def _decision_review_title(title: str) -> str:
    mapping = {
        "Best Overall Accuracy": "Highest Overall Accuracy",
        "Best F1 Operating Point": "Strongest Balanced Detection Setting",
        "High Recall View": "High-Catch Setting",
        "Best Selected-Threshold Specificity": "Lowest False-Positive Setting",
        "High Specificity View": "High-Specificity Setting",
        "Best Calibration Error": "Best Probability-Quality Setting",
        "Explanation Stability": "Most Stable Explanation Setting",
    }
    return mapping.get(title, title)


def _pick_showcase_row(showcase: pd.DataFrame, role: str, criteria: dict[str, str | None]) -> pd.Series | None:
    if showcase.empty or "showcase_role" not in showcase.columns:
        return None
    frame = showcase[showcase["showcase_role"].astype(str) == role].copy()
    if frame.empty:
        return None
    filtered = frame[frame.apply(lambda row: _row_matches(row, criteria), axis=1)]
    if not filtered.empty:
        frame = filtered
    sort_cols = [column for column in ["f1_mean", "roc_auc_mean", "pr_auc_mean", "ece_mean"] if column in frame.columns]
    ascend = [column == "ece_mean" for column in sort_cols]
    return frame.sort_values(sort_cols, ascending=ascend).iloc[0]


def _pick_probability_row(showcase: pd.DataFrame, criteria: dict[str, str | None]) -> pd.Series | None:
    if showcase.empty or "showcase_role" not in showcase.columns:
        return None
    frame = showcase[showcase["showcase_role"].astype(str) == "best_probability_quality"].copy()
    if frame.empty:
        return None
    filtered = frame[frame.apply(lambda row: _row_matches(row, criteria), axis=1)]
    if not filtered.empty:
        frame = filtered
    return frame.sort_values(["ece_mean", "brier_mean", "roc_auc_mean"], ascending=[True, True, False]).iloc[0]


def _best_trustworthiness(*rows: pd.Series | None) -> pd.Series | None:
    valid = [row for row in rows if row is not None]
    if not valid:
        return None
    frame = pd.DataFrame(valid)
    return frame.sort_values(["f1_mean", "roc_auc_mean", "ece_mean"], ascending=[False, False, True]).iloc[0]


def _hero_card(title: str, value: str, detail: str, note: str):
    from dash import html

    return html.Div(
        [
            html.Div(title, className="hero-title"),
            html.Div(value, className="hero-value"),
            html.Div(detail, className="hero-detail"),
            html.Div(note, className="hero-note"),
        ],
        className="hero-card",
    )


def _leaderboard_card(title: str, badges: list[str]):
    from dash import html

    if not badges:
        badges = ["No matching slice"]
    return html.Div(
        [
            html.Div(title, className="leaderboard-title"),
            html.Div([html.Span(text, className="badge") for text in badges], className="badge-row"),
        ],
        className="leaderboard-card",
    )


def _metric_line(row: pd.Series | None, column: str, label: str | None = None) -> str:
    if row is None:
        return "n/a"
    label_text = label or _humanize(column.replace("_mean", ""))
    return f"{label_text}: {_fmt(row.get(column))}"


def _sub_line(row: pd.Series | None, column: str, label: str) -> str:
    if row is None:
        return f"{label}: n/a"
    return f"{label}: {_fmt(row.get(column))} | 95% CI {_fmt(row.get(column.replace('_mean', '_ci95_low')))} to {_fmt(row.get(column.replace('_mean', '_ci95_high')))}"


def _gap_line(central: pd.Series | None, fed: pd.Series | None, column: str, prefix: str | None = None) -> str:
    label = prefix or "AUROC gap"
    if central is None or fed is None:
        return f"{label}: n/a"
    gap = float(fed.get(column, np.nan)) - float(central.get(column, np.nan))
    return f"{label}: {gap:+.3f}"


def _row_badges(row: pd.Series | None) -> list[str]:
    if row is None:
        return []
    pieces = [
        _humanize(row.get("run_type", "")),
        _humanize(row.get("model", "")),
        _humanize(row.get("algorithm", "")),
        _humanize(row.get("calibration", "")),
        _humanize(row.get("threshold_strategy", "")),
        f"Ranking Quality {_fmt(row.get('roc_auc_mean'))}",
        f"Balanced Detection {_fmt(row.get('f1_mean'))}",
        _metric_badge(row, "pr_auc_mean", "Positive-Case Ranking"),
        f"Risk Match Error {_fmt(row.get('ece_mean'))}",
    ]
    return [piece for piece in pieces if piece and piece != "Not Applicable"]


def _metric_badge(row: pd.Series, column: str, label: str) -> str | None:
    value = row.get(column)
    if pd.isna(value):
        return None
    return f"{label} {_fmt(value)}"


def _resolve_visual_row(frame: pd.DataFrame, row: pd.Series | None) -> tuple[pd.Series | None, bool]:
    if row is None or frame.empty:
        return None, False
    exact = _filter_exact(frame, row)
    if not exact.empty:
        return row, False

    id_columns = [col for col in ["dataset", "experiment_track", "clients", "partition", "alpha", "run_type", "algorithm", "model", "calibration", "threshold_strategy"] if col in frame.columns]
    candidates = frame[id_columns].drop_duplicates().copy()
    if candidates.empty:
        return None, False

    hard_filters = ["dataset", "experiment_track", "run_type", "threshold_strategy"]
    for column in hard_filters:
        if column in candidates.columns and column in row.index:
            candidates = candidates[candidates[column].astype(str) == str(row[column])]
    if candidates.empty:
        return None, False

    optional_match_columns = ["clients", "partition", "alpha", "algorithm", "calibration", "model"]
    working = candidates.copy()
    for column in optional_match_columns:
        if column in working.columns and column in row.index:
            match_value = str(row[column])
            matched = working[working[column].astype(str) == match_value]
            if not matched.empty:
                working = matched

    if len(working) == 1:
        return working.iloc[0], True

    scored = working.copy()
    score = pd.Series(0, index=scored.index, dtype=float)
    weights = {
        "algorithm": 8.0,
        "clients": 6.0,
        "partition": 6.0,
        "alpha": 5.0,
        "calibration": 4.0,
        "model": 2.0,
    }
    for column, weight in weights.items():
        if column in scored.columns and column in row.index:
            score = score + (scored[column].astype(str) == str(row[column])).astype(float) * weight
    scored["__match_score"] = score
    scored = scored.sort_values(["__match_score"], ascending=False)
    return scored.iloc[0].drop(labels="__match_score"), True


def _confusion_total(confusion: pd.DataFrame, row: pd.Series | None) -> int | None:
    if row is None or confusion.empty:
        return None
    subset = _filter_exact(confusion, row)
    if subset.empty or "count_mean" not in subset.columns:
        return None
    return int(round(float(subset["count_mean"].sum())))


def _curve_compare(curves: pd.DataFrame, rows: list[pd.Series | None], curve_type: str, x_title: str, y_title: str, title: str) -> go.Figure:
    frame = curves[curves["curve_type"].astype(str) == curve_type].copy() if not curves.empty else pd.DataFrame()
    if frame.empty:
        return _empty(title, "This comparison curve is not available for the current selection.")
    fig = go.Figure()
    colors = [ACCENT["primary"], ACCENT["secondary"]]
    labels = ["Centralized", "Federated"]
    for row, color, label in zip(rows, colors, labels):
        if row is None:
            continue
        subset = _filter_exact(frame, row)
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["curve_x_mean"],
                y=subset["curve_y_mean"],
                mode="lines",
                name=f"{label} ({_humanize(row.get('model', ''))})",
                line={"color": color, "width": 3},
            )
        )
        if {"curve_y_ci95_low", "curve_y_ci95_high"}.issubset(subset.columns):
            fig.add_trace(go.Scatter(x=subset["curve_x_mean"], y=subset["curve_y_ci95_high"], mode="lines", line={"width": 0}, showlegend=False))
            fig.add_trace(
                go.Scatter(
                    x=subset["curve_x_mean"],
                    y=subset["curve_y_ci95_low"],
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=_rgba(color, 0.14),
                    showlegend=False,
                )
            )
    if curve_type == "roc":
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash", "color": "#8B97A6"}, name="Chance"))
    _apply_figure_style(fig, title, x_title, y_title, height=430)
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


def _confusion_compare(confusion: pd.DataFrame, rows: list[pd.Series | None]) -> go.Figure:
    if confusion.empty:
        return _empty("How The Best Models Classify Cases", "This confusion-matrix view is not available for the current selection.")
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Centralized", "Federated"], horizontal_spacing=0.16)
    for idx, row in enumerate(rows, start=1):
        if row is None:
            fig.add_annotation(
                row=1,
                col=idx,
                text="No saved confusion view for this slice",
                showarrow=False,
                font={"size": 13, "color": ACCENT["muted"]},
            )
            continue
        subset = _filter_exact(confusion, row)
        if subset.empty:
            fig.add_annotation(
                row=1,
                col=idx,
                text="No saved confusion view for this slice",
                showarrow=False,
                font={"size": 13, "color": ACCENT["muted"]},
            )
            continue
        cell_map = {str(item["cell"]): float(item["count_mean"]) for _, item in subset.iterrows()}
        matrix = np.asarray(
            [
                [cell_map.get("TN", 0.0), cell_map.get("FP", 0.0)],
                [cell_map.get("FN", 0.0), cell_map.get("TP", 0.0)],
            ]
        )
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
    _apply_figure_style(fig, "How The Best Models Classify Cases", "", "", height=420, legend=False)
    return fig


def _distribution_compare(curves: pd.DataFrame, rows: list[pd.Series | None]) -> go.Figure:
    frame = curves[curves["curve_type"].astype(str) == "score_hist"].copy() if not curves.empty else pd.DataFrame()
    if frame.empty:
        return _empty("How Risk Scores Spread Across Classes", "This score-distribution view is not available for the current selection.")
    fig = go.Figure()
    style_map = {
        ("Centralized", "class_0"): (ACCENT["primary"], "dot"),
        ("Centralized", "class_1"): (ACCENT["primary"], "solid"),
        ("Federated", "class_0"): (ACCENT["secondary"], "dot"),
        ("Federated", "class_1"): (ACCENT["secondary"], "solid"),
    }
    labels = ["Centralized", "Federated"]
    for row, scenario_label in zip(rows, labels):
        if row is None:
            continue
        subset = _filter_exact(frame, row)
        if subset.empty:
            continue
        for class_label in sorted(subset["class_label"].astype(str).unique()):
            series = subset[subset["class_label"].astype(str) == class_label]
            color, dash = style_map.get((scenario_label, class_label), (ACCENT["muted"], "solid"))
            fig.add_trace(
                go.Scatter(
                    x=series["curve_x_mean"],
                    y=series["curve_y_mean"],
                    mode="lines",
                    name=f"{scenario_label} {_humanize(class_label)}",
                    line={"color": color, "dash": dash, "width": 3 if class_label.endswith("1") else 2},
                )
            )
    _apply_figure_style(fig, "How Risk Scores Spread Across Classes", "Predicted risk", "Normalized share within each class", height=430)
    fig.update_xaxes(range=[0, 1])
    return fig


def _showcase_threshold_figure(thresholds: pd.DataFrame, row: pd.Series | None) -> go.Figure:
    if row is None or thresholds.empty:
        return _empty("What Changes When We Raise Or Lower The Decision Rule", "No decision-rule sweep rows are available for the current selection.")
    subset = _filter_exact(thresholds, row)
    if subset.empty:
        return _empty("What Changes When We Raise Or Lower The Decision Rule", "No decision-rule sweep rows are available for the current selection.")
    subset = subset[subset["group_value"].astype(str) == "aggregate"] if "group_value" in subset.columns else subset
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Balanced detection and catch rate",
            "Precision and false-positive control",
            "Positive rate gap",
            "Error gap between groups",
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.16,
    )
    overall = subset.groupby("threshold", as_index=False)[["f1_mean", "precision_mean", "recall_mean", "specificity_mean"]].mean(numeric_only=True)
    metric_styles = [
        ("f1_mean", "Balanced Detection Score", ACCENT["primary"], "solid"),
        ("recall_mean", "Recall", ACCENT["secondary"], "solid"),
        ("precision_mean", "Precision", ACCENT["warning"], "solid"),
        ("specificity_mean", "Specificity", "#7C4DFF", "dash"),
    ]
    for column, label, color, dash in metric_styles:
        if column not in overall.columns:
            continue
        target_row, target_col = (1, 1) if column in {"f1_mean", "recall_mean"} else (1, 2)
        fig.add_trace(
            go.Scatter(
                x=overall["threshold"],
                y=overall[column],
                mode="lines",
                name=label,
                line={"color": color, "width": 3, "dash": dash},
            ),
            row=target_row,
            col=target_col,
        )
    palette = {"age_group": ACCENT["primary"], "bmi_category": ACCENT["warning"], "sex": ACCENT["secondary"]}
    for feature, group in subset.groupby("group_feature", dropna=False):
        ordered = group.sort_values("threshold")
        color = palette.get(str(feature), ACCENT["muted"])
        fig.add_trace(
            go.Scatter(
                x=ordered["threshold"],
                y=ordered["demographic_parity_difference_mean"],
                mode="lines",
                name=f"{_humanize(feature)} positive gap",
                legendgroup=str(feature),
                line={"color": color, "width": 2},
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ordered["threshold"],
                y=ordered["equalized_odds_difference_mean"],
                mode="lines",
                name=f"{_humanize(feature)} error gap",
                legendgroup=str(feature),
                line={"color": color, "width": 2, "dash": "dot"},
                showlegend=False,
            ),
            row=2,
            col=2,
        )
    if "selected_threshold_mean" in subset.columns:
        selected = float(pd.to_numeric(subset["selected_threshold_mean"], errors="coerce").dropna().mean())
        for row_idx in (1, 2):
            for col_idx in (1, 2):
                fig.add_vline(x=selected, line_dash="dash", line_color="#6B7280", row=row_idx, col=col_idx)
    for row_idx in (1, 2):
        for col_idx in (1, 2):
            fig.update_xaxes(title="Decision threshold", range=[0, 1], row=row_idx, col=col_idx)
    fig.update_yaxes(title="Score", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title="Score", range=[0, 1], row=1, col=2)
    fig.update_yaxes(title="Gap", row=2, col=1)
    fig.update_yaxes(title="Gap", row=2, col=2)
    _apply_figure_style(fig, "What Changes When We Raise Or Lower The Decision Rule", None, None, height=620)
    return fig


def _showcase_summary_markdown(
    central: pd.Series | None,
    federated: pd.Series | None,
    probability_quality: pd.Series | None,
    visual_note_bits: list[str] | None = None,
    test_total: int | None = None,
) -> str:
    lines = ["### What This Page Is Saying"]
    if central is not None:
        lines.append(
            f"- **Best Centralized reference:** {_humanize(central.get('model', ''))} with {_humanize(central.get('calibration', ''))}, ranking quality {_fmt(central.get('roc_auc_mean'))}, balanced detection {_fmt(central.get('f1_mean'))}, and positive-case ranking {_fmt(central.get('pr_auc_mean'))}."
        )
    if federated is not None:
        lines.append(
            f"- **Best Federated result:** {_humanize(federated.get('algorithm', ''))} {_humanize(federated.get('model', ''))}, ranking quality {_fmt(federated.get('roc_auc_mean'))}, balanced detection {_fmt(federated.get('f1_mean'))}, and positive-case ranking {_fmt(federated.get('pr_auc_mean'))}."
        )
    if central is not None and federated is not None:
        gap_auc = float(federated.get("roc_auc_mean", np.nan)) - float(central.get("roc_auc_mean", np.nan))
        gap_f1 = float(federated.get("f1_mean", np.nan)) - float(central.get("f1_mean", np.nan))
        lines.append(f"- **How close federated gets:** ranking gap {gap_auc:+.3f} and balanced detection gap {gap_f1:+.3f}.")
    if probability_quality is not None:
        lines.append(
            f"- **Most trustworthy risk percentages:** {_humanize(probability_quality.get('run_type', ''))} {_humanize(probability_quality.get('model', ''))} with risk-match error {_fmt(probability_quality.get('ece_mean'))} and probability error {_fmt(probability_quality.get('brier_mean'))}."
        )
    if test_total is not None:
        lines.append(
            f"- **What the confusion counts represent:** the classification counts come from the held-out full-CDC test fold, with about {test_total:,} cases in the selected evaluation slice."
        )
    for note in visual_note_bits or []:
        lines.append(f"- **Visual evidence note:** {note}.")
    lines.append("- **Why this matters:** the Centralized model remains the performance upper bound, while the Federated model stays close without pooling raw data.")
    lines.append("- **How to use the rest of the dashboard:** this page gives the headline story, and the other tabs show the evidence behind it in calibration, group checks, explanations, and communication cost.")
    return "\n".join(lines)


def _performance_cards(metrics: pd.DataFrame, criteria: dict[str, str | None]):
    from dash import html

    frame = _apply_filters(metrics, criteria)
    if frame.empty:
        return [html.Div([html.Div("Unsupported selection", className="card-title"), html.Div("This combination was not trained in the current study.", className="card-note")], className="card")]
    cards = []
    by_run_type = {
        "Best Centralized Model": frame[frame["run_type"].astype(str) == "centralized"],
        "Best Federated Model": frame[frame["run_type"].astype(str) == "federated"],
        "Average Local-Only": frame[frame["run_type"].astype(str) == "local_only_mean"],
    }
    best_rows: dict[str, pd.Series] = {}
    for title, subset in by_run_type.items():
        if subset.empty:
            continue
        row = subset.sort_values(["f1_mean", "roc_auc_mean", "ece_mean"], ascending=[False, False, True]).iloc[0]
        best_rows[title] = row
        cards.append(
            html.Div(
                [
                    html.Div(title, className="card-title"),
                    html.Div(f"Ranking Quality {_fmt(row.get('roc_auc_mean'))}", className="card-value"),
                    html.Div(f"Balanced Detection {_fmt(row.get('f1_mean'))} | Positive-Case Ranking {_fmt(row.get('pr_auc_mean'))}", className="card-meta"),
                    html.Div(f"{_humanize(row.get('model', ''))} / {_humanize(row.get('calibration', ''))}", className="card-note"),
                ],
                className="card",
            )
        )
    if "Best Centralized Model" in best_rows and "Best Federated Model" in best_rows:
        gap_auc = float(best_rows["Best Federated Model"].get("roc_auc_mean", np.nan)) - float(best_rows["Best Centralized Model"].get("roc_auc_mean", np.nan))
        gap_f1 = float(best_rows["Best Federated Model"].get("f1_mean", np.nan)) - float(best_rows["Best Centralized Model"].get("f1_mean", np.nan))
        cards.append(
            html.Div(
                [
                    html.Div("How Close Federated Gets", className="card-title"),
                    html.Div(f"Ranking gap {gap_auc:+.3f}", className="card-value"),
                    html.Div(f"Balanced detection gap {gap_f1:+.3f}", className="card-meta"),
                    html.Div("Negative values mean the Federated result trails the Centralized reference.", className="card-note"),
                ],
                className="card accent",
            )
        )
    return cards


def _performance_echart(metrics: pd.DataFrame, criteria: dict[str, str | None]) -> dict:
    frame = _apply_filters(metrics, criteria)
    if frame.empty:
        return _empty_echart("This combination was not trained in the current study.", "How The Trained Models Compare")
    frame = frame.copy()
    frame["run_type_name"] = frame["run_type"].map(_humanize)
    frame["model_name"] = frame["model"].map(_humanize)
    bubble_metric = "pr_auc_mean" if "pr_auc_mean" in frame.columns and frame["pr_auc_mean"].notna().any() else "roc_auc_mean"
    bubble_values = pd.to_numeric(frame[bubble_metric], errors="coerce").fillna(pd.to_numeric(frame["f1_mean"], errors="coerce"))
    min_bubble = float(bubble_values.min()) if not bubble_values.empty else 0.0
    max_bubble = float(bubble_values.max()) if not bubble_values.empty else 1.0
    bubble_span = max(max_bubble - min_bubble, 1e-6)
    series = []
    for run_type_name, group in frame.groupby("run_type_name", dropna=False):
        points = []
        for _, item in group.iterrows():
            bubble = pd.to_numeric(item.get(bubble_metric), errors="coerce")
            bubble_size = 12 + 16 * ((float(bubble) - min_bubble) / bubble_span if pd.notna(bubble) else 0.4)
            points.append(
                {
                    "name": f"{item['model_name']} ({run_type_name})",
                    "value": [
                        float(pd.to_numeric(item.get("roc_auc_mean"), errors="coerce")),
                        float(pd.to_numeric(item.get("f1_mean"), errors="coerce")),
                        float(pd.to_numeric(item.get("pr_auc_mean"), errors="coerce")) if pd.notna(item.get("pr_auc_mean")) else np.nan,
                        float(pd.to_numeric(item.get("ece_mean"), errors="coerce")) if pd.notna(item.get("ece_mean")) else np.nan,
                    ],
                    "symbolSize": bubble_size,
                }
            )
        series.append(
            {
                "name": str(run_type_name),
                "type": "scatter",
                "dimensions": [
                    "Ranking Quality (AUROC)",
                    "Balanced Detection Score (F1)",
                    "Positive-Case Ranking (PR AUC)",
                    "Risk Match Error (ECE)",
                ],
                "encode": {"x": 0, "y": 1, "tooltip": [0, 1, 2, 3]},
                "data": points,
                "itemStyle": {"color": RUN_TYPE_COLORS.get(str(run_type_name), ACCENT["primary"]), "opacity": 0.88},
            }
        )
    return {
        "backgroundColor": ACCENT["panel"],
        "title": _echart_title("How The Trained Models Compare"),
        "tooltip": _echart_tooltip(trigger="item"),
        "toolbox": _echart_toolbox(),
        "legend": _echart_legend(top=14),
        "grid": {"left": 72, "right": 36, "top": 70, "bottom": 58},
        "xAxis": _echart_axis("Average ranking quality", min_value=0, max_value=1),
        "yAxis": _echart_axis("Average balanced detection score", min_value=0, max_value=1),
        "series": series,
    }


def _calibration_curve_echart(calibration: pd.DataFrame, criteria: dict[str, str | None]) -> dict:
    frame = _apply_filters(calibration, criteria)
    if frame.empty:
        return _empty_echart("No aggregated probability-adjustment bins are available for this selection.", "Do The Risk Percentages Match Reality?")
    series: list[dict] = []
    palette = px.colors.qualitative.Plotly
    for idx, (label, group) in enumerate(frame.groupby("calibration_label", dropna=False)):
        group = group.sort_values("avg_predicted_risk_mean")
        x = pd.to_numeric(group["avg_predicted_risk_mean"], errors="coerce").fillna(0.0).tolist()
        y = pd.to_numeric(group["observed_event_rate_mean"], errors="coerce").fillna(0.0).tolist()
        color = palette[idx % len(palette)]
        if {"observed_event_rate_ci95_low", "observed_event_rate_ci95_high"}.issubset(group.columns):
            low = pd.to_numeric(group["observed_event_rate_ci95_low"], errors="coerce").fillna(pd.to_numeric(group["observed_event_rate_mean"], errors="coerce")).tolist()
            high = pd.to_numeric(group["observed_event_rate_ci95_high"], errors="coerce").fillna(pd.to_numeric(group["observed_event_rate_mean"], errors="coerce")).tolist()
            series.extend(_band_series(x, low, high, color=color, stack=f"cal_{idx}"))
        series.append(_line_series(str(label), x, y, color=color, show_symbol=True))
    series.append(
        {
            "name": "Perfect calibration",
            "type": "line",
            "data": [[0, 0], [1, 1]],
            "showSymbol": False,
            "lineStyle": {"color": "#8B97A6", "type": "dashed", "width": 2},
        }
    )
    return {
        "backgroundColor": ACCENT["panel"],
        "title": _echart_title("Do The Risk Percentages Match Reality?"),
        "tooltip": _echart_tooltip(),
        "toolbox": _echart_toolbox(),
        "legend": _echart_legend(top=14),
        "grid": {"left": 72, "right": 28, "top": 72, "bottom": 58},
        "xAxis": _echart_axis("Average predicted risk", min_value=0, max_value=1),
        "yAxis": _echart_axis("Observed event rate", min_value=0, max_value=1),
        "series": series,
    }


def _calibration_hist_echart(calibration: pd.DataFrame, criteria: dict[str, str | None]) -> dict:
    frame = _apply_filters(calibration, criteria)
    if frame.empty:
        return _empty_echart("No probability-range counts are available for this selection.", "How Many Cases Fall Into Each Risk Range")
    frame = frame.copy()
    if {"lower", "upper"}.issubset(frame.columns):
        frame["bin_label"] = frame.apply(lambda row: f"{float(row['lower']):.1f}-{float(row['upper']):.1f}", axis=1)
    else:
        frame["bin_label"] = frame["bin"].astype(str)
    pivot = frame.pivot_table(index="calibration_label", columns="bin_label", values="count_mean", aggfunc="mean").fillna(0.0)
    x_labels = list(pivot.columns)
    y_labels = [str(index) for index in pivot.index]
    values = []
    for y_idx, row_name in enumerate(y_labels):
        for x_idx, column in enumerate(x_labels):
            values.append([x_idx, y_idx, float(pivot.loc[row_name, column])])
    max_count = max((item[2] for item in values), default=1.0)
    return {
        "backgroundColor": ACCENT["panel"],
        "title": _echart_title("How Many Cases Fall Into Each Risk Range"),
        "tooltip": _echart_tooltip(trigger="item"),
        "toolbox": _echart_toolbox(),
        "grid": {"left": 92, "right": 88, "top": 64, "bottom": 68},
        "xAxis": _echart_axis("Risk range bin", axis_type="category", data=x_labels),
        "yAxis": _echart_axis("Probability adjustment", axis_type="category", data=y_labels),
        "visualMap": {
            "min": 0,
            "max": max_count,
            "calculable": True,
            "orient": "vertical",
            "right": 14,
            "top": "middle",
            "inRange": {"color": ["#EEF4FF", "#8CB9FF", "#165DFF"]},
        },
        "series": [{"name": "Mean count", "type": "heatmap", "data": values}],
    }


def _calibration_compare_echart(metrics: pd.DataFrame, criteria: dict[str, str | None]) -> dict:
    frame = _apply_filters(metrics, criteria)
    if frame.empty:
        return _empty_echart("No comparable probability-adjustment rows are available.", "Which Probability Adjustment Gives The Most Trustworthy Risks?")
    metric_specs = [
        ("ece_mean", "Risk Match Error (ECE)"),
        ("brier_mean", "Probability Error (Brier)"),
        ("log_loss_mean", "Confidence Penalty (Log Loss)"),
    ]
    metric_specs = [(column, title) for column, title in metric_specs if column in frame.columns]
    if not metric_specs:
        return _empty_echart("No calibration-quality summary columns are available for this selection.", "Which Probability Adjustment Gives The Most Trustworthy Risks?")
    grids = []
    x_axes = []
    y_axes = []
    titles = [_echart_title("Which Probability Adjustment Gives The Most Trustworthy Risks?")]
    series: list[dict] = []
    for idx, (column, title) in enumerate(metric_specs):
        left = 6 + idx * 31
        ordered = frame.sort_values(column, ascending=True).copy()
        categories = ordered["calibration"].map(_humanize).astype(str).tolist()
        values = pd.to_numeric(ordered[column], errors="coerce").fillna(0.0).tolist()
        grids.append({"left": f"{left}%", "top": 74, "width": "25%", "height": "68%"})
        x_axes.append(_echart_axis("Lower is better", min_value=0, grid_index=idx))
        y_axes.append(_echart_axis(axis_type="category", data=categories, inverse=True, grid_index=idx))
        titles.append(_echart_title(title, left=f"{left}%", top=42))
        series.append(
            {
                "type": "bar",
                "xAxisIndex": idx,
                "yAxisIndex": idx,
                "data": values,
                "barWidth": 10,
                "itemStyle": {"color": _rgba(ACCENT["primary"], 0.32), "borderRadius": [0, 5, 5, 0]},
                "tooltip": {"show": False},
            }
        )
        series.append(
            {
                "type": "scatter",
                "name": title,
                "xAxisIndex": idx,
                "yAxisIndex": idx,
                "data": values,
                "symbolSize": 11,
                "itemStyle": {"color": ACCENT["primary"]},
            }
        )
    return {
        "backgroundColor": ACCENT["panel"],
        "title": titles,
        "tooltip": _echart_tooltip(trigger="item"),
        "toolbox": _echart_toolbox(),
        "grid": grids,
        "xAxis": x_axes,
        "yAxis": y_axes,
        "series": series,
    }


def _showcase_threshold_echart(thresholds: pd.DataFrame, row: pd.Series | None) -> dict:
    if row is None or thresholds.empty:
        return _empty_echart("No decision-rule sweep rows are available for the current selection.", "What Changes When We Raise Or Lower The Decision Rule?")
    subset = _filter_exact(thresholds, row)
    if subset.empty:
        return _empty_echart("No decision-rule sweep rows are available for the current selection.", "What Changes When We Raise Or Lower The Decision Rule?")
    subset = subset[subset["group_value"].astype(str) == "aggregate"] if "group_value" in subset.columns else subset
    overall = subset.groupby("threshold", as_index=False)[["f1_mean", "precision_mean", "recall_mean", "specificity_mean", "demographic_parity_difference_mean", "equalized_odds_difference_mean"]].mean(numeric_only=True)
    x = pd.to_numeric(overall["threshold"], errors="coerce").fillna(0.0).tolist()
    selected_series = pd.to_numeric(subset["selected_threshold_mean"], errors="coerce").dropna()
    selected = float(selected_series.mean()) if not selected_series.empty else None
    series: list[dict] = [
        _line_series("Balanced Detection Score", x, pd.to_numeric(overall["f1_mean"], errors="coerce").fillna(0.0).tolist(), color=ACCENT["primary"], x_axis=0, y_axis=0),
        _line_series("Precision", x, pd.to_numeric(overall["precision_mean"], errors="coerce").fillna(0.0).tolist(), color=ACCENT["warning"], x_axis=0, y_axis=0),
        _line_series("Recall", x, pd.to_numeric(overall["recall_mean"], errors="coerce").fillna(0.0).tolist(), color=ACCENT["secondary"], x_axis=0, y_axis=0),
        _line_series("Specificity", x, pd.to_numeric(overall["specificity_mean"], errors="coerce").fillna(0.0).tolist(), color="#7C4DFF", x_axis=0, y_axis=0, dashed=True),
        _line_series("Positive Rate Gap", x, pd.to_numeric(overall["demographic_parity_difference_mean"], errors="coerce").fillna(0.0).tolist(), color="#00A3BF", x_axis=1, y_axis=1),
        _line_series("Error Gap Between Groups", x, pd.to_numeric(overall["equalized_odds_difference_mean"], errors="coerce").fillna(0.0).tolist(), color="#7C4DFF", x_axis=1, y_axis=1, dashed=True),
    ]
    if selected is not None:
        series.extend([_selected_threshold_mark(selected, x_axis=0, y_axis=0), _selected_threshold_mark(selected, x_axis=1, y_axis=1)])
    return {
        "backgroundColor": ACCENT["panel"],
        "title": [
            _echart_title("What Changes When We Raise Or Lower The Decision Rule?"),
            _echart_title("Detection tradeoff", left="7%", top=42),
            _echart_title("Group-gap tradeoff", left="56%", top=42),
        ],
        "tooltip": _echart_tooltip(),
        "toolbox": _echart_toolbox(),
        "legend": _echart_legend(top=14),
        "grid": [
            {"left": "7%", "top": 76, "width": "38%", "height": "72%"},
            {"left": "56%", "top": 76, "width": "38%", "height": "72%"},
        ],
        "xAxis": [
            _echart_axis("Decision threshold", min_value=0, max_value=1, grid_index=0),
            _echart_axis("Decision threshold", min_value=0, max_value=1, grid_index=1),
        ],
        "yAxis": [
            _echart_axis("Score", min_value=0, max_value=1, grid_index=0),
            _echart_axis("Gap", min_value=0, max_value=1, grid_index=1),
        ],
        "series": series,
    }


def _fairness_selection_echart(fairness: pd.DataFrame, criteria: dict[str, str | None]) -> dict:
    frame = _apply_filters(fairness, criteria)
    if frame.empty:
        return _empty_echart("No group-level decision summary rows are available for this selection.", "How Often Each Group Is Flagged Positive")
    features = [value for value in frame["group_feature"].dropna().astype(str).unique().tolist()]
    if not features:
        return _empty_echart("No subgroup selection-rate rows are available for this selection.", "How Often Each Group Is Flagged Positive")
    grids = []
    x_axes = []
    y_axes = []
    titles = [_echart_title("How Often Each Group Is Flagged Positive")]
    series: list[dict] = []
    palette = {"age_group": ACCENT["primary"], "bmi_category": ACCENT["warning"], "sex": ACCENT["secondary"]}
    width = max(24, int(88 / max(len(features), 1)) - 2)
    for idx, feature in enumerate(features):
        left = 6 + idx * (width + 2)
        subset = frame[frame["group_feature"].astype(str) == feature].copy()
        subset["group_value_name"] = subset["group_value"].map(_humanize_group_value)
        subset = subset.sort_values("selection_rate_mean", ascending=True)
        categories = subset["group_value_name"].astype(str).tolist()
        values = pd.to_numeric(subset["selection_rate_mean"], errors="coerce").fillna(0.0).tolist()
        grids.append({"left": f"{left}%", "top": 76, "width": f"{width}%", "height": "72%"})
        x_axes.append(_echart_axis("Positive prediction rate", min_value=0, max_value=1, grid_index=idx))
        y_axes.append(_echart_axis(axis_type="category", data=categories, grid_index=idx))
        titles.append(_echart_title(_humanize(feature), left=f"{left}%", top=42))
        series.append(
            {
                "type": "bar",
                "xAxisIndex": idx,
                "yAxisIndex": idx,
                "data": values,
                "barWidth": 12,
                "itemStyle": {"color": _rgba(palette.get(feature, ACCENT["primary"]), 0.62), "borderRadius": [0, 5, 5, 0]},
            }
        )
        series.append(
            {
                "type": "scatter",
                "xAxisIndex": idx,
                "yAxisIndex": idx,
                "data": values,
                "symbolSize": 10,
                "itemStyle": {"color": palette.get(feature, ACCENT["primary"])},
                "tooltip": {"show": False},
            }
        )
    return {
        "backgroundColor": ACCENT["panel"],
        "title": titles,
        "tooltip": _echart_tooltip(trigger="item"),
        "toolbox": _echart_toolbox(),
        "grid": grids,
        "xAxis": x_axes,
        "yAxis": y_axes,
        "series": series,
    }


def _fairness_gap_echart(fairness: pd.DataFrame, criteria: dict[str, str | None]) -> dict:
    frame = _apply_filters(fairness, criteria)
    if frame.empty:
        return _empty_echart("No group-gap rows are available for this selection.", "How Different The Results Are Between Groups")
    gap = frame.groupby("group_feature", dropna=False)[["demographic_parity_difference_mean", "equalized_odds_difference_mean"]].mean(numeric_only=True).reset_index()
    x_labels = [_humanize(value) for value in gap["group_feature"].astype(str).tolist()]
    z = []
    for idx, value in enumerate(pd.to_numeric(gap["demographic_parity_difference_mean"], errors="coerce").fillna(0.0).tolist()):
        z.append([idx, 0, float(value)])
    for idx, value in enumerate(pd.to_numeric(gap["equalized_odds_difference_mean"], errors="coerce").fillna(0.0).tolist()):
        z.append([idx, 1, float(value)])
    max_gap = max((item[2] for item in z), default=1.0)
    return {
        "backgroundColor": ACCENT["panel"],
        "title": _echart_title("How Different The Results Are Between Groups"),
        "tooltip": _echart_tooltip(trigger="item"),
        "toolbox": _echart_toolbox(),
        "grid": {"left": 92, "right": 84, "top": 64, "bottom": 54},
        "xAxis": _echart_axis(axis_type="category", data=x_labels),
        "yAxis": _echart_axis(axis_type="category", data=["Positive Rate Gap", "Error Gap Between Groups"]),
        "visualMap": {
            "min": 0,
            "max": max_gap,
            "calculable": True,
            "orient": "vertical",
            "right": 14,
            "top": "middle",
            "inRange": {"color": ["#FFF3E8", "#FFB46B", "#D65A00"]},
        },
        "series": [{"type": "heatmap", "data": z}],
    }


def _fairness_threshold_echart(thresholds: pd.DataFrame, criteria: dict[str, str | None]) -> dict:
    frame = _apply_filters(thresholds, criteria)
    if frame.empty:
        return _empty_echart("No decision-rule sweep rows are available for this selection.", "What Changes For Each Group When We Raise Or Lower The Decision Rule?")
    frame = frame[frame["group_value"].astype(str) == "aggregate"] if "group_value" in frame.columns else frame
    overall = frame.groupby("threshold", as_index=False)[["f1_mean", "specificity_mean"]].mean(numeric_only=True)
    by_feature = frame.groupby(["group_feature", "threshold"], as_index=False)[["selection_rate_mean", "demographic_parity_difference_mean", "equalized_odds_difference_mean"]].mean(numeric_only=True)
    selected_values = pd.to_numeric(frame["selected_threshold_mean"], errors="coerce").dropna()
    selected = float(selected_values.mean()) if not selected_values.empty else None
    palette = {"age_group": ACCENT["primary"], "bmi_category": ACCENT["warning"], "sex": ACCENT["secondary"]}
    series: list[dict] = []
    x_main = pd.to_numeric(overall["threshold"], errors="coerce").fillna(0.0).tolist()
    series.append(_line_series("Balanced Detection Score", x_main, pd.to_numeric(overall["f1_mean"], errors="coerce").fillna(0.0).tolist(), color=ACCENT["primary"], x_axis=0, y_axis=0))
    series.append(_line_series("Specificity", x_main, pd.to_numeric(overall["specificity_mean"], errors="coerce").fillna(0.0).tolist(), color="#7C4DFF", x_axis=0, y_axis=0, dashed=True))
    for feature, subset in by_feature.groupby("group_feature", dropna=False):
        x = pd.to_numeric(subset["threshold"], errors="coerce").fillna(0.0).tolist()
        label = _humanize(feature)
        color = palette.get(str(feature), ACCENT["muted"])
        series.append(_line_series(label, x, pd.to_numeric(subset["selection_rate_mean"], errors="coerce").fillna(0.0).tolist(), color=color, x_axis=1, y_axis=1))
        series.append(_line_series(label, x, pd.to_numeric(subset["demographic_parity_difference_mean"], errors="coerce").fillna(0.0).tolist(), color=color, x_axis=2, y_axis=2))
        series.append(_line_series(label, x, pd.to_numeric(subset["equalized_odds_difference_mean"], errors="coerce").fillna(0.0).tolist(), color=color, x_axis=3, y_axis=3, dashed=True))
    if selected is not None:
        for idx in range(4):
            series.append(_selected_threshold_mark(selected, x_axis=idx, y_axis=idx))
    return {
        "backgroundColor": ACCENT["panel"],
        "title": [
            _echart_title("What Changes For Each Group When We Raise Or Lower The Decision Rule?"),
            _echart_title("Overall detection tradeoff", left="6%", top=42),
            _echart_title("Positive prediction rate", left="56%", top=42),
            _echart_title("Positive rate gap", left="6%", top=352),
            _echart_title("Error gap between groups", left="56%", top=352),
        ],
        "tooltip": _echart_tooltip(),
        "toolbox": _echart_toolbox(),
        "legend": _echart_legend(top=14),
        "grid": [
            {"left": "6%", "top": 76, "width": "38%", "height": "24%"},
            {"left": "56%", "top": 76, "width": "38%", "height": "24%"},
            {"left": "6%", "top": 390, "width": "38%", "height": "24%"},
            {"left": "56%", "top": 390, "width": "38%", "height": "24%"},
        ],
        "xAxis": [
            _echart_axis("Decision threshold", min_value=0, max_value=1, grid_index=0),
            _echart_axis("Decision threshold", min_value=0, max_value=1, grid_index=1),
            _echart_axis("Decision threshold", min_value=0, max_value=1, grid_index=2),
            _echart_axis("Decision threshold", min_value=0, max_value=1, grid_index=3),
        ],
        "yAxis": [
            _echart_axis("Score", min_value=0, max_value=1, grid_index=0),
            _echart_axis("Rate", min_value=0, max_value=1, grid_index=1),
            _echart_axis("Gap", min_value=0, max_value=1, grid_index=2),
            _echart_axis("Gap", min_value=0, max_value=1, grid_index=3),
        ],
        "series": series,
    }


def _training_metrics_echart(rounds: pd.DataFrame, criteria: dict[str, str | None]) -> dict:
    frame = _apply_filters(rounds, {**criteria, "run_type": "federated"})
    if frame.empty:
        return _empty_echart("No federated round-by-round rows are available for this selection.", "How Federated Training Improves Round By Round")
    frame = frame.groupby("round", as_index=False).mean(numeric_only=True).sort_values("round")
    x = pd.to_numeric(frame["round"], errors="coerce").fillna(0.0).tolist()
    series: list[dict] = []
    if {"global_eval_auc_ci95_low", "global_eval_auc_ci95_high"}.issubset(frame.columns):
        series.extend(
            _band_series(
                x,
                pd.to_numeric(frame["global_eval_auc_ci95_low"], errors="coerce").fillna(pd.to_numeric(frame["global_eval_auc_mean"], errors="coerce")).tolist(),
                pd.to_numeric(frame["global_eval_auc_ci95_high"], errors="coerce").fillna(pd.to_numeric(frame["global_eval_auc_mean"], errors="coerce")).tolist(),
                color=ACCENT["primary"],
                stack="auc_ci",
                x_axis=0,
                y_axis=0,
            )
        )
    series.append(_line_series("Ranking Quality (AUROC)", x, pd.to_numeric(frame["global_eval_auc_mean"], errors="coerce").fillna(0.0).tolist(), color=ACCENT["primary"], x_axis=0, y_axis=0, show_symbol=True))
    if "global_eval_f1_mean" in frame.columns:
        series.append(_line_series("Balanced Detection Score (F1)", x, pd.to_numeric(frame["global_eval_f1_mean"], errors="coerce").fillna(0.0).tolist(), color=ACCENT["secondary"], x_axis=0, y_axis=0, show_symbol=True))
    if "global_eval_ece_mean" in frame.columns:
        if {"global_eval_ece_ci95_low", "global_eval_ece_ci95_high"}.issubset(frame.columns):
            series.extend(
                _band_series(
                    x,
                    pd.to_numeric(frame["global_eval_ece_ci95_low"], errors="coerce").fillna(pd.to_numeric(frame["global_eval_ece_mean"], errors="coerce")).tolist(),
                    pd.to_numeric(frame["global_eval_ece_ci95_high"], errors="coerce").fillna(pd.to_numeric(frame["global_eval_ece_mean"], errors="coerce")).tolist(),
                    color=ACCENT["warning"],
                    stack="ece_ci",
                    x_axis=1,
                    y_axis=1,
                )
            )
        series.append(_line_series("Risk Match Error (ECE)", x, pd.to_numeric(frame["global_eval_ece_mean"], errors="coerce").fillna(0.0).tolist(), color=ACCENT["warning"], x_axis=1, y_axis=1, show_symbol=True))
    return {
        "backgroundColor": ACCENT["panel"],
        "title": [
            _echart_title("How Federated Training Improves Round By Round"),
            _echart_title("Model quality over rounds", left="6%", top=42),
            _echart_title("Probability quality over rounds", left="6%", top=322),
        ],
        "tooltip": _echart_tooltip(),
        "toolbox": _echart_toolbox(),
        "legend": _echart_legend(top=14),
        "grid": [
            {"left": "6%", "top": 76, "width": "88%", "height": "25%"},
            {"left": "6%", "top": 356, "width": "88%", "height": "25%"},
        ],
        "xAxis": [
            _echart_axis(grid_index=0),
            _echart_axis("Training round", grid_index=1),
        ],
        "yAxis": [
            _echart_axis("Score", min_value=0, max_value=1, grid_index=0),
            _echart_axis("Error", min_value=0, max_value=1, grid_index=1),
        ],
        "series": series,
    }


def _communication_echart(rounds: pd.DataFrame, criteria: dict[str, str | None]) -> dict:
    frame = _apply_filters(rounds, {**criteria, "run_type": "federated"})
    if frame.empty or "cumulative_communication_bytes_mean" not in frame.columns:
        return _empty_echart("No communication rows are available for this selection.", "How Much Data Is Sent During Training?")
    frame = frame.groupby("round", as_index=False).mean(numeric_only=True).sort_values("round")
    x = pd.to_numeric(frame["round"], errors="coerce").fillna(0.0).tolist()
    round_bytes = pd.to_numeric(frame.get("round_communication_bytes_mean"), errors="coerce").fillna(0.0).tolist()
    cumulative = pd.to_numeric(frame["cumulative_communication_bytes_mean"], errors="coerce").fillna(0.0).tolist()
    return {
        "backgroundColor": ACCENT["panel"],
        "title": _echart_title("How Much Data Is Sent During Training?"),
        "tooltip": _echart_tooltip(),
        "toolbox": _echart_toolbox(),
        "legend": _echart_legend(top=14),
        "grid": {"left": 72, "right": 76, "top": 70, "bottom": 58},
        "xAxis": _echart_axis("Training round", min_value=min(x) if x else 0, max_value=max(x) if x else 1),
        "yAxis": [
            _echart_axis("Bytes sent this round", min_value=0),
            _echart_axis("Cumulative bytes", min_value=0),
        ],
        "series": [
            {
                "name": "Bytes sent this round",
                "type": "bar",
                "data": [[a, b] for a, b in zip(x, round_bytes)],
                "itemStyle": {"color": _rgba(ACCENT["primary"], 0.38), "borderRadius": [4, 4, 0, 0]},
                "yAxisIndex": 0,
            },
            {
                "name": "Cumulative bytes",
                "type": "line",
                "data": [[a, b] for a, b in zip(x, cumulative)],
                "lineStyle": {"color": ACCENT["primary"], "width": 3},
                "itemStyle": {"color": ACCENT["primary"]},
                "showSymbol": True,
                "yAxisIndex": 1,
            },
        ],
    }


def _study_comparison_echart(metrics: pd.DataFrame, criteria: dict[str, str | None]) -> dict:
    frame = _apply_filters(metrics, criteria)
    if frame.empty:
        return _empty_echart("No broad comparison rows are available for this selection.", "How Centralized, Local-Only, and Federated Compare")
    frame = frame.copy()
    frame["run_type_name"] = frame["run_type"].map(_humanize)
    categories = [name for name in ["Centralized", "Local-Only", "Average Local-Only", "Federated"] if name in frame["run_type_name"].unique().tolist()]
    box_data = []
    mean_points = []
    valid_categories = []
    for idx, category in enumerate(categories):
        values = pd.to_numeric(frame.loc[frame["run_type_name"] == category, "f1_mean"], errors="coerce").dropna().tolist()
        if not values:
            continue
        stats = np.percentile(values, [0, 25, 50, 75, 100])
        box_data.append(stats.tolist())
        mean_points.append([len(valid_categories), float(np.mean(values))])
        valid_categories.append(category)
    return {
        "backgroundColor": ACCENT["panel"],
        "title": _echart_title("How Centralized, Local-Only, and Federated Compare"),
        "tooltip": _echart_tooltip(trigger="item"),
        "toolbox": _echart_toolbox(),
        "grid": {"left": 72, "right": 28, "top": 64, "bottom": 58},
        "xAxis": _echart_axis(axis_type="category", data=valid_categories),
        "yAxis": _echart_axis("Balanced Detection Score (F1)", min_value=0, max_value=1),
        "series": [
            {
                "name": "Distribution",
                "type": "boxplot",
                "data": box_data,
                "itemStyle": {"color": _rgba(ACCENT["primary"], 0.22), "borderColor": ACCENT["primary"]},
            },
            {
                "name": "Mean",
                "type": "scatter",
                "data": mean_points,
                "symbolSize": 12,
                "itemStyle": {"color": ACCENT["ink"]},
            },
        ],
    }


def _performance_figure(metrics: pd.DataFrame, criteria: dict[str, str | None]) -> go.Figure:
    frame = _apply_filters(metrics, criteria)
    if frame.empty:
        return _empty("How The Trained Models Compare", "This combination was not trained in the current study.")
    frame = frame.copy()
    frame["display_name"] = frame["model_label"].fillna(frame["model"].astype(str)) + " | " + frame["run_type_label"].fillna(frame["run_type"].astype(str))
    frame["error_plus"] = frame["roc_auc_ci95_high"] - frame["roc_auc_mean"]
    frame["error_minus"] = frame["roc_auc_mean"] - frame["roc_auc_ci95_low"]
    if {"f1_ci95_high", "f1_ci95_low"}.issubset(frame.columns):
        frame["f1_error_plus"] = frame["f1_ci95_high"] - frame["f1_mean"]
        frame["f1_error_minus"] = frame["f1_mean"] - frame["f1_ci95_low"]
    hover_columns = {column: True for column in ["f1_mean", "pr_auc_mean", "ece_mean"] if column in frame.columns}
    size_column = "pr_auc_mean" if "pr_auc_mean" in frame.columns and frame["pr_auc_mean"].notna().any() else None
    fig = px.scatter(
        frame,
        x="roc_auc_mean",
        y="f1_mean",
        color="run_type_label",
        symbol="model_label",
        size=size_column,
        size_max=22,
        error_x="error_plus",
        error_x_minus="error_minus",
        error_y="f1_error_plus" if "f1_error_plus" in frame.columns else None,
        error_y_minus="f1_error_minus" if "f1_error_minus" in frame.columns else None,
        title="How The Trained Models Compare",
        color_discrete_map=RUN_TYPE_COLORS,
        hover_data=hover_columns,
        hover_name="display_name",
    )
    fig.update_traces(marker={"line": {"color": "#FFFFFF", "width": 1.2}, "opacity": 0.86})
    fig.update_layout(legend_title_text="")
    _apply_figure_style(fig, "How The Trained Models Compare", "Average ranking quality", "Average balanced detection score", height=470)
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


def _performance_note(metrics: pd.DataFrame, criteria: dict[str, str | None]) -> str:
    frame = _apply_filters(metrics, criteria)
    if frame.empty:
        return "This combination was not trained in the current study. Try a different run type, model family, or algorithm."
    row = frame.sort_values(["f1_mean", "roc_auc_mean", "ece_mean"], ascending=[False, False, True]).iloc[0]
    return (
        f"Current best match: {_humanize(row.get('run_type', ''))} / {_humanize(row.get('model', ''))} / "
        f"{_humanize(row.get('calibration', ''))} with ranking quality {_fmt(row.get('roc_auc_mean'))}, "
        f"balanced detection {_fmt(row.get('f1_mean'))}, positive-case ranking {_fmt(row.get('pr_auc_mean'))}, and risk-match error {_fmt(row.get('ece_mean'))}."
    )


def _calibration_curve(calibration: pd.DataFrame, criteria: dict[str, str | None]) -> go.Figure:
    frame = _apply_filters(calibration, criteria)
    if frame.empty:
        return _empty("Do The Risk Percentages Match Reality?", "No aggregated probability-adjustment bins are available for this selection.")
    fig = go.Figure()
    for label, group in frame.groupby("calibration_label", dropna=False):
        group = group.sort_values("avg_predicted_risk_mean")
        color = None
        fig.add_trace(
            go.Scatter(
                x=group["avg_predicted_risk_mean"],
                y=group["observed_event_rate_mean"],
                mode="lines+markers",
                name=str(label),
                line={"width": 3},
                marker={"size": 7},
            )
        )
        color = fig.data[-1].line.color
        if {"observed_event_rate_ci95_low", "observed_event_rate_ci95_high"}.issubset(group.columns):
            fig.add_trace(go.Scatter(x=group["avg_predicted_risk_mean"], y=group["observed_event_rate_ci95_high"], mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False))
            fig.add_trace(
                go.Scatter(
                    x=group["avg_predicted_risk_mean"],
                    y=group["observed_event_rate_ci95_low"],
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=_rgba(color or ACCENT["primary"], 0.12),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash", "color": "#8B97A6"}, name="Perfect calibration"))
    _apply_figure_style(fig, "Do The Risk Percentages Match Reality?", "Average predicted risk", "Observed event rate", height=430)
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


def _calibration_hist(calibration: pd.DataFrame, criteria: dict[str, str | None]) -> go.Figure:
    frame = _apply_filters(calibration, criteria)
    if frame.empty:
        return _empty("How Many Cases Fall Into Each Risk Range", "No probability-range counts are available for this selection.")
    frame = frame.copy()
    if {"lower", "upper"}.issubset(frame.columns):
        frame["bin_label"] = frame.apply(lambda row: f"{float(row['lower']):.1f}-{float(row['upper']):.1f}", axis=1)
    else:
        frame["bin_label"] = frame["bin"].astype(str)
    pivot = frame.pivot_table(index="calibration_label", columns="bin_label", values="count_mean", aggfunc="mean").fillna(0.0)
    z = np.log10(pivot.to_numpy() + 1.0)
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                x=list(pivot.columns),
                y=list(pivot.index),
                colorscale="Blues",
                colorbar={"title": "log10(count + 1)"},
                customdata=pivot.to_numpy(),
                hovertemplate="Probability adjustment: %{y}<br>Risk bin: %{x}<br>Mean count: %{customdata:.0f}<extra></extra>",
            )
        ]
    )
    _apply_figure_style(fig, "How Many Cases Fall Into Each Risk Range", "Risk range bin", "Probability adjustment", height=390)
    return fig


def _calibration_compare(metrics: pd.DataFrame, criteria: dict[str, str | None]) -> go.Figure:
    frame = _apply_filters(metrics, criteria)
    if frame.empty:
        return _empty("Which Probability Adjustment Gives The Most Trustworthy Risks?", "No comparable probability-adjustment rows are available.")
    metric_specs = [
        ("ece_mean", "Risk Match Error"),
        ("brier_mean", "Probability Error"),
        ("log_loss_mean", "Confidence Penalty"),
    ]
    metric_specs = [(column, title) for column, title in metric_specs if column in frame.columns]
    fig = make_subplots(rows=1, cols=len(metric_specs), subplot_titles=[title for _, title in metric_specs], horizontal_spacing=0.08)
    for idx, (column, title) in enumerate(metric_specs, start=1):
        ordered = frame.sort_values(column, ascending=True)
        fig.add_trace(
            go.Scatter(
                x=ordered[column],
                y=ordered["calibration_label"],
                mode="markers",
                marker={"color": ACCENT["primary"], "size": 10},
                showlegend=False,
                hovertemplate="Probability adjustment: %{y}<br>" + title + ": %{x:.3f}<extra></extra>",
            ),
            row=1,
            col=idx,
        )
        fig.update_xaxes(title="Lower is better", row=1, col=idx)
        if idx == 1:
            fig.update_yaxes(title="Probability adjustment", row=1, col=idx)
    _apply_figure_style(fig, "Which Probability Adjustment Gives The Most Trustworthy Risks?", None, None, height=400, legend=False)
    return fig


def _xai_bar(shap: pd.DataFrame, criteria: dict[str, str | None]) -> go.Figure:
    frame = _apply_filters(shap, criteria)
    if frame.empty and criteria["calibration"] is not None:
        relaxed = dict(criteria)
        relaxed["calibration"] = None
        frame = _apply_filters(shap, relaxed)
    if frame.empty:
        return _empty("Which Features Matter Most?", "No feature-importance summary rows are available for this selection.")
    if "client_id" in frame.columns:
        global_rows = frame[frame["client_id"].astype(str) == "global"]
        if not global_rows.empty:
            frame = global_rows
    top = frame.sort_values(["rank_mean", "mean_abs_shap_mean"], ascending=[True, False]).head(12).copy()
    top["error_plus"] = top["mean_abs_shap_ci95_high"] - top["mean_abs_shap_mean"]
    top["error_minus"] = top["mean_abs_shap_mean"] - top["mean_abs_shap_ci95_low"]
    fig = px.bar(
        top.sort_values("mean_abs_shap_mean"),
        x="mean_abs_shap_mean",
        y="feature",
        orientation="h",
        error_x="error_plus",
        error_x_minus="error_minus",
        title="Which Features Matter Most?",
        color_discrete_sequence=[ACCENT["primary"]],
    )
    _apply_figure_style(fig, "Which Features Matter Most?", "Average feature impact", "", height=430, legend=False)
    return fig


def _local_explanation_figure(local_explanations: pd.DataFrame, criteria: dict[str, str | None]) -> go.Figure:
    frame = _apply_filters(local_explanations, criteria)
    if frame.empty and criteria["calibration"] is not None:
        relaxed = dict(criteria)
        relaxed["calibration"] = None
        frame = _apply_filters(local_explanations, relaxed)
    if frame.empty:
        return _empty("What Pushed This One Prediction Up Or Down?", "No saved example-level explanation rows are available for this selection.")
    if "client_id" in frame.columns:
        global_rows = frame[frame["client_id"].astype(str) == "global"]
        if not global_rows.empty:
            frame = global_rows
    top = frame.sort_values(["rank_mean", "contribution_mean"], ascending=[True, False]).head(10).copy()
    top["direction"] = np.where(pd.to_numeric(top["contribution_mean"], errors="coerce") >= 0, "Increase", "Decrease")
    fig = px.bar(
        top.sort_values("contribution_mean"),
        x="contribution_mean",
        y="feature",
        color="direction",
        orientation="h",
        title="What Pushed This One Prediction Up Or Down?",
        color_discrete_map={"Increase": ACCENT["secondary"], "Decrease": ACCENT["warning"]},
    )
    _apply_figure_style(fig, "What Pushed This One Prediction Up Or Down?", "Average push on predicted risk", "", height=430)
    return fig


def _stability_line(stability: pd.DataFrame, criteria: dict[str, str | None]) -> go.Figure:
    frame = _apply_filters(stability, {**criteria, "run_type": "federated"})
    if frame.empty:
        return _empty("Do The Feature Explanations Stay Consistent Over Time?", "No federated explanation-stability rows are available for this selection.")
    round_rows = frame.dropna(subset=["round"]) if "round" in frame.columns else pd.DataFrame()
    round_rows = round_rows.dropna(subset=["spearman_top_feature_stability_mean"]) if not round_rows.empty else round_rows
    if round_rows.empty:
        return _empty("Do The Feature Explanations Stay Consistent Over Time?", "No round-wise explanation-stability values are available.")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=round_rows["round"],
            y=round_rows["spearman_top_feature_stability_mean"],
            mode="lines+markers",
            line={"color": ACCENT["primary"], "width": 3},
            name="Mean stability",
        )
    )
    fig.add_trace(go.Scatter(x=round_rows["round"], y=round_rows["spearman_top_feature_stability_ci95_high"], mode="lines", line={"width": 0}, showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=round_rows["round"],
            y=round_rows["spearman_top_feature_stability_ci95_low"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=_rgba(ACCENT["primary"], 0.14),
            showlegend=False,
        )
    )
    _apply_figure_style(fig, "Do The Feature Explanations Stay Consistent Over Time?", "Training round", "Average explanation stability", height=420)
    fig.update_yaxes(range=[0.0, 1.02])
    return fig


def _stability_heatmap(stability: pd.DataFrame, criteria: dict[str, str | None]) -> go.Figure:
    frame = _apply_filters(stability, {**criteria, "run_type": "federated"})
    if frame.empty:
        return _empty("Do Different Sites Focus On Similar Features?", "No cross-site feature-overlap rows are available for this selection.")
    overlap = frame.dropna(subset=["top_k_overlap_mean"]) if "top_k_overlap_mean" in frame.columns else pd.DataFrame()
    if overlap.empty:
        return _empty("Do Different Sites Focus On Similar Features?", "No cross-site feature-overlap values are available.")
    matrix = overlap.pivot_table(index="client_left", columns="client_right", values="top_k_overlap_mean", aggfunc="mean").fillna(0.0)
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=matrix.to_numpy(),
                x=[str(column) for column in matrix.columns],
                y=[str(index) for index in matrix.index],
                colorscale="Blues",
                colorbar={"title": "Overlap"},
            )
        ]
    )
    _apply_figure_style(fig, "Do Different Sites Focus On Similar Features?", "Site on the right", "Site on the left", height=420, legend=False)
    return fig


def _fairness_selection(fairness: pd.DataFrame, criteria: dict[str, str | None]) -> go.Figure:
    frame = _apply_filters(fairness, criteria)
    if frame.empty:
        return _empty("How Often Each Group Is Flagged Positive", "No group-level decision summary rows are available for this selection.")
    frame = frame.copy()
    frame["error_plus"] = frame["selection_rate_ci95_high"] - frame["selection_rate_mean"]
    frame["error_minus"] = frame["selection_rate_mean"] - frame["selection_rate_ci95_low"]
    features = [value for value in frame["group_feature"].dropna().astype(str).unique().tolist()]
    fig = make_subplots(rows=1, cols=max(len(features), 1), subplot_titles=[_humanize(value) for value in features], horizontal_spacing=0.08)
    color_map = {"age_group": ACCENT["primary"], "bmi_category": ACCENT["warning"], "sex": ACCENT["secondary"]}
    for idx, feature in enumerate(features, start=1):
        subset = frame[frame["group_feature"].astype(str) == feature].sort_values("selection_rate_mean", ascending=True)
        fig.add_trace(
            go.Scatter(
                x=subset["selection_rate_mean"],
                y=subset["group_value"],
                mode="markers",
                marker={"size": 12, "color": color_map.get(feature, ACCENT["primary"])},
                error_x={"type": "data", "array": subset["error_plus"], "arrayminus": subset["error_minus"]},
                showlegend=False,
                hovertemplate="Group: %{y}<br>Positive prediction rate: %{x:.3f}<extra></extra>",
            ),
            row=1,
            col=idx,
        )
        fig.update_xaxes(range=[0, 1], title="Positive prediction rate", row=1, col=idx)
        if idx == 1:
            fig.update_yaxes(title="Group", row=1, col=idx)
    _apply_figure_style(fig, "How Often Each Group Is Flagged Positive", None, None, height=430, legend=False)
    return fig


def _fairness_gap(fairness: pd.DataFrame, criteria: dict[str, str | None]) -> go.Figure:
    frame = _apply_filters(fairness, criteria)
    if frame.empty:
        return _empty("How Different The Results Are Between Groups", "No group-gap rows are available for this selection.")
    gap = frame.groupby("group_feature", dropna=False)[["demographic_parity_difference_mean", "equalized_odds_difference_mean"]].mean().reset_index()
    display = pd.DataFrame(
        {
            "group_feature": gap["group_feature"].map(_humanize),
            "Positive Rate Gap": gap["demographic_parity_difference_mean"],
            "Error Gap Between Groups": gap["equalized_odds_difference_mean"],
        }
    ).set_index("group_feature")
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=display.to_numpy().T,
                x=list(display.index),
                y=list(display.columns),
                text=np.round(display.to_numpy().T, 3),
                texttemplate="%{text}",
                colorscale="OrRd",
                colorbar={"title": "Gap size"},
                hovertemplate="Group check: %{x}<br>Metric: %{y}<br>Mean gap: %{z:.3f}<extra></extra>",
            )
        ]
    )
    _apply_figure_style(fig, "How Different The Results Are Between Groups", "", "", height=360)
    return fig


def _fairness_threshold(thresholds: pd.DataFrame, criteria: dict[str, str | None]) -> go.Figure:
    frame = _apply_filters(thresholds, criteria)
    if frame.empty:
        return _empty("What Changes For Each Group When We Raise Or Lower The Decision Rule?", "No decision-rule sweep rows are available for this selection.")
    frame = frame[frame["group_value"].astype(str) == "aggregate"] if "group_value" in frame.columns else frame
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Balanced detection and false-positive control",
            "Positive prediction rate by group check",
            "Positive rate gap by group check",
            "Error gap between groups",
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.16,
    )
    palette = {"age_group": ACCENT["primary"], "bmi_category": ACCENT["warning"], "sex": ACCENT["secondary"]}
    overall = frame.groupby("threshold", as_index=False)[["f1_mean", "specificity_mean"]].mean(numeric_only=True)
    if "f1_mean" in overall.columns:
        fig.add_trace(go.Scatter(x=overall["threshold"], y=overall["f1_mean"], mode="lines", name="Balanced Detection Score", line={"color": ACCENT["primary"], "width": 3}), row=1, col=1)
    if "specificity_mean" in overall.columns:
        fig.add_trace(go.Scatter(x=overall["threshold"], y=overall["specificity_mean"], mode="lines", name="Specificity", line={"color": ACCENT["warning"], "width": 3, "dash": "dash"}), row=1, col=1)
    for feature, subset in frame.groupby("group_feature", dropna=False):
        color = palette.get(str(feature), ACCENT["muted"])
        ordered = subset.sort_values("threshold")
        fig.add_trace(go.Scatter(x=ordered["threshold"], y=ordered["selection_rate_mean"], mode="lines", name=f"{_humanize(feature)} positive rate", legendgroup=str(feature), line={"color": color, "width": 2}), row=1, col=2)
        fig.add_trace(go.Scatter(x=ordered["threshold"], y=ordered["demographic_parity_difference_mean"], mode="lines", name=f"{_humanize(feature)} positive gap", legendgroup=str(feature), line={"color": color, "width": 2}, showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=ordered["threshold"], y=ordered["equalized_odds_difference_mean"], mode="lines", name=f"{_humanize(feature)} error gap", legendgroup=str(feature), line={"color": color, "width": 2, "dash": "dot"}, showlegend=False), row=2, col=2)
    selected = pd.to_numeric(frame["selected_threshold_mean"], errors="coerce").dropna()
    if not selected.empty:
        chosen = float(selected.mean())
        for row_idx in (1, 2):
            for col_idx in (1, 2):
                fig.add_vline(x=chosen, line_dash="dash", line_color="#6B7280", row=row_idx, col=col_idx)
    fig.update_xaxes(title="Decision threshold", range=[0, 1], row=1, col=1)
    fig.update_xaxes(title="Decision threshold", range=[0, 1], row=1, col=2)
    fig.update_xaxes(title="Decision threshold", range=[0, 1], row=2, col=1)
    fig.update_xaxes(title="Decision threshold", range=[0, 1], row=2, col=2)
    fig.update_yaxes(title="Score", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title="Rate", range=[0, 1], row=1, col=2)
    fig.update_yaxes(title="Gap", row=2, col=1)
    fig.update_yaxes(title="Gap", row=2, col=2)
    _apply_figure_style(fig, "What Changes For Each Group When We Raise Or Lower The Decision Rule?", None, None, height=620)
    return fig


def _training_metrics(rounds: pd.DataFrame, criteria: dict[str, str | None]) -> go.Figure:
    frame = _apply_filters(rounds, {**criteria, "run_type": "federated"})
    if frame.empty:
        return _empty("How Federated Training Improves Round By Round", "No federated round-by-round rows are available for this selection.")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14, subplot_titles=["Ranking quality and balanced detection", "Risk match error"])
    fig.add_trace(go.Scatter(x=frame["round"], y=frame["global_eval_auc_mean"], mode="lines+markers", name="Ranking Quality", line={"color": ACCENT["primary"], "width": 3}), row=1, col=1)
    if {"global_eval_auc_ci95_low", "global_eval_auc_ci95_high"}.issubset(frame.columns):
        fig.add_trace(go.Scatter(x=frame["round"], y=frame["global_eval_auc_ci95_high"], mode="lines", line={"width": 0}, showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=frame["round"], y=frame["global_eval_auc_ci95_low"], mode="lines", line={"width": 0}, fill="tonexty", fillcolor=_rgba(ACCENT["primary"], 0.14), showlegend=False), row=1, col=1)
    if "global_eval_f1_mean" in frame.columns:
        fig.add_trace(go.Scatter(x=frame["round"], y=frame["global_eval_f1_mean"], mode="lines+markers", name="Balanced Detection", line={"color": ACCENT["secondary"], "width": 2.5}), row=1, col=1)
    if "global_eval_ece_mean" in frame.columns:
        fig.add_trace(go.Scatter(x=frame["round"], y=frame["global_eval_ece_mean"], mode="lines+markers", name="Risk Match Error", line={"color": ACCENT["warning"], "width": 2.5}), row=2, col=1)
        if {"global_eval_ece_ci95_low", "global_eval_ece_ci95_high"}.issubset(frame.columns):
            fig.add_trace(go.Scatter(x=frame["round"], y=frame["global_eval_ece_ci95_high"], mode="lines", line={"width": 0}, showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=frame["round"], y=frame["global_eval_ece_ci95_low"], mode="lines", line={"width": 0}, fill="tonexty", fillcolor=_rgba(ACCENT["warning"], 0.14), showlegend=False), row=2, col=1)
    fig.update_xaxes(title="Training round", row=2, col=1)
    fig.update_yaxes(title="Score", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title="Error", row=2, col=1)
    _apply_figure_style(fig, "How Federated Training Improves Round By Round", None, None, height=560)
    return fig


def _communication_figure(rounds: pd.DataFrame, criteria: dict[str, str | None]) -> go.Figure:
    frame = _apply_filters(rounds, {**criteria, "run_type": "federated"})
    if frame.empty or "cumulative_communication_bytes_mean" not in frame.columns:
        return _empty("How Much Data Is Sent During Training?", "No communication rows are available for this selection.")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if "round_communication_bytes_mean" in frame.columns:
        fig.add_trace(
            go.Bar(
                x=frame["round"],
                y=frame["round_communication_bytes_mean"],
                name="Bytes sent this round",
                marker_color=_rgba(ACCENT["primary"], 0.35),
            ),
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=frame["round"],
            y=frame["cumulative_communication_bytes_mean"],
            mode="lines+markers",
            name="Cumulative bytes",
            line={"color": ACCENT["primary"], "width": 3},
        ),
        secondary_y=True,
    )
    if {"cumulative_communication_bytes_ci95_low", "cumulative_communication_bytes_ci95_high"}.issubset(frame.columns):
        fig.add_trace(go.Scatter(x=frame["round"], y=frame["cumulative_communication_bytes_ci95_high"], mode="lines", line={"width": 0}, showlegend=False), secondary_y=True)
        fig.add_trace(go.Scatter(x=frame["round"], y=frame["cumulative_communication_bytes_ci95_low"], mode="lines", line={"width": 0}, fill="tonexty", fillcolor=_rgba(ACCENT["primary"], 0.14), showlegend=False), secondary_y=True)
    fig.update_xaxes(title="Training round")
    fig.update_yaxes(title_text="Bytes sent this round", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative bytes", secondary_y=True)
    _apply_figure_style(fig, "How Much Data Is Sent During Training?", None, None, height=430)
    return fig


def _study_comparison(metrics: pd.DataFrame, criteria: dict[str, str | None]) -> go.Figure:
    frame = _apply_filters(metrics, criteria)
    if frame.empty:
        return _empty("How Centralized, Local-Only, and Federated Compare", "No broad comparison rows are available for this selection.")
    hover_columns = {column: True for column in ["roc_auc_mean", "pr_auc_mean", "ece_mean", "model_label"] if column in frame.columns}
    fig = px.violin(
        frame,
        x="run_type_label",
        y="f1_mean",
        color="run_type_label",
        box=True,
        points=False,
        hover_data=hover_columns,
        title="How Centralized, Local-Only, and Federated Compare",
        color_discrete_map=RUN_TYPE_COLORS,
    )
    means = frame.groupby("run_type_label", as_index=False)["f1_mean"].mean()
    fig.add_trace(
        go.Scatter(
            x=means["run_type_label"],
            y=means["f1_mean"],
            mode="markers+text",
            text=[f"{value:.3f}" for value in means["f1_mean"]],
            textposition="top center",
            marker={"color": ACCENT["ink"], "size": 10, "symbol": "diamond"},
            name="Mean",
        )
    )
    _apply_figure_style(fig, "How Centralized, Local-Only, and Federated Compare", "", "Balanced detection score", height=440)
    return fig


def _study_note(metrics: pd.DataFrame, showcase: pd.DataFrame, criteria: dict[str, str | None]) -> str:
    filtered = _apply_filters(metrics, criteria)
    lines = ["### What This Comparison Means"]
    if not filtered.empty:
        lines.append(f"- Matching audit rows: **{len(filtered)}**.")
        best = filtered.sort_values(["f1_mean", "roc_auc_mean", "ece_mean"], ascending=[False, False, True]).iloc[0]
        lines.append(
            f"- Strongest visible row: **{_humanize(best.get('run_type', ''))} / {_humanize(best.get('model', ''))} / {_humanize(best.get('calibration', ''))}** with balanced detection {_fmt(best.get('f1_mean'))}, ranking quality {_fmt(best.get('roc_auc_mean'))}, and positive-case ranking {_fmt(best.get('pr_auc_mean'))}."
        )
    if not showcase.empty:
        lines.append("- The key-findings page shows the strongest presentation-ready slices, while this comparison view preserves the broader study matrix.")
    return "\n".join(lines)


def _filter_exact(frame: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    output = frame.copy()
    if output.empty:
        return output
    for column in ["dataset", "experiment_track", "clients", "partition", "alpha", "run_type", "algorithm", "model", "calibration", "threshold_strategy"]:
        if column in output.columns and column in row.index:
            output = output[output[column].astype(str) == str(row[column])]
    return output


def _humanize(value) -> str:
    key = str(value)
    if key in LABELS:
        return LABELS[key]
    if key.startswith("class_"):
        return "Positive Class" if key.endswith("1") else "Negative Class"
    if key in {"TP", "FP", "TN", "FN"}:
        return key
    return key.replace("_", " ").title()


def _fmt(value) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "n/a"


def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _empty_echart(message: str, title: str = "Chart unavailable") -> dict:
    return {
        "backgroundColor": ACCENT["panel"],
        "title": {
            "text": title,
            "left": 16,
            "top": 12,
            "textStyle": {"color": ACCENT["ink"], "fontSize": 14, "fontWeight": 600},
        },
        "graphic": [
            {
                "type": "text",
                "left": "center",
                "top": "middle",
                "style": {
                    "text": message,
                    "fill": ACCENT["muted"],
                    "fontSize": 14,
                    "fontFamily": "Inter, Arial, Helvetica, sans-serif",
                    "width": 520,
                    "overflow": "break",
                    "align": "center",
                },
            }
        ],
        "xAxis": {"show": False},
        "yAxis": {"show": False},
        "series": [],
    }


def _echart_title(text: str, *, left: str | int = 16, top: int = 12) -> dict:
    return {
        "text": text,
        "left": left,
        "top": top,
        "textStyle": {"color": ACCENT["ink"], "fontSize": 14, "fontWeight": 600},
    }


def _echart_axis(
    name: str | None = None,
    *,
    axis_type: str = "value",
    min_value: float | None = None,
    max_value: float | None = None,
    inverse: bool = False,
    data: list[str] | None = None,
    grid_index: int | None = None,
) -> dict:
    axis: dict[str, object] = {
        "type": axis_type,
        "name": name or "",
        "nameLocation": "middle",
        "nameGap": 34 if name else 14,
        "axisLine": {"lineStyle": {"color": ACCENT["border"]}},
        "axisTick": {"show": False},
        "axisLabel": {"color": ACCENT["muted"], "fontSize": 11},
        "splitLine": {"show": axis_type == "value", "lineStyle": {"color": "#E7EDF5"}},
        "splitNumber": 5,
        "inverse": inverse,
    }
    if min_value is not None:
        axis["min"] = min_value
    if max_value is not None:
        axis["max"] = max_value
    if data is not None:
        axis["data"] = data
    if grid_index is not None:
        axis["gridIndex"] = grid_index
    return axis


def _echart_legend(*, top: int = 16, left: str = "center") -> dict:
    return {
        "top": top,
        "left": left,
        "textStyle": {"color": ACCENT["muted"], "fontSize": 11},
        "itemWidth": 14,
        "itemHeight": 8,
    }


def _echart_tooltip(*, trigger: str = "axis") -> dict:
    return {
        "trigger": trigger,
        "backgroundColor": "#FFFFFF",
        "borderColor": ACCENT["border"],
        "borderWidth": 1,
        "textStyle": {"color": ACCENT["ink"], "fontSize": 12},
        "axisPointer": {"type": "line", "lineStyle": {"color": "#A9B5C5", "type": "dashed"}},
    }


def _echart_toolbox() -> dict:
    return {
        "right": 12,
        "top": 8,
        "feature": {
            "saveAsImage": {"title": "Save"},
            "restore": {"title": "Reset"},
        },
        "iconStyle": {"borderColor": ACCENT["muted"]},
    }


def _line_series(
    name: str,
    x: list[float],
    y: list[float],
    *,
    color: str,
    x_axis: int = 0,
    y_axis: int = 0,
    dashed: bool = False,
    smooth: bool = False,
    show_symbol: bool = False,
    area_opacity: float | None = None,
) -> dict:
    series: dict[str, object] = {
        "type": "line",
        "name": name,
        "xAxisIndex": x_axis,
        "yAxisIndex": y_axis,
        "data": [[float(a), float(b)] for a, b in zip(x, y)],
        "showSymbol": show_symbol,
        "smooth": smooth,
        "symbol": "circle",
        "symbolSize": 5,
        "lineStyle": {"color": color, "width": 2.5, "type": "dashed" if dashed else "solid"},
        "itemStyle": {"color": color},
    }
    if area_opacity is not None:
        series["areaStyle"] = {"color": _rgba(color, area_opacity)}
    return series


def _band_series(
    x: list[float],
    low: list[float],
    high: list[float],
    *,
    color: str,
    stack: str,
    x_axis: int = 0,
    y_axis: int = 0,
) -> list[dict]:
    low_values = np.asarray(low, dtype=float)
    high_values = np.asarray(high, dtype=float)
    diff_values = np.clip(high_values - low_values, 0.0, None)
    return [
        {
            "type": "line",
            "xAxisIndex": x_axis,
            "yAxisIndex": y_axis,
            "data": [[float(a), float(b)] for a, b in zip(x, low_values)],
            "stack": stack,
            "showSymbol": False,
            "lineStyle": {"opacity": 0},
            "areaStyle": {"opacity": 0},
            "tooltip": {"show": False},
            "silent": True,
        },
        {
            "type": "line",
            "xAxisIndex": x_axis,
            "yAxisIndex": y_axis,
            "data": [[float(a), float(b)] for a, b in zip(x, diff_values)],
            "stack": stack,
            "showSymbol": False,
            "lineStyle": {"opacity": 0},
            "areaStyle": {"color": _rgba(color, 0.14)},
            "tooltip": {"show": False},
            "silent": True,
        },
    ]


def _selected_threshold_mark(selected: float, *, x_axis: int = 0, y_axis: int = 0) -> dict:
    return {
        "type": "line",
        "name": "Selected threshold",
        "xAxisIndex": x_axis,
        "yAxisIndex": y_axis,
        "data": [],
        "markLine": {
            "silent": True,
            "symbol": ["none", "none"],
            "lineStyle": {"color": "#6B7280", "type": "dashed", "width": 1.5},
            "label": {"show": False},
            "data": [{"xAxis": float(selected)}],
        },
        "tooltip": {"show": False},
    }


def _humanize_group_value(value) -> str:
    text = str(value)
    mapping = {
        "40_59": "40-59",
        "60_plus": "60+",
        "under_40": "Under 40",
        "healthy": "Healthy BMI",
        "overweight": "Overweight",
        "underweight": "Underweight",
        "obese": "Obese",
        "female": "Female",
        "male": "Male",
        "aggregate": "Overall",
    }
    return mapping.get(text, _humanize(text))


def _apply_figure_style(
    fig: go.Figure,
    title: str,
    xaxis_title: str | None,
    yaxis_title: str | None,
    *,
    height: int = 420,
    legend: bool = True,
) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        paper_bgcolor=ACCENT["panel"],
        plot_bgcolor=ACCENT["panel"],
        font={"family": "Inter, Arial, Helvetica, sans-serif", "color": ACCENT["ink"], "size": 13},
        hovermode="closest",
        height=height,
        margin={"l": 56, "r": 24, "t": 74, "b": 54},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
            "title": {"text": ""},
        },
        showlegend=legend,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E7EDF5", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E7EDF5", zeroline=False)
    if xaxis_title is not None:
        fig.update_xaxes(title=xaxis_title)
    if yaxis_title is not None:
        fig.update_yaxes(title=yaxis_title)
    return fig


def _empty(title: str, subtitle: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        annotations=[
            {
                "text": subtitle,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 14, "color": ACCENT["muted"]},
            }
        ],
    )
    return fig


def _guide_markdown() -> str:
    return """
### How to read this dashboard

This dashboard has two layers.

1. **Key Findings** gives the strongest honest story from the study in one place.  
2. **The other tabs** show the broader evidence behind that story.

### What the study is trying to show

The study compares three ways of training diabetes models:

- **Centralized**: one model trained on data brought together in one place.
- **Local-Only**: each site trains alone and does not collaborate.
- **Federated**: each site trains locally, then only model updates are shared and combined.

The main question is whether federated learning can stay close to the Centralized reference while also giving us useful checks on probability quality, group differences, explanations, and communication cost.

### Filter names in plain English

- **Data Source**: which public diabetes dataset is being used.
- **Run Type**: Centralized, Local-Only, Average Local-Only, or Federated training.
- **Model Family**: the type of model, such as Logistic Regression or XGBoost.
- **Algorithm**: the way local model updates are combined, such as FedAvg or FedProx.
- **Probability Adjustment**: whether raw probabilities were adjusted after training so that the risk percentages better match reality.
- **Number Of Sites**: how many simulated sites or clients took part in federated training.
- **Data Split Type**: whether the sites saw similar data or intentionally different data.
- **Data Difference Level**: how uneven the site data was. Lower values mean bigger differences between sites.
- **Decision Rule**: the cutoff used to turn a risk percentage into a yes/no prediction.

### What the main performance terms mean

- **Ranking Quality (AUROC)** tells us how well the model places people with diabetes above people without diabetes across many possible cutoffs. Higher is better.
- **Balanced Detection Score (F1)** tells us how well the model balances catching positive cases and avoiding incorrect positive calls at one selected decision rule. Higher is better.
- **Positive-Case Ranking (PR AUC)** focuses on how well the model handles the positive class, which matters when the data are imbalanced. Higher is better.
- **Overall accuracy** is the share of all cases predicted correctly, but on imbalanced health data it should not be used alone.

### What the probability-quality terms mean

- **Risk Match Error (ECE)** checks whether the predicted percentages line up with what really happened. Lower is better.
- **Probability Error (Brier)** measures how far the predicted percentages are from the true outcomes. Lower is better.
- **Confidence Penalty (Log Loss)** strongly penalizes confident but wrong predictions. Lower is better.

### What the group-check terms mean

- **Positive Prediction Rate** shows how often each group is flagged positive.
- **Positive Rate Gap** shows how different those positive rates are between groups.
- **Error Gap Between Groups** shows how different the model's error behavior is between groups.

These are subgroup disparity checks, not a final claim of clinical fairness.

### What the explanation terms mean

- **Feature importance** shows which inputs the model relies on most overall.
- **Local explanation** shows which features pushed one example toward higher or lower risk.
- **Explanation stability** shows whether the explanation pattern stays similar as federated training continues.
- **Cross-site feature agreement** shows whether different sites focus on similar important features.

### What the learning-process terms mean

- **Round-by-round training** shows whether the federated model improves steadily as sites keep exchanging updates.
- **Communication cost** shows how much information has to be sent during federated training.

### How to talk about the results

- Use **Centralized** when talking about the strongest raw benchmark.
- Use **Federated** when talking about privacy-friendly training that keeps raw data local.
- Use **decision view** when talking about a chosen cutoff, such as a low false-positive setting or a high-recall setting.
- Use **probability-quality view** when talking about how trustworthy the risk percentages are.

### What the pages are for

- **Key Findings**: the strongest overall story from the study.
- **Model Comparison**: which model families and run types perform best overall.
- **Risk Percentages**: whether the reported risk percentages are trustworthy.
- **Feature Explanations**: what the model is using to make decisions.
- **Group Checks**: whether results differ across age, BMI, or sex groups.
- **Learning Process**: how federated training behaves over time and how much it communicates.
- **Full Study Comparison**: the broader matrix of trained scenarios used for the thesis evidence.
"""


def _index_string() -> str:
    return """<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      body { margin: 0; font-family: Inter, Arial, Helvetica, sans-serif; color: #10243E; background: #F4F7FB; }
      .shell { min-height: 100vh; }
      .header { padding: 30px 34px 22px; border-bottom: 1px solid #D6DEEA; background: #FFFFFF; }
      .eyebrow { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #5F6C7B; margin-bottom: 10px; }
      .header h1 { margin: 0 0 10px; font-size: 36px; line-height: 1.1; }
      .header p { margin: 0; color: #5F6C7B; max-width: 1180px; font-size: 15px; line-height: 1.6; }
      .filters { display: grid; grid-template-columns: repeat(5, minmax(180px, 1fr)); gap: 12px; padding: 18px 24px; }
      .filter label { display: block; margin: 0 0 6px; font-size: 11px; font-weight: 700; color: #5F6C7B; text-transform: uppercase; letter-spacing: 0.08em; }
      .hero-cards, .cards { display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 12px; padding: 18px 24px 0; }
      .hero-card, .card, .leaderboard-card, .chart-card, .summary-card { background: #FFFFFF; border: 1px solid #D6DEEA; border-radius: 8px; box-shadow: 0 8px 24px rgba(16, 36, 62, 0.05); }
      .hero-card { padding: 16px; }
      .hero-title { font-size: 12px; color: #5F6C7B; text-transform: uppercase; font-weight: 700; letter-spacing: 0.06em; }
      .hero-value { font-size: 28px; font-weight: 800; margin-top: 8px; }
      .hero-detail { margin-top: 8px; color: #243447; font-size: 14px; }
      .hero-note { margin-top: 10px; color: #5F6C7B; font-size: 13px; line-height: 1.5; }
      .leaderboard-grid { display: grid; grid-template-columns: repeat(4, minmax(220px, 1fr)); gap: 12px; padding: 14px 24px 0; }
      .leaderboard-card { padding: 14px; }
      .leaderboard-title { font-size: 14px; font-weight: 700; margin-bottom: 10px; }
      .badge-row { display: flex; flex-wrap: wrap; gap: 8px; }
      .badge { display: inline-flex; align-items: center; padding: 6px 10px; border-radius: 8px; background: #F4F7FB; border: 1px solid #D6DEEA; font-size: 12px; color: #243447; }
      .score-audit-grid { display: grid; grid-template-columns: repeat(4, minmax(230px, 1fr)); gap: 12px; padding: 14px 24px 0; }
      .score-card { background: #FFFFFF; border: 1px solid #D6DEEA; border-left: 4px solid #165DFF; border-radius: 8px; padding: 14px; box-shadow: 0 8px 24px rgba(16, 36, 62, 0.05); }
      .score-card.truth { border-left-color: #C75C00; background: #FFF9F2; }
      .score-kicker { font-size: 11px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: #5F6C7B; }
      .score-title { margin-top: 6px; font-size: 15px; font-weight: 800; color: #10243E; }
      .score-value { margin-top: 8px; font-size: 22px; font-weight: 850; color: #10243E; }
      .score-context { margin-top: 8px; font-size: 12px; color: #243447; line-height: 1.45; }
      .score-note { margin-top: 10px; font-size: 13px; color: #5F6C7B; line-height: 1.55; }
      .card { padding: 14px; }
      .card-title { font-size: 12px; color: #5F6C7B; text-transform: uppercase; font-weight: 700; letter-spacing: 0.06em; }
      .card-value { margin-top: 8px; font-size: 24px; font-weight: 800; }
      .card-meta { margin-top: 8px; color: #243447; font-size: 14px; }
      .card-note { margin-top: 8px; color: #5F6C7B; font-size: 13px; line-height: 1.5; }
      .accent { border-color: #BFD1F7; background: #F8FBFF; }
      .chart-grid { display: grid; gap: 12px; padding: 16px 24px 0; }
      .two-up { grid-template-columns: repeat(2, minmax(340px, 1fr)); }
      .chart-card { padding: 12px; }
      .graph-help { margin: 0 8px 6px; padding-top: 6px; border-top: 1px solid #E7EDF5; }
      .graph-help-line { margin: 6px 0; color: #5F6C7B; line-height: 1.55; font-size: 13px; }
      .graph-help-label { color: #10243E; font-weight: 700; }
      .summary-card { margin: 16px 24px; padding: 16px; }
      .summary-markdown { white-space: pre-wrap; line-height: 1.7; color: #243447; }
      .caption { margin: 0 8px 6px; color: #5F6C7B; line-height: 1.5; }
      .guide { padding: 24px 28px 40px; max-width: 1180px; line-height: 1.75; }
      @media (max-width: 1440px) { .leaderboard-grid, .score-audit-grid { grid-template-columns: repeat(2, minmax(220px, 1fr)); } }
      @media (max-width: 1280px) { .filters { grid-template-columns: repeat(3, minmax(180px, 1fr)); } .hero-cards, .cards { grid-template-columns: repeat(2, minmax(220px, 1fr)); } }
      @media (max-width: 900px) { .filters, .hero-cards, .cards, .leaderboard-grid, .score-audit-grid, .two-up { grid-template-columns: 1fr; } }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
  </body>
</html>"""
