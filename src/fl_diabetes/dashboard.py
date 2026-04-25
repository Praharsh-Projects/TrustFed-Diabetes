"""Static dashboard generation for experiment artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


def build_dashboard(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    metrics = _read_csv(output_path / "metrics.csv")
    round_history = _read_csv(output_path / "round_history.csv")
    calibration = _read_csv(output_path / "calibration_bins.csv")
    fairness = _read_csv(output_path / "fairness.csv")
    shap_summary = _read_csv(output_path / "shap_summary.csv")
    stability = _read_csv(output_path / "stability.csv")

    sections = [
        _metrics_section(metrics, include_js=True),
        _training_section(round_history),
        _calibration_section(calibration),
        _fairness_section(fairness),
        _xai_section(shap_summary),
        _stability_section(stability),
    ]

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Federated Diabetes Trustworthiness Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1c2430;
      --muted: #53616f;
      --line: #d6dde4;
      --surface: #ffffff;
      --band: #f4f7f9;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: var(--surface);
      line-height: 1.45;
    }}
    header {{
      padding: 28px 32px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--band);
    }}
    main {{ padding: 24px 32px 40px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 28px 0 12px; font-size: 20px; letter-spacing: 0; }}
    p {{ margin: 0; color: var(--muted); max-width: 980px; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 12px 0 20px;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: var(--band); }}
    .chart {{ margin: 8px 0 26px; }}
  </style>
</head>
<body>
  <header>
    <h1>Federated Diabetes Trustworthiness Dashboard</h1>
    <p>Calibration, fairness, explanations, federated training dynamics, and communication cost from the latest experiment run.</p>
  </header>
  <main>
    {''.join(sections)}
  </main>
</body>
</html>"""

    dashboard_path = output_path / "dashboard.html"
    dashboard_path.write_text(html, encoding="utf-8")
    return dashboard_path


def _read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _metrics_section(metrics: pd.DataFrame, include_js: bool = False) -> str:
    if metrics.empty:
        return "<h2>Metrics</h2><p>No metrics were generated.</p>"
    display_cols = ["run_type", "model", "calibration", "roc_auc", "f1", "brier", "ece", "log_loss"]
    available = [column for column in display_cols if column in metrics.columns]
    table = metrics[available].sort_values(["run_type", "model", "calibration"]).round(4).to_html(index=False)
    fig = px.bar(
        metrics,
        x="model",
        y="roc_auc",
        color="calibration",
        barmode="group",
        facet_col="run_type",
        title="AUROC by Model and Calibration",
    )
    return f"<h2>Metrics</h2>{table}<div class='chart'>{_fig_html(fig, include_js=include_js)}</div>"


def _training_section(round_history: pd.DataFrame) -> str:
    if round_history.empty:
        return "<h2>Federated Training</h2><p>No round history was generated.</p>"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=round_history["round"], y=round_history["global_eval_auc"], mode="lines+markers", name="Global AUROC"))
    fig.add_trace(go.Scatter(x=round_history["round"], y=round_history["global_eval_ece"], mode="lines+markers", name="Global ECE", yaxis="y2"))
    fig.update_layout(
        title="Federated Training Progress",
        xaxis_title="Round",
        yaxis_title="AUROC",
        yaxis2={"title": "ECE", "overlaying": "y", "side": "right"},
        legend={"orientation": "h"},
    )
    comm_fig = px.area(round_history, x="round", y="cumulative_communication_bytes", title="Cumulative Communication Bytes")
    return f"<h2>Federated Training</h2><div class='chart'>{_fig_html(fig)}</div><div class='chart'>{_fig_html(comm_fig)}</div>"


def _calibration_section(calibration: pd.DataFrame) -> str:
    if calibration.empty:
        return "<h2>Calibration</h2><p>No calibration bins were generated.</p>"
    fig = px.line(
        calibration,
        x="avg_predicted_risk",
        y="observed_event_rate",
        color="calibration",
        markers=True,
        title="Reliability Diagram",
    )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line={"dash": "dash", "color": "#53616f"}))
    fig.update_xaxes(range=[0, 1], title="Average predicted risk")
    fig.update_yaxes(range=[0, 1], title="Observed event rate")
    return f"<h2>Calibration</h2><div class='chart'>{_fig_html(fig)}</div>"


def _fairness_section(fairness: pd.DataFrame) -> str:
    if fairness.empty:
        return "<h2>Fairness</h2><p>No fairness metadata was available.</p>"
    fig = px.bar(
        fairness,
        x="group_value",
        y="selection_rate",
        color="group_feature",
        barmode="group",
        title="Selection Rate by Subgroup",
    )
    table = fairness.round(4).to_html(index=False)
    return f"<h2>Fairness</h2><div class='chart'>{_fig_html(fig)}</div>{table}"


def _xai_section(shap_summary: pd.DataFrame) -> str:
    if shap_summary.empty:
        return "<h2>Explanations</h2><p>No explanation summary was generated.</p>"
    top = shap_summary.nsmallest(10, "rank")
    fig = px.bar(top.sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature", orientation="h", title="Top Feature Attributions")
    return f"<h2>Explanations</h2><div class='chart'>{_fig_html(fig)}</div>"


def _stability_section(stability: pd.DataFrame) -> str:
    if stability.empty:
        return "<h2>Explanation Stability</h2><p>No stability history was generated.</p>"
    fig = px.line(
        stability,
        x="round",
        y="spearman_top_feature_stability",
        markers=True,
        title="Round-to-Round Feature Ranking Stability",
    )
    fig.update_yaxes(range=[-1, 1])
    table = stability.round(4).to_html(index=False)
    return f"<h2>Explanation Stability</h2><div class='chart'>{_fig_html(fig)}</div>{table}"


def _fig_html(fig: go.Figure, include_js: bool = False) -> str:
    return pio.to_html(fig, include_plotlyjs=include_js, full_html=False, config={"displaylogo": False})
