"""Evaluation metrics for performance, calibration, fairness, thresholds, and stability."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: Iterable[int], probabilities: Iterable[float], n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = _bin_mask(probabilities, lower, upper)
        if not np.any(mask):
            continue
        observed = y_true[mask].mean()
        predicted = probabilities[mask].mean()
        ece += mask.mean() * abs(observed - predicted)
    return float(ece)


def reliability_bins(y_true: Iterable[int], probabilities: Iterable[float], n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for bin_id, (lower, upper) in enumerate(zip(bins[:-1], bins[1:]), start=1):
        mask = _bin_mask(probabilities, lower, upper)
        rows.append(
            {
                "bin": bin_id,
                "lower": lower,
                "upper": upper,
                "count": int(mask.sum()),
                "avg_predicted_risk": float(probabilities[mask].mean()) if np.any(mask) else np.nan,
                "observed_event_rate": float(y_true[mask].mean()) if np.any(mask) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def classification_report_row(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    threshold: float = 0.5,
    n_bins: int = 10,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    threshold = float(np.clip(threshold, 1e-6, 1.0 - 1e-6))
    y_pred = (probabilities >= threshold).astype(int)
    positives = y_true == 1
    negatives = y_true == 0
    specificity = float((y_pred[negatives] == 0).mean()) if np.any(negatives) else np.nan
    recall = float(recall_score(y_true, y_pred, zero_division=0))

    row: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": float(np.nanmean([recall, specificity])),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": _safe_pr_auc(y_true, probabilities),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "ece": expected_calibration_error(y_true, probabilities, n_bins=n_bins),
        "selection_rate": float(y_pred.mean()),
    }
    row["roc_auc"] = _safe_auc(y_true, probabilities)
    return row


def fairness_report(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    metadata: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    y_pred = (probabilities >= threshold).astype(int)
    rows = []

    for column in metadata.columns:
        values = metadata[column].fillna("unknown").astype(str).to_numpy()
        group_rows = []
        for group in sorted(set(values)):
            mask = values == group
            yt = y_true[mask]
            yp = y_pred[mask]
            positives = yt == 1
            negatives = yt == 0
            tpr = float((yp[positives] == 1).mean()) if np.any(positives) else np.nan
            fpr = float((yp[negatives] == 1).mean()) if np.any(negatives) else np.nan
            group_rows.append(
                {
                    "group_feature": column,
                    "group_value": group,
                    "count": int(mask.sum()),
                    "selection_rate": float(yp.mean()),
                    "true_positive_rate": tpr,
                    "false_positive_rate": fpr,
                }
            )

        if not group_rows:
            continue
        selection_rates = [row["selection_rate"] for row in group_rows]
        tprs = [row["true_positive_rate"] for row in group_rows if not math.isnan(row["true_positive_rate"])]
        fprs = [row["false_positive_rate"] for row in group_rows if not math.isnan(row["false_positive_rate"])]
        dpd = max(selection_rates) - min(selection_rates) if selection_rates else np.nan
        tpr_gap = max(tprs) - min(tprs) if tprs else np.nan
        fpr_gap = max(fprs) - min(fprs) if fprs else np.nan
        eod = np.nanmax([tpr_gap, fpr_gap]) if tprs or fprs else np.nan
        for row in group_rows:
            row["demographic_parity_difference"] = float(dpd)
            row["equalized_odds_difference"] = float(eod)
            rows.append(row)

    return pd.DataFrame(rows)


def select_decision_threshold(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    strategy: str = "fixed_0p5",
    thresholds: Iterable[float] | None = None,
) -> float:
    key = str(strategy).strip().lower()
    if key in {"fixed_0p5", "fixed", "0.5"}:
        return 0.5

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    candidate_thresholds = _candidate_thresholds(probabilities, thresholds)
    best_threshold = 0.5
    best_key: tuple[float, float] | None = None

    for threshold in candidate_thresholds:
        predictions = (probabilities >= threshold).astype(int)
        if key == "calib_f1_optimal":
            score = float(f1_score(y_true, predictions, zero_division=0))
        elif key == "calib_youden_j":
            score = _youden_j(y_true, predictions)
        else:
            raise ValueError(f"Unknown threshold strategy: {strategy}")
        tie_break = -abs(float(threshold) - 0.5)
        current_key = (score, tie_break)
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_threshold = float(threshold)
    return float(best_threshold)


def threshold_sweep_frame(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    metadata: pd.DataFrame,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    rows = []
    for threshold in _candidate_thresholds(probabilities, thresholds):
        base = classification_report_row(y_true, probabilities, threshold=float(threshold))
        fairness = fairness_report(y_true, probabilities, metadata, threshold=float(threshold))
        if fairness.empty:
            rows.append(
                {
                    "threshold": float(threshold),
                    "group_feature": "overall",
                    "group_value": "overall",
                    "demographic_parity_difference": np.nan,
                    "equalized_odds_difference": np.nan,
                    **base,
                }
            )
            continue
        for group_feature, group in fairness.groupby("group_feature", dropna=False):
            rows.append(
                {
                    "threshold": float(threshold),
                    "group_feature": str(group_feature),
                    "group_value": "aggregate",
                    "demographic_parity_difference": float(pd.to_numeric(group["demographic_parity_difference"], errors="coerce").max()),
                    "equalized_odds_difference": float(pd.to_numeric(group["equalized_odds_difference"], errors="coerce").max()),
                    "selection_rate": float(pd.to_numeric(group["selection_rate"], errors="coerce").mean()),
                    **{key: value for key, value in base.items() if key != "selection_rate"},
                }
            )
    return pd.DataFrame(rows)


def coefficient_stability(coefficient_history: list[np.ndarray], feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for round_id in range(1, len(coefficient_history)):
        previous = np.abs(np.asarray(coefficient_history[round_id - 1]).reshape(-1))
        current = np.abs(np.asarray(coefficient_history[round_id]).reshape(-1))
        rows.append(
            {
                "round": round_id + 1,
                "spearman_top_feature_stability": _spearman_from_scores(previous, current),
                "previous_top_feature": feature_names[int(np.argmax(previous))],
                "current_top_feature": feature_names[int(np.argmax(current))],
            }
        )
    return pd.DataFrame(rows)


def cross_client_stability(score_vectors: list[np.ndarray], feature_names: list[str], top_k: int = 5) -> pd.DataFrame:
    rows = []
    for left in range(len(score_vectors)):
        for right in range(left + 1, len(score_vectors)):
            first = np.abs(np.asarray(score_vectors[left]).reshape(-1))
            second = np.abs(np.asarray(score_vectors[right]).reshape(-1))
            rows.append(
                {
                    "client_left": left,
                    "client_right": right,
                    "spearman_rank_correlation": _spearman_from_scores(first, second),
                    "top_k_overlap": _top_k_overlap(first, second, top_k=top_k),
                    "top_k": top_k,
                    "left_top_feature": feature_names[int(np.argmax(first))],
                    "right_top_feature": feature_names[int(np.argmax(second))],
                }
            )
    return pd.DataFrame(rows)


def aggregate_with_confidence_intervals(
    frame: pd.DataFrame,
    group_columns: list[str],
    metric_columns: list[str],
) -> pd.DataFrame:
    rows = []
    if frame.empty:
        return pd.DataFrame()
    for values, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        row = {column: value for column, value in zip(group_columns, values)}
        for metric in metric_columns:
            series = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(series.mean()) if len(series) else np.nan
            row[f"{metric}_std"] = float(series.std(ddof=1)) if len(series) > 1 else 0.0
            row[f"{metric}_ci95_low"], row[f"{metric}_ci95_high"] = _ci95(series)
            row[f"{metric}_n"] = int(len(series))
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probabilities))


def _safe_pr_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, probabilities))


def _spearman_from_scores(first: np.ndarray, second: np.ndarray) -> float:
    if first.size != second.size or first.size < 2:
        return float("nan")
    first_rank = pd.Series(first).rank(method="average").to_numpy()
    second_rank = pd.Series(second).rank(method="average").to_numpy()
    if np.std(first_rank) == 0 or np.std(second_rank) == 0:
        return float("nan")
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def _top_k_overlap(first: np.ndarray, second: np.ndarray, top_k: int) -> float:
    top_k = max(1, min(int(top_k), len(first), len(second)))
    first_top = set(np.argsort(-np.abs(first))[:top_k].tolist())
    second_top = set(np.argsort(-np.abs(second))[:top_k].tolist())
    return float(len(first_top & second_top) / top_k)


def _ci95(series: pd.Series) -> tuple[float, float]:
    if len(series) == 0:
        return float("nan"), float("nan")
    mean = float(series.mean())
    if len(series) == 1:
        return mean, mean
    half_width = 1.96 * float(series.std(ddof=1)) / np.sqrt(len(series))
    return mean - half_width, mean + half_width


def _bin_mask(probabilities: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if upper == 1.0:
        return (probabilities >= lower) & (probabilities <= upper)
    return (probabilities >= lower) & (probabilities < upper)


def _candidate_thresholds(probabilities: np.ndarray, thresholds: Iterable[float] | None = None) -> np.ndarray:
    if thresholds is not None:
        values = np.asarray(list(thresholds), dtype=float)
    else:
        quantiles = np.quantile(probabilities, np.linspace(0.05, 0.95, 19))
        grid = np.linspace(0.05, 0.95, 19)
        values = np.unique(np.clip(np.concatenate([grid, quantiles, np.asarray([0.5])]), 1e-3, 1.0 - 1e-3))
    return np.asarray(sorted(set(np.round(values, 4).tolist())), dtype=float)


def _youden_j(y_true: np.ndarray, predictions: np.ndarray) -> float:
    positives = y_true == 1
    negatives = y_true == 0
    tpr = float((predictions[positives] == 1).mean()) if np.any(positives) else 0.0
    fpr = float((predictions[negatives] == 1).mean()) if np.any(negatives) else 0.0
    return tpr - fpr
