"""Explanation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def shap_summary(
    model: object,
    background: np.ndarray,
    sample: np.ndarray,
    feature_names: list[str],
    seed: int = 42,
    max_samples: int = 120,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    background = _sample_rows(background, max_samples=min(max_samples, len(background)), rng=rng)
    sample = _sample_rows(sample, max_samples=min(max_samples, len(sample)), rng=rng)

    try:
        import shap

        if hasattr(model, "coef_") and hasattr(model, "intercept_"):
            explainer = shap.LinearExplainer(model, background)
            values = explainer(sample).values
        elif hasattr(model, "params") or hasattr(model, "feature_importances_"):
            raise RuntimeError("Use fast deterministic attribution fallback for custom NumPy models.")
        else:
            explainer = shap.Explainer(lambda data: model.predict_proba(data)[:, 1], background)
            values = explainer(sample).values
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        method = "shap"
    except Exception:
        values = _linear_fallback_values(model=model, sample=sample)
        method = "linear_logit_attribution_fallback"

    mean_abs = np.mean(np.abs(values), axis=0)
    order = np.argsort(-mean_abs)
    rows = []
    for rank, feature_idx in enumerate(order, start=1):
        rows.append(
            {
                "rank": rank,
                "feature": feature_names[int(feature_idx)],
                "mean_abs_shap": float(mean_abs[int(feature_idx)]),
                "method": method,
            }
        )
    return pd.DataFrame(rows)


def local_explanation(
    model: object,
    instance: np.ndarray,
    background: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    instance = np.asarray(instance, dtype=float).reshape(1, -1)
    baseline = np.asarray(background, dtype=float).mean(axis=0).reshape(1, -1)
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).reshape(-1)
        contributions = (instance.reshape(-1) - baseline.reshape(-1)) * coef
        method = "linear_shap_approximation"
    else:
        base_probability = float(model.predict_proba(baseline)[:, 1][0])
        contributions = []
        for feature_idx in range(instance.shape[1]):
            perturbed = instance.copy()
            perturbed[0, feature_idx] = baseline[0, feature_idx]
            contributions.append(float(model.predict_proba(instance)[:, 1][0] - model.predict_proba(perturbed)[:, 1][0]))
        contributions = np.asarray(contributions)
        method = "single_feature_probability_delta"

    order = np.argsort(-np.abs(contributions))
    return pd.DataFrame(
        [
            {
                "rank": rank,
                "feature": feature_names[int(feature_idx)],
                "contribution": float(contributions[int(feature_idx)]),
                "method": method,
            }
            for rank, feature_idx in enumerate(order, start=1)
        ]
    )


def model_feature_scores(model: object, n_features: int, hidden_units: int = 16) -> np.ndarray:
    if hasattr(model, "coef_"):
        return np.asarray(model.coef_).reshape(-1)
    if hasattr(model, "params") and getattr(model, "model_type", "") == "mlp":
        params = np.asarray(model.params)
        w1_size = n_features * hidden_units
        w1 = params[:w1_size].reshape(n_features, hidden_units)
        cursor = w1_size + hidden_units
        w2 = params[cursor : cursor + hidden_units]
        return w1 @ w2
    return np.zeros(n_features)


def _sample_rows(values: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if len(values) <= max_samples:
        return values
    indices = rng.choice(len(values), size=max_samples, replace=False)
    return values[np.sort(indices)]


def _linear_fallback_values(model: object, sample: np.ndarray) -> np.ndarray:
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).reshape(-1)
        centered = sample - np.mean(sample, axis=0)
        return centered * coef
    if hasattr(model, "params"):
        scores = model_feature_scores(model, n_features=sample.shape[1], hidden_units=getattr(model, "hidden_units", 16))
        centered = sample - np.mean(sample, axis=0)
        return centered * scores
    if hasattr(model, "feature_importances_"):
        scores = np.asarray(model.feature_importances_).reshape(-1)
        centered = sample - np.mean(sample, axis=0)
        return centered * scores
    return np.zeros_like(sample, dtype=float)
