"""A small FedAvg-style simulator for binary logistic models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

from .metrics import classification_report_row


AggregationMode = Literal["fedavg", "performance", "calibration_aware"]


class LinearProbabilityModel:
    """Binary linear model wrapper with a scikit-learn-like predict_proba API."""

    def __init__(self, coef: np.ndarray, intercept: float, feature_names: list[str] | None = None):
        self.coef_ = np.asarray(coef, dtype=float).reshape(1, -1)
        self.intercept_ = np.asarray([float(intercept)], dtype=float)
        self.classes_ = np.asarray([0, 1], dtype=int)
        self.feature_names = feature_names or [f"feature_{idx}" for idx in range(self.coef_.shape[1])]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_.reshape(-1) + self.intercept_[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.decision_function(X)
        positive = 1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0)))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


@dataclass
class FederatedRunResult:
    model: object
    round_history: pd.DataFrame
    coefficient_history: list[np.ndarray]
    client_validation_indices: list[np.ndarray]
    client_weights: np.ndarray
    client_history: pd.DataFrame | None = None
    client_models: list[object] | None = None


def train_federated_logistic(
    X: np.ndarray,
    y: np.ndarray,
    client_indices: list[np.ndarray],
    feature_names: list[str],
    rounds: int = 12,
    local_epochs: int = 2,
    aggregation: AggregationMode = "fedavg",
    seed: int = 42,
    X_eval: np.ndarray | None = None,
    y_eval: np.ndarray | None = None,
) -> FederatedRunResult:
    rng = np.random.default_rng(seed)
    n_features = X.shape[1]
    positive_rate = float(np.clip(y.mean(), 1e-3, 1.0 - 1e-3))
    coef = np.zeros(n_features, dtype=float)
    intercept = float(np.log(positive_rate / (1.0 - positive_rate)))
    coefficient_history = [coef.copy()]
    cumulative_bytes = 0
    rows = []

    client_splits = [
        _client_train_validation_split(indices, y=y, seed=seed + client_id)
        for client_id, indices in enumerate(client_indices)
    ]
    client_validation_indices = [validation for _, validation in client_splits]
    client_weights = np.asarray([max(len(train), 1) for train, _ in client_splits], dtype=float)
    client_weights = client_weights / client_weights.sum()

    for round_id in range(1, rounds + 1):
        local_coefs = []
        local_intercepts = []
        local_counts = []
        local_scores = []
        local_metrics = []

        for client_id, (train_idx, val_idx) in enumerate(client_splits):
            if len(train_idx) == 0:
                continue
            model = _make_initialized_sgd(
                X=X,
                y=y,
                train_idx=train_idx,
                coef=coef,
                intercept=intercept,
                seed=seed + 1000 * round_id + client_id,
            )
            for _ in range(local_epochs):
                shuffled = np.asarray(train_idx, dtype=int).copy()
                rng.shuffle(shuffled)
                model.partial_fit(X[shuffled], y[shuffled])

            eval_idx = val_idx if len(val_idx) else train_idx
            probabilities = model.predict_proba(X[eval_idx])[:, 1]
            labels = y[eval_idx]
            metrics = classification_report_row(labels, probabilities)

            local_coefs.append(model.coef_.reshape(-1).copy())
            local_intercepts.append(float(model.intercept_[0]))
            local_counts.append(float(len(train_idx)))
            local_scores.append(_aggregation_score(metrics, aggregation=aggregation))
            local_metrics.append(metrics)

        weights = _aggregation_weights(np.asarray(local_counts), np.asarray(local_scores), aggregation=aggregation)
        coef = np.average(np.vstack(local_coefs), axis=0, weights=weights)
        intercept = float(np.average(np.asarray(local_intercepts), weights=weights))
        coefficient_history.append(coef.copy())

        params_bytes = int((coef.size + 1) * 8)
        round_bytes = params_bytes * len(client_indices) * 2
        cumulative_bytes += round_bytes

        global_metrics = {}
        if X_eval is not None and y_eval is not None:
            global_model = LinearProbabilityModel(coef, intercept, feature_names=feature_names)
            global_metrics = classification_report_row(y_eval, global_model.predict_proba(X_eval)[:, 1])

        rows.append(
            {
                "round": round_id,
                "aggregation": aggregation,
                "local_val_auc_mean": _nanmean([metric["roc_auc"] for metric in local_metrics]),
                "local_val_f1_mean": _nanmean([metric["f1"] for metric in local_metrics]),
                "local_val_ece_mean": _nanmean([metric["ece"] for metric in local_metrics]),
                "global_eval_auc": global_metrics.get("roc_auc", np.nan),
                "global_eval_f1": global_metrics.get("f1", np.nan),
                "global_eval_ece": global_metrics.get("ece", np.nan),
                "params_bytes": params_bytes,
                "round_communication_bytes": round_bytes,
                "cumulative_communication_bytes": cumulative_bytes,
                "coef_l2_norm": float(np.linalg.norm(coef)),
            }
        )

    final_model = LinearProbabilityModel(coef, intercept, feature_names=feature_names)
    return FederatedRunResult(
        model=final_model,
        round_history=pd.DataFrame(rows),
        coefficient_history=coefficient_history,
        client_validation_indices=client_validation_indices,
        client_weights=client_weights,
    )


def _make_initialized_sgd(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    seed: int,
) -> SGDClassifier:
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        learning_rate="optimal",
        random_state=seed,
        fit_intercept=True,
        max_iter=1,
        tol=None,
    )
    first = np.asarray(train_idx[:1], dtype=int)
    model.partial_fit(X[first], y[first], classes=np.asarray([0, 1], dtype=int))
    model.coef_ = np.asarray(coef, dtype=float).reshape(1, -1).copy()
    model.intercept_ = np.asarray([intercept], dtype=float)
    return model


def _client_train_validation_split(indices: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    if len(indices) < 8:
        return indices, indices
    labels = y[indices]
    stratify = labels if len(np.unique(labels)) > 1 and min(np.bincount(labels, minlength=2)) >= 2 else None
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=seed,
        stratify=stratify,
    )
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


def _aggregation_score(metrics: dict[str, float], aggregation: AggregationMode) -> float:
    auc = metrics.get("roc_auc", np.nan)
    f1 = metrics.get("f1", np.nan)
    ece = metrics.get("ece", np.nan)
    performance = auc if not np.isnan(auc) else f1
    if np.isnan(performance):
        performance = 0.5
    if aggregation == "calibration_aware":
        calibration_penalty = ece if not np.isnan(ece) else 0.25
        return float(performance - calibration_penalty)
    return float(performance)


def _aggregation_weights(counts: np.ndarray, scores: np.ndarray, aggregation: AggregationMode) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    counts = np.maximum(counts, 1.0)
    if aggregation == "fedavg":
        weights = counts
    else:
        centered = scores - np.nanmax(scores)
        score_weights = np.exp(np.nan_to_num(centered, nan=0.0))
        weights = counts * score_weights
    return weights / weights.sum()


def _nanmean(values: list[float]) -> float:
    values_array = np.asarray(values, dtype=float)
    if np.all(np.isnan(values_array)):
        return float("nan")
    return float(np.nanmean(values_array))


class NumpyBinaryModel:
    """Small binary model used by the dependency-light federated simulator."""

    def __init__(
        self,
        model_type: str,
        params: np.ndarray,
        n_features: int,
        hidden_units: int = 16,
        feature_names: list[str] | None = None,
    ):
        self.model_type = model_type
        self.params = np.asarray(params, dtype=float).copy()
        self.n_features = int(n_features)
        self.hidden_units = int(hidden_units)
        self.feature_names = feature_names or [f"feature_{idx}" for idx in range(self.n_features)]
        self.classes_ = np.asarray([0, 1], dtype=int)
        if self.model_type == "logistic":
            self.coef_ = self.params[: self.n_features].reshape(1, -1)
            self.intercept_ = np.asarray([self.params[self.n_features]])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        positive = _predict_positive(X, self.params, self.model_type, self.n_features, self.hidden_units)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_federated_numpy(
    X: np.ndarray,
    y: np.ndarray,
    client_indices: list[np.ndarray],
    feature_names: list[str],
    model_type: str = "logistic",
    algorithm: str = "fedavg",
    rounds: int = 12,
    local_epochs: int = 2,
    learning_rate: float = 0.03,
    batch_size: int = 128,
    hidden_units: int = 16,
    fedprox_mu: float = 0.01,
    seed: int = 42,
    X_eval: np.ndarray | None = None,
    y_eval: np.ndarray | None = None,
    target_auc: float | None = None,
    target_f1: float | None = None,
    target_ece: float | None = None,
) -> FederatedRunResult:
    """Train a FedAvg/FedProx binary model with explicit byte accounting."""

    model_type = _normalize_numpy_model_type(model_type)
    algorithm = algorithm.lower()
    if algorithm not in {"fedavg", "fedprox"}:
        raise ValueError(f"algorithm must be fedavg or fedprox; got {algorithm}")

    rng = np.random.default_rng(seed)
    n_features = X.shape[1]
    params = _init_params(model_type, n_features, hidden_units, y, rng)
    coefficient_history = [_feature_scores_from_params(params, model_type, n_features, hidden_units)]
    cumulative_bytes = 0
    rows = []
    client_rows = []
    final_client_models: list[NumpyBinaryModel] = []

    client_splits = [
        _client_train_validation_split(indices, y=y, seed=seed + client_id)
        for client_id, indices in enumerate(client_indices)
    ]
    client_validation_indices = [validation for _, validation in client_splits]
    client_weights = np.asarray([max(len(train), 1) for train, _ in client_splits], dtype=float)
    client_weights = client_weights / client_weights.sum()

    for round_id in range(1, rounds + 1):
        local_params = []
        local_counts = []
        local_metrics = []
        global_params = params.copy()

        for client_id, (train_idx, val_idx) in enumerate(client_splits):
            if len(train_idx) == 0:
                continue
            trained = _local_train_numpy(
                X=X[train_idx],
                y=y[train_idx],
                initial_params=global_params,
                model_type=model_type,
                n_features=n_features,
                hidden_units=hidden_units,
                algorithm=algorithm,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                fedprox_mu=fedprox_mu,
                seed=seed + round_id * 1000 + client_id,
            )
            eval_idx = val_idx if len(val_idx) else train_idx
            model = NumpyBinaryModel(model_type, trained, n_features, hidden_units, feature_names)
            probabilities = model.predict_proba(X[eval_idx])[:, 1]
            metrics = classification_report_row(y[eval_idx], probabilities)
            local_params.append(trained)
            local_counts.append(float(len(train_idx)))
            local_metrics.append(metrics)
            client_rows.append(
                {
                    "round": round_id,
                    "client_id": client_id,
                    "train_rows": int(len(train_idx)),
                    "validation_rows": int(len(eval_idx)),
                    "algorithm": algorithm,
                    "model": model_type,
                    "roc_auc": metrics["roc_auc"],
                    "f1": metrics["f1"],
                    "ece": metrics["ece"],
                    "brier": metrics["brier"],
                }
            )
            if round_id == rounds:
                final_client_models.append(model)

        weights = np.asarray(local_counts, dtype=float)
        weights = weights / weights.sum()
        params = np.average(np.vstack(local_params), axis=0, weights=weights)
        coefficient_history.append(_feature_scores_from_params(params, model_type, n_features, hidden_units))

        params_bytes = int(params.size * 8)
        server_to_client_bytes = params_bytes * len(client_indices)
        client_to_server_bytes = params_bytes * len(local_params)
        round_bytes = server_to_client_bytes + client_to_server_bytes
        cumulative_bytes += round_bytes

        global_metrics = {}
        if X_eval is not None and y_eval is not None:
            global_model = NumpyBinaryModel(model_type, params, n_features, hidden_units, feature_names)
            global_metrics = classification_report_row(y_eval, global_model.predict_proba(X_eval)[:, 1])

        reached_target = _target_reached(global_metrics, target_auc=target_auc, target_f1=target_f1, target_ece=target_ece)
        rows.append(
            {
                "round": round_id,
                "algorithm": algorithm,
                "model": model_type,
                "local_val_auc_mean": _nanmean([metric["roc_auc"] for metric in local_metrics]),
                "local_val_f1_mean": _nanmean([metric["f1"] for metric in local_metrics]),
                "local_val_ece_mean": _nanmean([metric["ece"] for metric in local_metrics]),
                "global_eval_auc": global_metrics.get("roc_auc", np.nan),
                "global_eval_f1": global_metrics.get("f1", np.nan),
                "global_eval_ece": global_metrics.get("ece", np.nan),
                "params_bytes": params_bytes,
                "server_to_client_bytes": server_to_client_bytes,
                "client_to_server_bytes": client_to_server_bytes,
                "round_communication_bytes": round_bytes,
                "cumulative_communication_bytes": cumulative_bytes,
                "target_reached": reached_target,
                "params_l2_norm": float(np.linalg.norm(params)),
            }
        )

    final_model = NumpyBinaryModel(model_type, params, n_features, hidden_units, feature_names)
    return FederatedRunResult(
        model=final_model,
        round_history=pd.DataFrame(rows),
        coefficient_history=coefficient_history,
        client_validation_indices=client_validation_indices,
        client_weights=client_weights,
        client_history=pd.DataFrame(client_rows),
        client_models=final_client_models,
    )


def _normalize_numpy_model_type(model_type: str) -> str:
    key = model_type.strip().lower()
    if key in {"logistic", "lr", "logistic_regression"}:
        return "logistic"
    if key in {"mlp", "shallow_mlp"}:
        return "mlp"
    raise ValueError(f"Unknown federated model type: {model_type}")


def _init_params(
    model_type: str,
    n_features: int,
    hidden_units: int,
    y: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    positive_rate = float(np.clip(y.mean(), 1e-3, 1.0 - 1e-3))
    bias = float(np.log(positive_rate / (1.0 - positive_rate)))
    if model_type == "logistic":
        return np.concatenate([np.zeros(n_features), np.asarray([bias])])
    w1 = rng.normal(0.0, 1.0 / np.sqrt(max(n_features, 1)), size=(n_features, hidden_units))
    b1 = np.zeros(hidden_units)
    w2 = rng.normal(0.0, 1.0 / np.sqrt(max(hidden_units, 1)), size=hidden_units)
    b2 = np.asarray([bias])
    return np.concatenate([w1.reshape(-1), b1, w2, b2])


def _local_train_numpy(
    X: np.ndarray,
    y: np.ndarray,
    initial_params: np.ndarray,
    model_type: str,
    n_features: int,
    hidden_units: int,
    algorithm: str,
    local_epochs: int,
    learning_rate: float,
    batch_size: int,
    fedprox_mu: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    params = initial_params.copy()
    global_params = initial_params.copy()
    batch_size = max(1, int(batch_size))
    for _ in range(local_epochs):
        order = np.arange(len(y))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            gradient = _gradient(X[batch_idx], y[batch_idx], params, model_type, n_features, hidden_units)
            if algorithm == "fedprox":
                gradient = gradient + fedprox_mu * (params - global_params)
            params = params - learning_rate * gradient
    return params


def _gradient(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    model_type: str,
    n_features: int,
    hidden_units: int,
) -> np.ndarray:
    y = y.astype(float)
    n = max(len(y), 1)
    if model_type == "logistic":
        w = params[:n_features]
        b = params[n_features]
        pred = _sigmoid(X @ w + b)
        error = pred - y
        grad_w = X.T @ error / n
        grad_b = np.asarray([error.mean()])
        return np.concatenate([grad_w, grad_b])

    w1, b1, w2, _b2 = _unpack_mlp(params, n_features, hidden_units)
    z1 = X @ w1 + b1
    hidden = np.tanh(z1)
    pred = _sigmoid(hidden @ w2 + _b2)
    error = pred - y
    grad_w2 = hidden.T @ error / n
    grad_b2 = np.asarray([error.mean()])
    hidden_error = np.outer(error, w2) * (1.0 - hidden**2)
    grad_w1 = X.T @ hidden_error / n
    grad_b1 = hidden_error.mean(axis=0)
    return np.concatenate([grad_w1.reshape(-1), grad_b1, grad_w2, grad_b2])


def _predict_positive(
    X: np.ndarray,
    params: np.ndarray,
    model_type: str,
    n_features: int,
    hidden_units: int,
) -> np.ndarray:
    if model_type == "logistic":
        return _sigmoid(X @ params[:n_features] + params[n_features])
    w1, b1, w2, b2 = _unpack_mlp(params, n_features, hidden_units)
    return _sigmoid(np.tanh(X @ w1 + b1) @ w2 + b2)


def _unpack_mlp(params: np.ndarray, n_features: int, hidden_units: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    cursor = 0
    w1_size = n_features * hidden_units
    w1 = params[cursor : cursor + w1_size].reshape(n_features, hidden_units)
    cursor += w1_size
    b1 = params[cursor : cursor + hidden_units]
    cursor += hidden_units
    w2 = params[cursor : cursor + hidden_units]
    cursor += hidden_units
    b2 = float(params[cursor])
    return w1, b1, w2, b2


def _feature_scores_from_params(
    params: np.ndarray,
    model_type: str,
    n_features: int,
    hidden_units: int,
) -> np.ndarray:
    if model_type == "logistic":
        return params[:n_features].copy()
    w1, _b1, w2, _b2 = _unpack_mlp(params, n_features, hidden_units)
    return w1 @ w2


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def _target_reached(
    metrics: dict[str, float],
    target_auc: float | None,
    target_f1: float | None,
    target_ece: float | None,
) -> bool:
    if not metrics:
        return False
    checks = []
    if target_auc is not None:
        checks.append(metrics.get("roc_auc", -np.inf) >= target_auc)
    if target_f1 is not None:
        checks.append(metrics.get("f1", -np.inf) >= target_f1)
    if target_ece is not None:
        checks.append(metrics.get("ece", np.inf) <= target_ece)
    return bool(checks and all(checks))
