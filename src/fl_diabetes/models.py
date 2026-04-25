"""Model registries and bounded tuning helpers for the thesis experiments."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from .metrics import classification_report_row, select_decision_threshold

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without xgboost
    XGBClassifier = None


CENTRALIZED_MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "decision_tree",
    "mlp",
    "xgboost",
]

FEDERATED_MODEL_NAMES = ["logistic", "mlp"]


def make_sklearn_model(model_name: str, seed: int, params: dict | None = None) -> BaseEstimator:
    name = normalize_model_name(model_name)
    params = dict(params or {})
    if name == "logistic_regression":
        defaults = {
            "C": 1.0,
            "class_weight": None,
            "max_iter": 1500,
            "solver": "lbfgs",
            "random_state": seed,
        }
        defaults.update(params)
        return LogisticRegression(**defaults)
    if name == "random_forest":
        defaults = {
            "n_estimators": 220,
            "max_depth": None,
            "min_samples_leaf": 3,
            "class_weight": None,
            "random_state": seed,
            "n_jobs": 1,
        }
        defaults.update(params)
        return RandomForestClassifier(**defaults)
    if name == "gradient_boosting":
        defaults = {
            "n_estimators": 140,
            "learning_rate": 0.05,
            "max_depth": 3,
            "random_state": seed,
        }
        defaults.update(params)
        return GradientBoostingClassifier(**defaults)
    if name == "decision_tree":
        defaults = {
            "max_depth": 5,
            "min_samples_leaf": 10,
            "class_weight": None,
            "random_state": seed,
        }
        defaults.update(params)
        return DecisionTreeClassifier(**defaults)
    if name == "mlp":
        defaults = {
            "hidden_layer_sizes": (24,),
            "activation": "relu",
            "alpha": 1e-3,
            "learning_rate_init": 1e-3,
            "max_iter": 800,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "n_iter_no_change": 20,
            "random_state": seed,
        }
        defaults.update(params)
        return MLPClassifier(**defaults)
    if name == "xgboost":
        if XGBClassifier is None:
            raise ModuleNotFoundError("xgboost is required for the xgboost baseline. Install it with `py -m pip install xgboost`.")
        defaults = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_estimators": 220,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "scale_pos_weight": 1.0,
            "random_state": seed,
            "n_jobs": 1,
            "verbosity": 0,
            "tree_method": "hist",
        }
        defaults.update(params)
        return XGBClassifier(**defaults)
    raise ValueError(f"Unknown sklearn model: {model_name}")


def make_model_registry(model_names: Iterable[str] | None, seed: int) -> dict[str, BaseEstimator]:
    names = list(model_names) if model_names else CENTRALIZED_MODEL_NAMES
    return {normalize_model_name(name): make_sklearn_model(name, seed=seed) for name in names}


def fit_best_sklearn_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    dataset_name: str | None = None,
    tune: bool = False,
    search_profile: str = "audit",
    selection_profile: str | None = None,
    threshold_strategy: str = "fixed_0p5",
) -> tuple[BaseEstimator, dict[str, object], pd.DataFrame]:
    name = normalize_model_name(model_name)
    profile = str(selection_profile or search_profile or "audit").strip().lower()
    if not tune:
        model, fitted_params = _fit_model_with_fallback(name, X_train, y_train, seed=seed, params={})
        scores = _score_model(model, X_val, y_val, threshold_strategy=threshold_strategy)
        return model, fitted_params, pd.DataFrame([{**scores, "selected": True, "params": repr(fitted_params)}])

    candidates = list(_search_space(name, y_train, dataset_name, search_profile=search_profile))
    if not candidates:
        candidates = [{}]

    evaluations: list[dict[str, object]] = []
    best_model: BaseEstimator | None = None
    best_params: dict[str, object] = {}
    best_key: tuple[float, float, float] | None = None

    for idx, params in enumerate(candidates):
        model, fitted_params = _fit_model_with_fallback(name, X_train, y_train, seed=seed + idx, params=params)
        scores = _score_model(model, X_val, y_val, threshold_strategy=threshold_strategy)
        key = _selection_key(scores, profile=profile)
        record = {**scores, "params": repr(fitted_params), "selected": False}
        evaluations.append(record)
        if best_key is None or key > best_key:
            best_key = key
            best_model = model
            best_params = dict(fitted_params)

    assert best_model is not None
    if evaluations:
        chosen = max(range(len(evaluations)), key=lambda idx: _selection_key(evaluations[idx], profile=profile))
        evaluations[chosen]["selected"] = True
    return best_model, best_params, pd.DataFrame(evaluations)


def clone_model(model: BaseEstimator) -> BaseEstimator:
    return clone(model)


def normalize_model_name(model_name: str) -> str:
    aliases = {
        "lr": "logistic_regression",
        "logistic": "logistic_regression",
        "rf": "random_forest",
        "gb": "gradient_boosting",
        "gbc": "gradient_boosting",
        "dt": "decision_tree",
        "shallow_mlp": "mlp",
        "xgb": "xgboost",
    }
    key = model_name.strip().lower()
    return aliases.get(key, key)


def normalize_federated_model_name(model_name: str) -> str:
    key = model_name.strip().lower()
    if key in {"lr", "logistic_regression", "logistic"}:
        return "logistic"
    if key in {"mlp", "shallow_mlp"}:
        return "mlp"
    raise ValueError(f"Federated model must be one of {FEDERATED_MODEL_NAMES}; got {model_name}")


def _search_space(
    model_name: str,
    y_train: np.ndarray,
    dataset_name: str | None,
    search_profile: str = "audit",
) -> list[dict[str, object]]:
    positive = float(max(np.sum(y_train == 1), 1))
    negative = float(max(np.sum(y_train == 0), 1))
    imbalance = negative / positive
    profile = str(search_profile or "audit").strip().lower()
    if model_name == "logistic_regression":
        class_weights = [None, "balanced"] if (dataset_name or "").lower() == "cdc" else [None]
        return list(ParameterGrid({"C": [0.1, 1.0, 3.0], "class_weight": class_weights}))
    if model_name == "random_forest":
        class_weights = [None, "balanced"] if (dataset_name or "").lower() == "cdc" else [None]
        if profile == "showcase":
            return list(
                ParameterGrid(
                    {
                        "n_estimators": [300, 500],
                        "max_depth": [None, 12],
                        "min_samples_leaf": [1, 5],
                        "class_weight": class_weights,
                    }
                )
            )
        return list(
            ParameterGrid(
                {
                    "n_estimators": [160, 220],
                    "max_depth": [None, 8],
                    "min_samples_leaf": [3],
                    "class_weight": class_weights,
                }
            )
        )
    if model_name == "gradient_boosting":
        if profile == "showcase":
            return list(
                ParameterGrid(
                    {
                        "n_estimators": [150, 250],
                        "learning_rate": [0.03, 0.05],
                        "max_depth": [2, 3],
                        "min_samples_leaf": [20, 50],
                    }
                )
            )
        return list(
            ParameterGrid(
                {
                    "n_estimators": [100, 160],
                    "learning_rate": [0.05],
                    "max_depth": [2, 3],
                }
            )
        )
    if model_name == "decision_tree":
        class_weights = [None, "balanced"] if (dataset_name or "").lower() == "cdc" else [None]
        return list(
            ParameterGrid(
                {
                    "max_depth": [4, 6, 8],
                    "min_samples_leaf": [5, 10],
                    "class_weight": class_weights,
                }
            )
        )
    if model_name == "mlp":
        if profile == "showcase":
            return list(
                ParameterGrid(
                    {
                        "hidden_layer_sizes": [(64, 32), (128, 64)],
                        "alpha": [1e-4, 1e-3],
                        "learning_rate_init": [5e-4, 1e-3],
                        "max_iter": [500],
                        "early_stopping": [True],
                    }
                )
            )
        return list(
            ParameterGrid(
                {
                    "hidden_layer_sizes": [(24,), (32,)],
                    "alpha": [1e-3],
                    "learning_rate_init": [1e-3],
                    "max_iter": [350],
                    "early_stopping": [True],
                }
            )
        )
    if model_name == "xgboost":
        if profile == "showcase":
            scale_pos = [1.0]
            if (dataset_name or "").lower() == "cdc":
                scale_pos = sorted(set([round(imbalance, 4), round(1.15 * imbalance, 4)]))
            return list(
                ParameterGrid(
                    {
                        "n_estimators": [300, 500],
                        "max_depth": [3, 5],
                        "learning_rate": [0.03, 0.05],
                        "subsample": [0.9],
                        "colsample_bytree": [0.9],
                        "scale_pos_weight": scale_pos,
                    }
                )
            )
        return list(
            ParameterGrid(
                {
                    "n_estimators": [140, 220],
                    "max_depth": [3, 5],
                    "learning_rate": [0.05],
                    "subsample": [0.9],
                    "colsample_bytree": [0.9],
                    "scale_pos_weight": [1.0, imbalance] if (dataset_name or "").lower() == "cdc" else [1.0],
                }
            )
        )
    return [{}]


def _score_model(
    model: BaseEstimator,
    X_val: np.ndarray,
    y_val: np.ndarray,
    threshold_strategy: str = "fixed_0p5",
) -> dict[str, float]:
    probabilities = np.asarray(model.predict_proba(X_val)[:, 1], dtype=float)
    threshold = select_decision_threshold(y_val, probabilities, strategy=threshold_strategy)
    scores = classification_report_row(y_val, probabilities, threshold=threshold)
    scores["decision_threshold"] = float(threshold)
    return scores


def _fit_model_with_fallback(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    params: dict[str, object],
) -> tuple[BaseEstimator, dict[str, object]]:
    effective_params = dict(params)
    model = make_sklearn_model(model_name, seed=seed, params=effective_params)
    try:
        model.fit(X_train, y_train)
        return model, effective_params
    except ValueError as exc:
        if not _should_retry_mlp_without_early_stopping(model_name, y_train, effective_params, exc):
            raise
        fallback_params = dict(effective_params)
        fallback_params["early_stopping"] = False
        model = make_sklearn_model(model_name, seed=seed, params=fallback_params)
        model.fit(X_train, y_train)
        return model, fallback_params


def _should_retry_mlp_without_early_stopping(
    model_name: str,
    y_train: np.ndarray,
    params: dict[str, object],
    exc: ValueError,
) -> bool:
    if model_name != "mlp":
        return False
    if not bool(params.get("early_stopping", True)):
        return False
    counts = np.bincount(np.asarray(y_train, dtype=int), minlength=2)
    if int(counts.min()) < 2:
        return True
    message = str(exc).lower()
    retry_markers = [
        "least populated class",
        "validation set",
        "test_size",
        "number of classes",
    ]
    return any(marker in message for marker in retry_markers)


def _selection_key(scores: dict[str, object], profile: str = "audit") -> tuple[float, ...]:
    profile = str(profile or "audit").strip().lower()
    auc = float(scores.get("roc_auc", np.nan))
    auc_key = auc if not np.isnan(auc) else -1.0
    f1 = float(scores.get("f1", np.nan))
    f1_key = f1 if not np.isnan(f1) else -1.0
    ece = float(scores.get("ece", np.inf))
    brier = float(scores.get("brier", np.inf))
    if profile == "showcase":
        return f1_key, auc_key, -brier, -ece
    return auc_key, -brier, -abs(auc_key - 0.5)


def _safe_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probabilities))
