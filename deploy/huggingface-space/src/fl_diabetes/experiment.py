"""Experiment orchestration for the thesis implementation artifact."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .calibration import FederatedCalibrator, ProbabilityCalibrator
from .dashboard import build_dashboard
from .data import DatasetBundle, load_dataset, partition_clients, save_dataset_manifest
from .federated import FederatedRunResult, train_federated_numpy
from .metrics import (
    classification_report_row,
    coefficient_stability,
    cross_client_stability,
    fairness_report,
    reliability_bins,
    select_decision_threshold,
    threshold_sweep_frame,
)
from .models import fit_best_sklearn_model, normalize_federated_model_name
from .xai import local_explanation, model_feature_scores, shap_summary


DEFAULT_EXPERIMENT = {
    "dataset": "synthetic",
    "data_path": None,
    "max_rows": None,
    "synthetic_samples": 1200,
    "seed": 42,
    "clients": 5,
    "partition": "iid",
    "alpha": 0.5,
    "centralized_models": ["logistic_regression", "random_forest", "gradient_boosting", "decision_tree", "mlp", "xgboost"],
    "local_only_models": None,
    "federated_models": ["logistic"],
    "algorithms": ["fedavg"],
    "calibrations": ["none", "sigmoid", "isotonic"],
    "rounds": 12,
    "local_epochs": 2,
    "learning_rate": 0.03,
    "batch_size": 128,
    "hidden_units": 16,
    "fedprox_mu": 0.01,
    "target_auc": None,
    "target_f1": None,
    "target_ece": None,
    "output_dir": "results/runs/latest",
    "build_static_dashboard": True,
    "tune_models": False,
    "tune_local_models": None,
    "tune_federated": False,
    "decision_threshold_strategy": "fixed_0p5",
    "search_profile": "audit",
    "save_prediction_artifacts": False,
    "experiment_track": "audit",
}


@dataclass
class ExperimentArtifacts:
    output_dir: Path
    metrics: pd.DataFrame
    summary: dict[str, Any]


def run_single_experiment(config: dict[str, Any]) -> ExperimentArtifacts:
    cfg = {**DEFAULT_EXPERIMENT, **config}
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"running {output_dir.name}: dataset={cfg['dataset']} seed={cfg['seed']} "
        f"clients={cfg['clients']} partition={cfg['partition']} alpha={cfg['alpha']} "
        f"threshold={cfg.get('decision_threshold_strategy', 'fixed_0p5')}",
        flush=True,
    )

    bundle = load_dataset(
        dataset=cfg["dataset"],
        data_path=cfg.get("data_path"),
        seed=int(cfg["seed"]),
        n_samples=int(cfg.get("synthetic_samples") or 1200),
        max_rows=cfg.get("max_rows"),
    )
    save_dataset_manifest(bundle, output_dir)
    (output_dir / "config_snapshot.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    client_indices = partition_clients(
        y=bundle.y_train,
        n_clients=int(cfg["clients"]),
        strategy=cfg["partition"],
        alpha=float(cfg["alpha"]),
        seed=int(cfg["seed"]),
    )
    _save_client_manifest(bundle, client_indices, output_dir)

    metrics_rows: list[dict[str, Any]] = []
    calibration_rows: list[pd.DataFrame] = []
    fairness_rows: list[pd.DataFrame] = []
    communication_frames: list[pd.DataFrame] = []
    client_metric_frames: list[pd.DataFrame] = []
    xai_frames: list[pd.DataFrame] = []
    local_xai_frames: list[pd.DataFrame] = []
    stability_frames: list[pd.DataFrame] = []
    threshold_sweep_frames: list[pd.DataFrame] = []
    test_prediction_frames: list[pd.DataFrame] = []
    calibration_prediction_frames: list[pd.DataFrame] = []

    _run_centralized_and_local(
        cfg=cfg,
        bundle=bundle,
        client_indices=client_indices,
        metrics_rows=metrics_rows,
        calibration_rows=calibration_rows,
        fairness_rows=fairness_rows,
        xai_frames=xai_frames,
        local_xai_frames=local_xai_frames,
        threshold_sweep_frames=threshold_sweep_frames,
        test_prediction_frames=test_prediction_frames,
        calibration_prediction_frames=calibration_prediction_frames,
    )
    _run_federated(
        cfg=cfg,
        bundle=bundle,
        client_indices=client_indices,
        metrics_rows=metrics_rows,
        calibration_rows=calibration_rows,
        fairness_rows=fairness_rows,
        communication_frames=communication_frames,
        client_metric_frames=client_metric_frames,
        xai_frames=xai_frames,
        local_xai_frames=local_xai_frames,
        stability_frames=stability_frames,
        threshold_sweep_frames=threshold_sweep_frames,
        test_prediction_frames=test_prediction_frames,
        calibration_prediction_frames=calibration_prediction_frames,
    )

    metrics = pd.DataFrame(metrics_rows)
    _write_frame(metrics, output_dir / "metrics.csv")
    _write_frame(_concat_or_empty(calibration_rows), output_dir / "calibration_bins.csv")
    _write_frame(_concat_or_empty(fairness_rows), output_dir / "fairness.csv")
    _write_frame(_concat_or_empty(communication_frames), output_dir / "communication.csv")
    _write_frame(_concat_or_empty(communication_frames), output_dir / "round_history.csv")
    _write_frame(_concat_or_empty(client_metric_frames), output_dir / "client_metrics.csv")
    _write_frame(_concat_or_empty(xai_frames), output_dir / "shap_summary.csv")
    _write_frame(_concat_or_empty(local_xai_frames), output_dir / "local_explanations.csv")
    _write_frame(_concat_or_empty(stability_frames), output_dir / "stability.csv")
    _write_frame(_concat_or_empty(threshold_sweep_frames), output_dir / "threshold_sweeps.csv")
    if cfg.get("save_prediction_artifacts", False):
        _write_frame(_concat_or_empty(test_prediction_frames), output_dir / "test_predictions.csv")
        _write_frame(_concat_or_empty(calibration_prediction_frames), output_dir / "calibration_predictions.csv")

    dashboard = None
    if cfg.get("build_static_dashboard", True):
        dashboard = str(build_dashboard(output_dir))

    summary = {
        "run_id": output_dir.name,
        "dataset": bundle.name,
        "seed": int(cfg["seed"]),
        "clients": int(cfg["clients"]),
        "partition": cfg["partition"],
        "alpha": float(cfg["alpha"]),
        "algorithms": cfg["algorithms"],
        "federated_models": cfg["federated_models"],
        "calibrations": cfg["calibrations"],
        "decision_threshold_strategy": cfg.get("decision_threshold_strategy", "fixed_0p5"),
        "search_profile": cfg.get("search_profile", "audit"),
        "experiment_track": cfg.get("experiment_track", "audit"),
        "rows": {
            "train": len(bundle.y_train),
            "calibration": len(bundle.y_calib),
            "test": len(bundle.y_test),
        },
        "best_metric_row": _best_metric_row(metrics),
        "dashboard": dashboard,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ExperimentArtifacts(output_dir=output_dir, metrics=metrics, summary=summary)


def _run_centralized_and_local(
    cfg: dict[str, Any],
    bundle: DatasetBundle,
    client_indices: list[np.ndarray],
    metrics_rows: list[dict[str, Any]],
    calibration_rows: list[pd.DataFrame],
    fairness_rows: list[pd.DataFrame],
    xai_frames: list[pd.DataFrame],
    local_xai_frames: list[pd.DataFrame],
    threshold_sweep_frames: list[pd.DataFrame],
    test_prediction_frames: list[pd.DataFrame],
    calibration_prediction_frames: list[pd.DataFrame],
) -> None:
    for model_name in cfg.get("centralized_models", DEFAULT_EXPERIMENT["centralized_models"]):
        print(f"  centralized/local baseline: {model_name}", flush=True)
        model, _, _ = fit_best_sklearn_model(
            model_name=model_name,
            X_train=bundle.X_train,
            y_train=bundle.y_train,
            X_val=bundle.X_calib,
            y_val=bundle.y_calib,
            seed=int(cfg["seed"]),
            dataset_name=bundle.name,
            tune=bool(cfg.get("tune_models", False)),
            search_profile=str(cfg.get("search_profile", "audit")),
            selection_profile=str(cfg.get("search_profile", "audit")),
            threshold_strategy=str(cfg.get("decision_threshold_strategy", "fixed_0p5")),
        )
        raw_calib = model.predict_proba(bundle.X_calib)[:, 1]
        raw_test = model.predict_proba(bundle.X_test)[:, 1]
        for calibration in cfg["calibrations"]:
            probabilities = _apply_calibration(calibration, raw_calib, bundle.y_calib, raw_test)
            calib_probabilities = _apply_calibration(calibration, raw_calib, bundle.y_calib, raw_calib)
            context = _context(cfg, "centralized", model_name, calibration)
            _record_outputs(
                context=context,
                test_labels=bundle.y_test,
                probabilities=probabilities,
                raw_probabilities=raw_test,
                test_meta=bundle.test_meta,
                test_row_ids=bundle.test_indices,
                calib_probabilities=calib_probabilities,
                raw_calib_probabilities=raw_calib,
                calib_labels=bundle.y_calib,
                calib_meta=bundle.calib_meta,
                calib_row_ids=bundle.calib_indices,
                metrics_rows=metrics_rows,
                calibration_rows=calibration_rows,
                fairness_rows=fairness_rows,
                threshold_sweep_frames=threshold_sweep_frames,
                test_prediction_frames=test_prediction_frames,
                calibration_prediction_frames=calibration_prediction_frames,
                save_prediction_artifacts=bool(cfg.get("save_prediction_artifacts", False)),
            )
        if model_name in {"logistic_regression", "random_forest", "gradient_boosting", "xgboost"}:
            xai = shap_summary(model, bundle.X_train, bundle.X_test, bundle.feature_names, seed=int(cfg["seed"]))
            xai_frames.append(_with_context(xai, _context(cfg, "centralized", model_name, "none")))
            local = local_explanation(model, bundle.X_test[0], bundle.X_train, bundle.feature_names)
            local_xai_frames.append(_with_context(local, _context(cfg, "centralized", model_name, "none")))

    local_only_models = cfg.get("local_only_models")
    if local_only_models is None:
        local_only_models = cfg.get("centralized_models", DEFAULT_EXPERIMENT["centralized_models"])
    tune_local = cfg.get("tune_local_models")
    if tune_local is None:
        tune_local = bool(cfg.get("tune_models", False))
    for model_name in local_only_models:
        print(f"  local-only baseline: {model_name}", flush=True)
        local_rows = []
        for client_id, indices in enumerate(client_indices):
            if len(np.unique(bundle.y_train[indices])) < 2:
                continue
            train_idx, val_idx = _local_split(indices, bundle.y_train, seed=int(cfg["seed"]) + client_id)
            if len(np.unique(bundle.y_train[train_idx])) < 2:
                continue
            model, _, _ = fit_best_sklearn_model(
                model_name=model_name,
                X_train=bundle.X_train[train_idx],
                y_train=bundle.y_train[train_idx],
                X_val=bundle.X_train[val_idx],
                y_val=bundle.y_train[val_idx],
                seed=int(cfg["seed"]) + client_id,
                dataset_name=bundle.name,
                tune=bool(tune_local),
                search_profile=str(cfg.get("search_profile", "audit")),
                selection_profile=str(cfg.get("search_profile", "audit")),
                threshold_strategy=str(cfg.get("decision_threshold_strategy", "fixed_0p5")),
            )
            raw_val = model.predict_proba(bundle.X_train[val_idx])[:, 1]
            raw_test = model.predict_proba(bundle.X_test)[:, 1]
            for calibration in cfg["calibrations"]:
                probabilities = _apply_calibration(calibration, raw_val, bundle.y_train[val_idx], raw_test)
                calib_probabilities = _apply_calibration(calibration, raw_val, bundle.y_train[val_idx], raw_val)
                context = _context(cfg, "local_only", model_name, calibration, client_id=client_id)
                row = _record_outputs(
                    context=context,
                    test_labels=bundle.y_test,
                    probabilities=probabilities,
                    raw_probabilities=raw_test,
                    test_meta=bundle.test_meta,
                    test_row_ids=bundle.test_indices,
                    calib_probabilities=calib_probabilities,
                    raw_calib_probabilities=raw_val,
                    calib_labels=bundle.y_train[val_idx],
                    calib_meta=bundle.train_meta.iloc[val_idx].reset_index(drop=True),
                    calib_row_ids=np.asarray(bundle.train_indices, dtype=int)[val_idx].tolist(),
                    metrics_rows=metrics_rows,
                    calibration_rows=calibration_rows,
                    fairness_rows=fairness_rows,
                    threshold_sweep_frames=threshold_sweep_frames,
                    test_prediction_frames=test_prediction_frames,
                    calibration_prediction_frames=calibration_prediction_frames,
                    save_prediction_artifacts=bool(cfg.get("save_prediction_artifacts", False)),
                )
                local_rows.append(row)
        if local_rows:
            local_frame = pd.DataFrame(local_rows)
            for calibration, group in local_frame.groupby("calibration", dropna=False):
                mean_row = group.select_dtypes(include=[np.number]).mean(numeric_only=True).to_dict()
                context = _context(cfg, "local_only_mean", model_name, str(calibration))
                metrics_rows.append({**context, **{key: float(value) for key, value in mean_row.items() if key in _metric_names() or key == "decision_threshold"}})


def _run_federated(
    cfg: dict[str, Any],
    bundle: DatasetBundle,
    client_indices: list[np.ndarray],
    metrics_rows: list[dict[str, Any]],
    calibration_rows: list[pd.DataFrame],
    fairness_rows: list[pd.DataFrame],
    communication_frames: list[pd.DataFrame],
    client_metric_frames: list[pd.DataFrame],
    xai_frames: list[pd.DataFrame],
    local_xai_frames: list[pd.DataFrame],
    stability_frames: list[pd.DataFrame],
    threshold_sweep_frames: list[pd.DataFrame],
    test_prediction_frames: list[pd.DataFrame],
    calibration_prediction_frames: list[pd.DataFrame],
) -> None:
    for model_name in cfg.get("federated_models", ["logistic"]):
        federated_model = normalize_federated_model_name(model_name)
        for algorithm in cfg.get("algorithms", ["fedavg"]):
            print(f"  federated model: {algorithm}/{federated_model}", flush=True)
            result = _fit_federated_model(cfg, bundle, client_indices, federated_model, algorithm)
            base_context = _context(cfg, "federated", federated_model, "none", algorithm=algorithm)
            communication_frames.append(_with_context(result.round_history, base_context))
            if result.client_history is not None:
                client_metric_frames.append(_with_context(result.client_history, base_context))

            raw_calib = result.model.predict_proba(bundle.X_calib)[:, 1]
            raw_test = result.model.predict_proba(bundle.X_test)[:, 1]
            client_val_probs = [result.model.predict_proba(bundle.X_train[val_idx])[:, 1] for val_idx in result.client_validation_indices]
            client_val_labels = [bundle.y_train[val_idx] for val_idx in result.client_validation_indices]

            _record_outputs(
                context=base_context,
                test_labels=bundle.y_test,
                probabilities=raw_test,
                raw_probabilities=raw_test,
                test_meta=bundle.test_meta,
                test_row_ids=bundle.test_indices,
                calib_probabilities=raw_calib,
                raw_calib_probabilities=raw_calib,
                calib_labels=bundle.y_calib,
                calib_meta=bundle.calib_meta,
                calib_row_ids=bundle.calib_indices,
                metrics_rows=metrics_rows,
                calibration_rows=calibration_rows,
                fairness_rows=fairness_rows,
                threshold_sweep_frames=threshold_sweep_frames,
                test_prediction_frames=test_prediction_frames,
                calibration_prediction_frames=calibration_prediction_frames,
                save_prediction_artifacts=bool(cfg.get("save_prediction_artifacts", False)),
            )
            for calibration in cfg["calibrations"]:
                if calibration == "none":
                    continue
                global_probabilities = _apply_calibration(calibration, raw_calib, bundle.y_calib, raw_test)
                global_calib_probabilities = _apply_calibration(calibration, raw_calib, bundle.y_calib, raw_calib)
                global_context = _context(cfg, "federated", federated_model, f"global_{calibration}", algorithm=algorithm)
                _record_outputs(
                    context=global_context,
                    test_labels=bundle.y_test,
                    probabilities=global_probabilities,
                    raw_probabilities=raw_test,
                    test_meta=bundle.test_meta,
                    test_row_ids=bundle.test_indices,
                    calib_probabilities=global_calib_probabilities,
                    raw_calib_probabilities=raw_calib,
                    calib_labels=bundle.y_calib,
                    calib_meta=bundle.calib_meta,
                    calib_row_ids=bundle.calib_indices,
                    metrics_rows=metrics_rows,
                    calibration_rows=calibration_rows,
                    fairness_rows=fairness_rows,
                    threshold_sweep_frames=threshold_sweep_frames,
                    test_prediction_frames=test_prediction_frames,
                    calibration_prediction_frames=calibration_prediction_frames,
                    save_prediction_artifacts=bool(cfg.get("save_prediction_artifacts", False)),
                )

                federated_calibrator = FederatedCalibrator(calibration).fit(
                    client_val_probs,
                    client_val_labels,
                    weights=result.client_weights,
                )
                federated_probabilities = federated_calibrator.transform(raw_test)
                federated_calib_probabilities = federated_calibrator.transform(raw_calib)
                fed_context = _context(cfg, "federated", federated_model, f"federated_{calibration}", algorithm=algorithm)
                _record_outputs(
                    context=fed_context,
                    test_labels=bundle.y_test,
                    probabilities=federated_probabilities,
                    raw_probabilities=raw_test,
                    test_meta=bundle.test_meta,
                    test_row_ids=bundle.test_indices,
                    calib_probabilities=federated_calib_probabilities,
                    raw_calib_probabilities=raw_calib,
                    calib_labels=bundle.y_calib,
                    calib_meta=bundle.calib_meta,
                    calib_row_ids=bundle.calib_indices,
                    metrics_rows=metrics_rows,
                    calibration_rows=calibration_rows,
                    fairness_rows=fairness_rows,
                    threshold_sweep_frames=threshold_sweep_frames,
                    test_prediction_frames=test_prediction_frames,
                    calibration_prediction_frames=calibration_prediction_frames,
                    save_prediction_artifacts=bool(cfg.get("save_prediction_artifacts", False)),
                )

            xai = shap_summary(result.model, bundle.X_train, bundle.X_test, bundle.feature_names, seed=int(cfg["seed"]))
            xai_frames.append(_with_context(xai, base_context))
            local = local_explanation(result.model, bundle.X_test[0], bundle.X_train, bundle.feature_names)
            local_xai_frames.append(_with_context(local, base_context))

            stability = coefficient_stability(result.coefficient_history, bundle.feature_names)
            stability_frames.append(_with_context(stability, base_context))
            if result.client_models:
                scores = [
                    model_feature_scores(client_model, len(bundle.feature_names), hidden_units=int(cfg["hidden_units"]))
                    for client_model in result.client_models
                ]
                stability_frames.append(_with_context(cross_client_stability(scores, bundle.feature_names), base_context))
                for client_id, client_model in enumerate(result.client_models):
                    client_xai = shap_summary(
                        client_model,
                        bundle.X_train,
                        bundle.X_test,
                        bundle.feature_names,
                        seed=int(cfg["seed"]) + client_id,
                    )
                    xai_frames.append(_with_context(client_xai, {**base_context, "client_id": client_id}))


def _fit_federated_model(
    cfg: dict[str, Any],
    bundle: DatasetBundle,
    client_indices: list[np.ndarray],
    model_name: str,
    algorithm: str,
) -> FederatedRunResult:
    candidates = list(_federated_candidates(cfg, model_name)) if cfg.get("tune_federated") else [
        {
            "rounds": int(cfg["rounds"]),
            "local_epochs": int(cfg["local_epochs"]),
            "learning_rate": float(cfg["learning_rate"]),
            "batch_size": int(cfg["batch_size"]),
            "hidden_units": int(cfg["hidden_units"]),
        }
    ]
    best_result: FederatedRunResult | None = None
    best_key: tuple[float, ...] | None = None
    selection_profile = str(cfg.get("search_profile", "audit")).strip().lower()
    for candidate in candidates:
        result = train_federated_numpy(
            X=bundle.X_train,
            y=bundle.y_train,
            client_indices=client_indices,
            feature_names=bundle.feature_names,
            model_type=model_name,
            algorithm=algorithm,
            rounds=int(candidate["rounds"]),
            local_epochs=int(candidate["local_epochs"]),
            learning_rate=float(candidate["learning_rate"]),
            batch_size=int(candidate["batch_size"]),
            hidden_units=int(candidate["hidden_units"]),
            fedprox_mu=float(cfg["fedprox_mu"]),
            seed=int(cfg["seed"]),
            X_eval=bundle.X_calib,
            y_eval=bundle.y_calib,
            target_auc=cfg.get("target_auc"),
            target_f1=cfg.get("target_f1"),
            target_ece=cfg.get("target_ece"),
        )
        raw_calib = result.model.predict_proba(bundle.X_calib)[:, 1]
        threshold = select_decision_threshold(
            bundle.y_calib,
            raw_calib,
            strategy=str(cfg.get("decision_threshold_strategy", "fixed_0p5")),
        )
        scores = classification_report_row(bundle.y_calib, raw_calib, threshold=threshold)
        key = _selection_key(scores, profile=selection_profile)
        if best_key is None or key > best_key:
            best_key = key
            best_result = result
    assert best_result is not None
    return best_result


def _federated_candidates(cfg: dict[str, Any], model_name: str) -> list[dict[str, float | int]]:
    if model_name == "logistic":
        grid = {
            "rounds": [20, 30, 40],
            "local_epochs": [1, 2],
            "learning_rate": [0.01, 0.03],
            "batch_size": [int(cfg["batch_size"])],
            "hidden_units": [int(cfg["hidden_units"])],
        }
    else:
        grid = {
            "rounds": [20, 30],
            "local_epochs": [1, 2],
            "learning_rate": [0.01, 0.03],
            "batch_size": [128, 256],
            "hidden_units": [16, 32],
        }
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]


def _apply_calibration(calibration: str, calib_prob: np.ndarray, y_calib: np.ndarray, target_prob: np.ndarray) -> np.ndarray:
    if calibration == "none":
        return np.clip(target_prob, 1e-6, 1.0 - 1e-6)
    return ProbabilityCalibrator(calibration).fit(calib_prob, y_calib).transform(target_prob)


def _record_outputs(
    context: dict[str, Any],
    test_labels: np.ndarray,
    probabilities: np.ndarray,
    raw_probabilities: np.ndarray,
    test_meta: pd.DataFrame,
    test_row_ids: list[int],
    calib_probabilities: np.ndarray,
    raw_calib_probabilities: np.ndarray,
    calib_labels: np.ndarray,
    calib_meta: pd.DataFrame,
    calib_row_ids: list[int],
    metrics_rows: list[dict[str, Any]],
    calibration_rows: list[pd.DataFrame],
    fairness_rows: list[pd.DataFrame],
    threshold_sweep_frames: list[pd.DataFrame],
    test_prediction_frames: list[pd.DataFrame],
    calibration_prediction_frames: list[pd.DataFrame],
    save_prediction_artifacts: bool,
) -> dict[str, Any]:
    threshold_strategy = context.get("threshold_strategy", "fixed_0p5")
    decision_threshold = select_decision_threshold(calib_labels, calib_probabilities, strategy=threshold_strategy)
    metric_row = {
        **context,
        "decision_threshold": float(decision_threshold),
        **classification_report_row(test_labels, probabilities, threshold=decision_threshold),
    }
    metrics_rows.append(metric_row)
    bins = reliability_bins(test_labels, probabilities)
    calibration_rows.append(_with_context(bins, {**context, "decision_threshold": float(decision_threshold)}))
    fairness = fairness_report(test_labels, probabilities, test_meta, threshold=decision_threshold)
    if not fairness.empty:
        fairness_rows.append(_with_context(fairness, {**context, "decision_threshold": float(decision_threshold)}))
    sweep = threshold_sweep_frame(test_labels, probabilities, test_meta)
    if not sweep.empty:
        sweep["selected_threshold"] = float(decision_threshold)
        sweep["is_selected_threshold"] = np.isclose(sweep["threshold"].astype(float), float(decision_threshold))
        threshold_sweep_frames.append(_with_context(sweep, {**context, "decision_threshold": float(decision_threshold)}))
    if save_prediction_artifacts:
        test_prediction_frames.append(
            _prediction_frame(
                context=context,
                split="test",
                row_ids=test_row_ids,
                y_true=test_labels,
                raw_probabilities=raw_probabilities,
                calibrated_probabilities=probabilities,
                decision_threshold=float(decision_threshold),
                metadata=test_meta,
            )
        )
        calibration_prediction_frames.append(
            _prediction_frame(
                context=context,
                split="calibration",
                row_ids=calib_row_ids,
                y_true=calib_labels,
                raw_probabilities=raw_calib_probabilities,
                calibrated_probabilities=calib_probabilities,
                decision_threshold=float(decision_threshold),
                metadata=calib_meta,
            )
        )
    return metric_row


def _context(
    cfg: dict[str, Any],
    run_type: str,
    model: str,
    calibration: str,
    algorithm: str | None = None,
    client_id: int | None = None,
) -> dict[str, Any]:
    return {
        "dataset": cfg["dataset"],
        "seed": int(cfg["seed"]),
        "clients": int(cfg["clients"]),
        "partition": cfg["partition"],
        "alpha": float(cfg["alpha"]),
        "run_type": run_type,
        "algorithm": algorithm or "not_applicable",
        "model": model,
        "calibration": calibration,
        "threshold_strategy": cfg.get("decision_threshold_strategy", "fixed_0p5"),
        "experiment_track": cfg.get("experiment_track", "audit"),
        "client_id": client_id if client_id is not None else "global",
    }


def _with_context(frame: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    output = frame.copy()
    for key, value in reversed(list(context.items())):
        if key in output.columns:
            output[key] = value
        else:
            output.insert(0, key, value)
    return output


def _local_split(indices: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    if len(indices) < 8:
        return indices, indices
    labels = y[indices]
    stratify = labels if len(np.unique(labels)) > 1 and min(np.bincount(labels, minlength=2)) >= 2 else None
    train_idx, val_idx = train_test_split(indices, test_size=0.25, random_state=seed, stratify=stratify)
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


def _save_client_manifest(bundle: DatasetBundle, client_indices: list[np.ndarray], output_dir: Path) -> None:
    rows = []
    train_row_ids = np.asarray(bundle.train_indices)
    for client_id, indices in enumerate(client_indices):
        labels = bundle.y_train[indices]
        for row_idx, label in zip(indices, labels):
            rows.append(
                {
                    "client_id": client_id,
                    "train_index": int(row_idx),
                    "source_row_id": int(train_row_ids[row_idx]),
                    "label": int(label),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "client_manifest.csv", index=False)


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _best_metric_row(metrics: pd.DataFrame) -> dict[str, Any]:
    if metrics.empty:
        return {}
    sortable = metrics.copy()
    sortable["roc_auc"] = pd.to_numeric(sortable["roc_auc"], errors="coerce").fillna(-1)
    sortable["f1"] = pd.to_numeric(sortable["f1"], errors="coerce").fillna(-1)
    sortable["ece"] = pd.to_numeric(sortable["ece"], errors="coerce").fillna(np.inf)
    row = sortable.sort_values(["roc_auc", "f1", "ece"], ascending=[False, False, True]).iloc[0]
    return {column: _json_safe(row[column]) for column in metrics.columns if column in row.index}


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _metric_names() -> set[str]:
    return {"accuracy", "precision", "recall", "f1", "pr_auc", "brier", "log_loss", "ece", "roc_auc", "selection_rate"}


def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _prediction_frame(
    context: dict[str, Any],
    split: str,
    row_ids: list[int],
    y_true: np.ndarray,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    decision_threshold: float,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "split": split,
            "row_id": np.asarray(row_ids, dtype=int),
            "true_label": np.asarray(y_true, dtype=int),
            "raw_probability": np.clip(np.asarray(raw_probabilities, dtype=float), 1e-6, 1.0 - 1e-6),
            "calibrated_probability": np.clip(np.asarray(calibrated_probabilities, dtype=float), 1e-6, 1.0 - 1e-6),
            "selected_threshold": float(decision_threshold),
        }
    )
    rows["predicted_label"] = (rows["calibrated_probability"] >= float(decision_threshold)).astype(int)
    if metadata is not None and not metadata.empty:
        meta = metadata.reset_index(drop=True).copy()
        for column in meta.columns:
            rows[column] = meta[column].astype(str)
    return _with_context(rows, context)


def _selection_key(scores: dict[str, Any], profile: str = "audit") -> tuple[float, ...]:
    profile = str(profile or "audit").strip().lower()
    auc = float(scores.get("roc_auc", np.nan))
    auc_key = auc if not np.isnan(auc) else -1.0
    f1 = float(scores.get("f1", np.nan))
    f1_key = f1 if not np.isnan(f1) else -1.0
    brier = float(scores.get("brier", np.inf))
    ece = float(scores.get("ece", np.inf))
    if profile == "showcase":
        return f1_key, auc_key, -brier, -ece
    return auc_key, -brier
