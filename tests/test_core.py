from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from aggregate_results import _add_derived_threshold_metrics, _aggregate_score_ceiling
from fl_diabetes.calibration import ProbabilityCalibrator
from fl_diabetes.dashboard_app import _load_dashboard_data
from fl_diabetes.data import load_dataset, partition_clients
from fl_diabetes.experiment import run_single_experiment
from fl_diabetes.federated import train_federated_numpy
from fl_diabetes.metrics import classification_report_row, fairness_report, select_decision_threshold, threshold_sweep_frame
from fl_diabetes.models import fit_best_sklearn_model, make_sklearn_model


class CorePipelineTests(unittest.TestCase):
    def test_split_indices_do_not_overlap(self) -> None:
        bundle = load_dataset("synthetic", seed=7, n_samples=300)
        train = set(bundle.train_indices)
        calib = set(bundle.calib_indices)
        test = set(bundle.test_indices)
        self.assertFalse(train & calib)
        self.assertFalse(train & test)
        self.assertFalse(calib & test)

    def test_iid_partition_preserves_class_ratio_approximately(self) -> None:
        bundle = load_dataset("synthetic", seed=8, n_samples=400)
        global_rate = bundle.y_train.mean()
        clients = partition_clients(bundle.y_train, n_clients=4, strategy="iid", seed=8)
        for indices in clients:
            self.assertLess(abs(bundle.y_train[indices].mean() - global_rate), 0.12)

    def test_dirichlet_partition_is_reproducible(self) -> None:
        y = np.asarray([0] * 60 + [1] * 40)
        first = partition_clients(y, n_clients=5, strategy="non_iid", alpha=0.3, seed=9)
        second = partition_clients(y, n_clients=5, strategy="non_iid", alpha=0.3, seed=9)
        self.assertEqual([client.tolist() for client in first], [client.tolist() for client in second])

    def test_calibration_uses_supplied_validation_distribution(self) -> None:
        probabilities = np.asarray([0.05, 0.20, 0.35, 0.70, 0.90])
        labels = np.asarray([0, 0, 1, 1, 1])
        calibrator = ProbabilityCalibrator("sigmoid").fit(probabilities, labels)
        transformed = calibrator.transform(probabilities)
        self.assertEqual(transformed.shape, probabilities.shape)
        self.assertTrue(np.all((transformed > 0) & (transformed < 1)))

    def test_fairness_metrics_on_toy_data(self) -> None:
        y_true = np.asarray([0, 1, 0, 1])
        probabilities = np.asarray([0.1, 0.9, 0.8, 0.7])
        metadata = pd.DataFrame({"group": ["a", "a", "b", "b"]})
        report = fairness_report(y_true, probabilities, metadata)
        self.assertIn("demographic_parity_difference", report.columns)
        self.assertIn("equalized_odds_difference", report.columns)
        self.assertAlmostEqual(float(report["demographic_parity_difference"].iloc[0]), 0.5)

    def test_federated_byte_accounting(self) -> None:
        bundle = load_dataset("synthetic", seed=10, n_samples=240)
        clients = partition_clients(bundle.y_train, n_clients=3, strategy="iid", seed=10)
        result = train_federated_numpy(
            bundle.X_train,
            bundle.y_train,
            clients,
            bundle.feature_names,
            model_type="logistic",
            algorithm="fedavg",
            rounds=2,
            local_epochs=1,
            seed=10,
            X_eval=bundle.X_calib,
            y_eval=bundle.y_calib,
        )
        first_round = result.round_history.iloc[0]
        expected = int((bundle.X_train.shape[1] + 1) * 8 * 3 * 2)
        self.assertEqual(int(first_round["round_communication_bytes"]), expected)

    def test_threshold_strategy_can_move_off_point_five(self) -> None:
        y_true = np.asarray([0, 0, 0, 1, 1, 1])
        probabilities = np.asarray([0.20, 0.30, 0.45, 0.46, 0.47, 0.90])
        threshold = select_decision_threshold(y_true, probabilities, strategy="calib_f1_optimal")
        self.assertNotEqual(round(float(threshold), 3), 0.5)

    def test_threshold_sweep_metrics_stay_in_valid_range(self) -> None:
        y_true = np.asarray([0, 1, 0, 1, 1, 0])
        probabilities = np.asarray([0.1, 0.8, 0.3, 0.6, 0.9, 0.2])
        metadata = pd.DataFrame({"group": ["a", "a", "b", "b", "a", "b"]})
        sweep = threshold_sweep_frame(y_true, probabilities, metadata)
        self.assertTrue(sweep["f1"].between(0, 1).all())
        self.assertTrue(sweep["pr_auc"].between(0, 1).all())
        self.assertTrue(sweep["selection_rate"].between(0, 1).all())
        self.assertTrue(sweep["demographic_parity_difference"].dropna().ge(0).all())
        self.assertTrue(sweep["equalized_odds_difference"].dropna().ge(0).all())

    def test_classification_report_includes_pr_auc(self) -> None:
        y_true = np.asarray([0, 1, 0, 1, 1, 0])
        probabilities = np.asarray([0.1, 0.8, 0.3, 0.6, 0.9, 0.2])
        report = classification_report_row(y_true, probabilities, threshold=0.5)
        self.assertIn("pr_auc", report)
        self.assertIn("specificity", report)
        self.assertIn("balanced_accuracy", report)
        self.assertGreaterEqual(float(report["pr_auc"]), 0.0)
        self.assertLessEqual(float(report["pr_auc"]), 1.0)
        self.assertGreaterEqual(float(report["specificity"]), 0.0)
        self.assertLessEqual(float(report["specificity"]), 1.0)

    def test_threshold_aggregation_derives_specificity(self) -> None:
        thresholds = pd.DataFrame(
            [
                {
                    "dataset": "cdc",
                    "seed": 1,
                    "run_type": "federated",
                    "algorithm": "fedavg",
                    "model": "logistic",
                    "calibration": "none",
                    "threshold_strategy": "calib_f1_optimal",
                    "group_feature": "age_group",
                    "group_value": "aggregate",
                    "threshold": 0.2,
                    "accuracy": 0.80,
                    "precision": 0.50,
                    "recall": 0.75,
                    "selection_rate": 0.30,
                }
            ]
        )
        derived = _add_derived_threshold_metrics(thresholds)
        self.assertIn("specificity", derived.columns)
        self.assertIn("balanced_accuracy", derived.columns)
        self.assertTrue(derived["specificity"].between(0, 1).all())
        self.assertTrue(derived["balanced_accuracy"].between(0, 1).all())

    def test_score_ceiling_keeps_near99_recall_labeled_as_recall(self) -> None:
        thresholds = pd.DataFrame(
            [
                {
                    "dataset": "cdc",
                    "run_type": "federated",
                    "algorithm": "fedavg",
                    "model": "logistic",
                    "calibration": "none",
                    "threshold_strategy": "calib_f1_optimal",
                    "clients": "5",
                    "partition": "iid",
                    "alpha": "0.5",
                    "threshold": 0.01,
                    "group_feature": "age_group",
                    "group_value": "aggregate",
                    "accuracy_mean": 0.25,
                    "precision_mean": 0.15,
                    "recall_mean": 0.995,
                    "specificity_mean": 0.12,
                    "balanced_accuracy_mean": 0.557,
                    "f1_mean": 0.26,
                    "pr_auc_mean": 0.40,
                }
            ]
        )
        score = _aggregate_score_ceiling(pd.DataFrame(), thresholds, pd.DataFrame(), pd.DataFrame())
        near99 = score[pd.to_numeric(score["value"], errors="coerce") >= 0.99]
        self.assertFalse(near99.empty)
        self.assertTrue((near99["metric"].astype(str) == "recall").all())

    def test_xgboost_model_registration(self) -> None:
        model = make_sklearn_model("xgboost", seed=11)
        self.assertTrue(hasattr(model, "fit"))
        self.assertTrue(hasattr(model, "predict_proba"))

    def test_mlp_falls_back_when_early_stopping_split_is_invalid(self) -> None:
        rng = np.random.RandomState(12)
        X_train = rng.normal(size=(12, 5))
        y_train = np.asarray([0] * 11 + [1])
        X_val = rng.normal(size=(6, 5))
        y_val = np.asarray([0, 0, 0, 1, 1, 1])
        model, best_params, evaluations = fit_best_sklearn_model(
            "mlp",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            seed=12,
            tune=True,
        )
        self.assertTrue(hasattr(model, "predict_proba"))
        self.assertFalse(bool(best_params.get("early_stopping", True)))
        self.assertFalse(evaluations.empty)

    def test_dashboard_loader_reads_curated_tables(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "dataset": "cdc",
                        "clients": 3,
                        "partition": "non_iid",
                        "alpha": 0.5,
                        "run_type": "federated",
                        "algorithm": "fedavg",
                        "model": "logistic",
                        "calibration": "global_isotonic",
                        "threshold_strategy": "calib_f1_optimal",
                        "experiment_track": "showcase",
                        "roc_auc_mean": 0.81,
                        "roc_auc_std": 0.01,
                        "roc_auc_ci95_low": 0.80,
                        "roc_auc_ci95_high": 0.82,
                        "roc_auc_n": 5,
                        "f1_mean": 0.20,
                        "f1_std": 0.01,
                        "f1_ci95_low": 0.19,
                        "f1_ci95_high": 0.21,
                        "f1_n": 5,
                        "pr_auc_mean": 0.33,
                        "pr_auc_std": 0.02,
                        "pr_auc_ci95_low": 0.30,
                        "pr_auc_ci95_high": 0.35,
                        "pr_auc_n": 5,
                        "brier_mean": 0.11,
                        "brier_std": 0.01,
                        "brier_ci95_low": 0.10,
                        "brier_ci95_high": 0.12,
                        "brier_n": 5,
                        "ece_mean": 0.03,
                        "ece_std": 0.01,
                        "ece_ci95_low": 0.02,
                        "ece_ci95_high": 0.04,
                        "ece_n": 5,
                        "log_loss_mean": 0.31,
                        "log_loss_std": 0.01,
                        "log_loss_ci95_low": 0.30,
                        "log_loss_ci95_high": 0.32,
                        "log_loss_n": 5,
                    }
                ]
            ).to_csv(root / "dashboard_metrics.csv", index=False)
            for name in [
                "dashboard_calibration.csv",
                "dashboard_fairness.csv",
                "dashboard_rounds.csv",
                "dashboard_shap.csv",
                "dashboard_stability.csv",
                "dashboard_thresholds.csv",
                "dashboard_local_explanations.csv",
                "dashboard_curves.csv",
                "dashboard_confusion.csv",
                "dashboard_score_ceiling.csv",
            ]:
                pd.DataFrame().to_csv(root / name, index=False)
            pd.DataFrame(
                [
                    {
                        "dataset": "cdc",
                        "experiment_track": "showcase",
                        "run_type": "centralized",
                        "algorithm": "not_applicable",
                        "model": "xgboost",
                        "calibration": "global_isotonic",
                        "threshold_strategy": "calib_f1_optimal",
                        "showcase_role": "best_centralized_decision",
                        "roc_auc_mean": 0.81,
                        "f1_mean": 0.20,
                        "pr_auc_mean": 0.33,
                        "ece_mean": 0.03,
                    }
                ]
            ).to_csv(root / "dashboard_showcase_metrics.csv", index=False)
            data = _load_dashboard_data(root)
            self.assertEqual(len(data.metrics), 1)
            self.assertEqual(data.metrics["dataset_label"].iloc[0], "CDC Diabetes Indicators")
            self.assertEqual(len(data.showcase), 1)

    def test_showcase_experiment_writes_prediction_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "showcase_run"
            run_single_experiment(
                {
                    "dataset": "synthetic",
                    "seed": 13,
                    "clients": 3,
                    "partition": "iid",
                    "alpha": 0.5,
                    "centralized_models": ["logistic_regression"],
                    "local_only_models": [],
                    "federated_models": ["logistic"],
                    "algorithms": ["fedavg"],
                    "calibrations": ["none"],
                    "tune_models": False,
                    "tune_local_models": False,
                    "tune_federated": False,
                    "save_prediction_artifacts": True,
                    "experiment_track": "showcase",
                    "search_profile": "showcase",
                    "decision_threshold_strategy": "calib_f1_optimal",
                    "output_dir": str(output_dir),
                    "build_static_dashboard": False,
                }
            )
            test_predictions = pd.read_csv(output_dir / "test_predictions.csv")
            calibration_predictions = pd.read_csv(output_dir / "calibration_predictions.csv")
            required_columns = {
                "dataset",
                "experiment_track",
                "run_type",
                "algorithm",
                "model",
                "calibration",
                "threshold_strategy",
                "split",
                "row_id",
                "true_label",
                "raw_probability",
                "calibrated_probability",
                "selected_threshold",
                "predicted_label",
            }
            self.assertTrue(required_columns.issubset(set(test_predictions.columns)))
            self.assertTrue(required_columns.issubset(set(calibration_predictions.columns)))
            self.assertTrue(test_predictions["predicted_label"].isin([0, 1]).all())

    def test_showcase_core_config_uses_full_cdc(self) -> None:
        raw = (ROOT / "configs" / "showcase_core.json").read_text(encoding="utf-8")
        self.assertIn('"dataset": "cdc"', raw)
        self.assertNotIn('"max_rows"', raw.split('"dataset": "cdc"', 1)[1].split("}", 1)[0])


if __name__ == "__main__":
    unittest.main()
