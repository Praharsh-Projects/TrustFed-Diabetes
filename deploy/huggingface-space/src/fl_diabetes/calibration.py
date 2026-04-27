"""Probability calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _as_probability_column(probabilities: Sequence[float]) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    return np.clip(values, 1e-6, 1.0 - 1e-6)


@dataclass
class ProbabilityCalibrator:
    """Post-hoc binary probability calibrator."""

    method: str = "isotonic"
    fitted_: bool = False
    estimator_: object | None = None

    def fit(self, probabilities: Sequence[float], y_true: Sequence[int]) -> "ProbabilityCalibrator":
        method = self.method.lower()
        probabilities = _as_probability_column(probabilities)
        y_true = np.asarray(y_true, dtype=int).reshape(-1)

        if method in {"none", "identity", "uncalibrated"} or len(np.unique(y_true)) < 2:
            self.method = "none"
            self.fitted_ = True
            self.estimator_ = None
            return self

        if method == "isotonic":
            estimator = IsotonicRegression(out_of_bounds="clip")
            estimator.fit(probabilities, y_true)
        elif method == "sigmoid":
            estimator = LogisticRegression(solver="lbfgs")
            estimator.fit(probabilities.reshape(-1, 1), y_true)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.estimator_ = estimator
        self.fitted_ = True
        return self

    def transform(self, probabilities: Sequence[float]) -> np.ndarray:
        probabilities = _as_probability_column(probabilities)
        if not self.fitted_:
            raise RuntimeError("Calibrator must be fitted before transform().")
        if self.method == "none" or self.estimator_ is None:
            return probabilities
        if self.method == "isotonic":
            calibrated = self.estimator_.predict(probabilities)
        else:
            calibrated = self.estimator_.predict_proba(probabilities.reshape(-1, 1))[:, 1]
        return np.clip(calibrated, 1e-6, 1.0 - 1e-6)

    def fit_transform(self, probabilities: Sequence[float], y_true: Sequence[int]) -> np.ndarray:
        return self.fit(probabilities, y_true).transform(probabilities)


@dataclass
class FederatedCalibrator:
    """Aggregates client-side calibrators without sharing raw validation rows."""

    method: str = "isotonic"
    calibrators_: list[ProbabilityCalibrator] | None = None
    weights_: np.ndarray | None = None

    def fit(
        self,
        client_probabilities: Iterable[Sequence[float]],
        client_labels: Iterable[Sequence[int]],
        weights: Sequence[float] | None = None,
    ) -> "FederatedCalibrator":
        calibrators: list[ProbabilityCalibrator] = []
        valid_weights: list[float] = []
        raw_weights = list(weights) if weights is not None else []

        for idx, (probabilities, labels) in enumerate(zip(client_probabilities, client_labels)):
            labels = np.asarray(labels, dtype=int)
            probabilities = _as_probability_column(probabilities)
            if len(labels) == 0 or len(np.unique(labels)) < 2:
                continue
            calibrator = ProbabilityCalibrator(self.method).fit(probabilities, labels)
            calibrators.append(calibrator)
            valid_weights.append(float(raw_weights[idx]) if raw_weights else float(len(labels)))

        if not calibrators:
            calibrators = [ProbabilityCalibrator("none").fit([0.25, 0.75], [0, 1])]
            valid_weights = [1.0]

        weights_array = np.asarray(valid_weights, dtype=float)
        weights_array = weights_array / weights_array.sum()
        self.calibrators_ = calibrators
        self.weights_ = weights_array
        return self

    def transform(self, probabilities: Sequence[float]) -> np.ndarray:
        if self.calibrators_ is None or self.weights_ is None:
            raise RuntimeError("FederatedCalibrator must be fitted before transform().")
        probabilities = _as_probability_column(probabilities)
        calibrated = np.zeros_like(probabilities, dtype=float)
        for weight, calibrator in zip(self.weights_, self.calibrators_):
            calibrated += weight * calibrator.transform(probabilities)
        return np.clip(calibrated, 1e-6, 1.0 - 1e-6)
