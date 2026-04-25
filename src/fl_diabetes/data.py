"""Dataset loading, preprocessing, and client partitioning."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


PIMA_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]


@dataclass
class DatasetBundle:
    name: str
    feature_names: list[str]
    X_train: np.ndarray
    X_calib: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_calib: np.ndarray
    y_test: np.ndarray
    train_meta: pd.DataFrame
    calib_meta: pd.DataFrame
    test_meta: pd.DataFrame
    target_name: str = "target"
    source_path: str | None = None
    raw_shape: tuple[int, int] = (0, 0)
    train_indices: list[int] = field(default_factory=list)
    calib_indices: list[int] = field(default_factory=list)
    test_indices: list[int] = field(default_factory=list)
    preprocessing_summary: dict[str, Any] = field(default_factory=dict)


def load_dataset(
    dataset: Literal["synthetic", "pima", "cdc", "early_stage"] = "synthetic",
    data_path: str | Path | None = None,
    seed: int = 42,
    n_samples: int = 1200,
    max_rows: int | None = None,
) -> DatasetBundle:
    if dataset == "synthetic":
        features, target, source_path, target_name = _make_synthetic_diabetes(n_samples=n_samples, seed=seed)
        name = "synthetic"
    elif dataset == "pima":
        features, target, source_path, target_name = _load_pima(data_path)
        name = "pima"
    elif dataset == "cdc":
        features, target, source_path, target_name = _load_cdc(data_path)
        name = "cdc"
    elif dataset == "early_stage":
        features, target, source_path, target_name = _load_early_stage(data_path)
        name = "early_stage"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if max_rows is not None and len(features) > max_rows:
        features, target = _limit_rows(features, target, max_rows=max_rows, seed=seed)

    features = _normalise_feature_frame(features)
    return _split_and_preprocess(
        features,
        target,
        name=name,
        seed=seed,
        source_path=source_path,
        target_name=target_name,
    )


def _make_synthetic_diabetes(n_samples: int, seed: int) -> tuple[pd.DataFrame, pd.Series, str | None, str]:
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 82, n_samples)
    bmi = np.clip(rng.normal(29.5, 6.5, n_samples), 17.5, 52.0)
    glucose = np.clip(rng.normal(118.0, 32.0, n_samples), 55.0, 230.0)
    blood_pressure = np.clip(rng.normal(72.0, 12.0, n_samples), 42.0, 122.0)
    skin = np.clip(rng.normal(28.0, 9.0, n_samples), 7.0, 60.0)
    insulin = np.clip(rng.lognormal(mean=4.65, sigma=0.55, size=n_samples), 15.0, 420.0)
    pregnancies = np.clip(((age - 18) / 9.5 + rng.normal(0, 1.5, n_samples)).round(), 0, 15)
    pedigree = np.clip(rng.gamma(shape=2.1, scale=0.22, size=n_samples), 0.05, 2.5)

    logit = (
        -8.4
        + 0.036 * glucose
        + 0.080 * bmi
        + 0.018 * age
        + 0.006 * blood_pressure
        + 0.36 * pedigree
        + 0.035 * pregnancies
        + rng.normal(0, 0.85, n_samples)
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    target = rng.binomial(1, probability)

    features = pd.DataFrame(
        {
            "Pregnancies": pregnancies.astype(float),
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": age.astype(float),
        }
    )
    return features, pd.Series(target, name="Outcome"), None, "Outcome"


def _load_pima(data_path: str | Path | None) -> tuple[pd.DataFrame, pd.Series, str, str]:
    if data_path is None:
        data_path = Path("data/raw/pima_diabetes.csv")
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Pima dataset not found at {path}. Place the CSV there or run with --dataset synthetic."
        )

    frame = pd.read_csv(path)
    if len(frame.columns) == 9 and not set(PIMA_COLUMNS).issubset(frame.columns):
        frame.columns = PIMA_COLUMNS

    target_col = "Outcome" if "Outcome" in frame.columns else frame.columns[-1]
    target = frame[target_col].astype(int)
    features = frame.drop(columns=[target_col]).copy()

    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for column in zero_as_missing:
        if column in features.columns:
            features.loc[features[column] == 0, column] = np.nan
    return features, target, str(path), target_col


def _load_cdc(data_path: str | Path | None) -> tuple[pd.DataFrame, pd.Series, str, str]:
    path = _resolve_data_path(
        data_path,
        candidates=[
            "data/raw/cdc_diabetes_health_indicators.csv",
            "data/raw/diabetes_binary_health_indicators_BRFSS2015.csv",
            "data/raw/diabetes_012_health_indicators_BRFSS2015.csv",
            "data/raw/cdc.csv",
        ],
        dataset_name="CDC Diabetes Health Indicators",
    )
    frame = pd.read_csv(path)
    target_col = _first_existing_column(frame, ["Diabetes_binary", "diabetes_binary", "Diabetes_012", "diabetes_012"])
    if target_col is None:
        target_col = frame.columns[-1]
    target = pd.to_numeric(frame[target_col], errors="coerce").fillna(0).astype(int)
    if target_col.lower() in {"diabetes_012", "diabetes_012"}:
        target = (target > 0).astype(int)

    features = frame.drop(columns=[target_col]).copy()
    for column in list(features.columns):
        if str(column).lower() in {"id", "patientid", "patient_id"}:
            features = features.drop(columns=[column])
    return features, pd.Series(target, name="Diabetes_binary"), str(path), target_col


def _load_early_stage(data_path: str | Path | None) -> tuple[pd.DataFrame, pd.Series, str, str]:
    path = _resolve_data_path(
        data_path,
        candidates=[
            "data/raw/early_stage_diabetes_risk_prediction.csv",
            "data/raw/diabetes_data_upload.csv",
            "data/raw/early_stage.csv",
        ],
        dataset_name="Early Stage Diabetes Risk Prediction",
    )
    frame = pd.read_csv(path)
    target_col = _first_existing_column(frame, ["class", "Class", "target", "Outcome"]) or frame.columns[-1]
    raw_target = frame[target_col].astype(str).str.strip().str.lower()
    target = raw_target.map({"positive": 1, "yes": 1, "1": 1, "negative": 0, "no": 0, "0": 0})
    if target.isna().any():
        target = pd.to_numeric(frame[target_col], errors="coerce")
    target = target.fillna(0).astype(int)
    features = frame.drop(columns=[target_col]).copy()
    return features, pd.Series(target, name="Outcome"), str(path), target_col


def _resolve_data_path(data_path: str | Path | None, candidates: list[str], dataset_name: str) -> Path:
    if data_path is not None:
        path = Path(data_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"{dataset_name} file not found at {path}.")
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    searched = ", ".join(candidates)
    raise FileNotFoundError(f"{dataset_name} file not found. Searched: {searched}.")


def _first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        column = lookup.get(candidate.lower())
        if column is not None:
            return column
    return None


def _limit_rows(
    features: pd.DataFrame,
    target: pd.Series,
    max_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    max_rows = int(max_rows)
    if max_rows <= 0:
        raise ValueError("max_rows must be positive when provided.")
    frame = features.copy()
    frame["_target"] = target.to_numpy()
    grouped = frame.groupby("_target", group_keys=False)
    sampled = grouped.apply(
        lambda group: group.sample(
            n=max(1, int(round(max_rows * len(group) / len(frame)))),
            random_state=seed,
            replace=False,
        )
    )
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=seed)
    sampled = sampled.sample(frac=1.0, random_state=seed)
    limited_target = sampled.pop("_target").astype(int)
    return sampled.reset_index(drop=True), limited_target.reset_index(drop=True)


def _normalise_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    frame.columns = [str(column) for column in frame.columns]
    for column in frame.columns:
        if _looks_numeric(frame[column]):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _split_and_preprocess(
    features: pd.DataFrame,
    target: pd.Series,
    name: str,
    seed: int,
    source_path: str | None,
    target_name: str,
) -> DatasetBundle:
    target = target.astype(int)
    features = features.reset_index(drop=True)
    target = target.reset_index(drop=True)
    original_indices = pd.Series(np.arange(len(features)), name="_row_id")
    x_temp, x_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
        features,
        target,
        original_indices,
        test_size=0.20,
        stratify=target,
        random_state=seed,
    )
    x_train, x_calib, y_train, y_calib, ids_train, ids_calib = train_test_split(
        x_temp,
        y_temp,
        ids_temp,
        test_size=0.25,
        stratify=y_temp,
        random_state=seed + 1,
    )

    preprocessor, feature_names = _make_preprocessor(x_train)
    X_train = preprocessor.fit_transform(x_train)
    X_calib = preprocessor.transform(x_calib)
    X_test = preprocessor.transform(x_test)

    return DatasetBundle(
        name=name,
        feature_names=feature_names,
        X_train=np.asarray(X_train, dtype=float),
        X_calib=np.asarray(X_calib, dtype=float),
        X_test=np.asarray(X_test, dtype=float),
        y_train=y_train.to_numpy(dtype=int),
        y_calib=y_calib.to_numpy(dtype=int),
        y_test=y_test.to_numpy(dtype=int),
        train_meta=_make_metadata(x_train.reset_index(drop=True)),
        calib_meta=_make_metadata(x_calib.reset_index(drop=True)),
        test_meta=_make_metadata(x_test.reset_index(drop=True)),
        target_name=target_name,
        source_path=source_path,
        raw_shape=(int(features.shape[0]), int(features.shape[1] + 1)),
        train_indices=[int(value) for value in ids_train.to_list()],
        calib_indices=[int(value) for value in ids_calib.to_list()],
        test_indices=[int(value) for value in ids_test.to_list()],
        preprocessing_summary=_preprocessing_summary(x_train),
    )


def _make_metadata(raw_features: pd.DataFrame) -> pd.DataFrame:
    meta = pd.DataFrame(index=raw_features.index)
    age_column = _find_column(raw_features, "age")
    bmi_column = _find_column(raw_features, "bmi")
    sex_column = _find_column(raw_features, "sex")

    if age_column:
        age_values = pd.to_numeric(raw_features[age_column], errors="coerce")
        if age_values.max(skipna=True) <= 13:
            bins = [0, 4, 8, 13]
            labels = ["under_40", "40_59", "60_plus"]
        else:
            bins = [0, 39, 59, 200]
            labels = ["under_40", "40_59", "60_plus"]
        meta["age_group"] = pd.cut(
            age_values,
            bins=bins,
            labels=labels,
            include_lowest=True,
        ).astype(str)
    if bmi_column:
        meta["bmi_category"] = pd.cut(
            pd.to_numeric(raw_features[bmi_column], errors="coerce"),
            bins=[0, 18.5, 25.0, 30.0, 200.0],
            labels=["underweight", "healthy", "overweight", "obese"],
            include_lowest=True,
        ).astype(str)
    if sex_column:
        sex_values = raw_features[sex_column].astype(str).str.strip()
        meta["sex"] = sex_values.replace({"0": "female", "0.0": "female", "1": "male", "1.0": "male"})
    return meta.reset_index(drop=True)


def _make_preprocessor(frame: pd.DataFrame) -> tuple[ColumnTransformer, list[str]]:
    numeric_columns = [
        str(column)
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column]) or _looks_numeric(frame[column])
    ]
    categorical_columns = [str(column) for column in frame.columns if str(column) not in set(numeric_columns)]

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError("Dataset has no usable feature columns.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    preprocessor.fit(frame)
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    return preprocessor, feature_names


def _looks_numeric(series: pd.Series) -> bool:
    converted = pd.to_numeric(series, errors="coerce")
    return converted.notna().mean() >= 0.95


def _preprocessing_summary(frame: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = [
        str(column)
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column]) or _looks_numeric(frame[column])
    ]
    categorical_columns = [str(column) for column in frame.columns if str(column) not in set(numeric_columns)]
    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "imputation": {
            "numeric": "median",
            "categorical": "most_frequent",
        },
        "scaling": "standard_scaler_for_numeric_columns",
        "encoding": "one_hot_encoder_for_categorical_columns",
    }


def dataset_profile(bundle: DatasetBundle) -> dict[str, Any]:
    y_all = np.concatenate([bundle.y_train, bundle.y_calib, bundle.y_test])
    return {
        "dataset": bundle.name,
        "source_path": bundle.source_path,
        "target_name": bundle.target_name,
        "raw_shape": list(bundle.raw_shape),
        "transformed_features": len(bundle.feature_names),
        "feature_names": bundle.feature_names,
        "class_counts": {str(label): int((y_all == label).sum()) for label in sorted(np.unique(y_all))},
        "splits": {
            "train": len(bundle.y_train),
            "calibration": len(bundle.y_calib),
            "test": len(bundle.y_test),
        },
        "metadata_columns": list(bundle.test_meta.columns),
        "preprocessing": bundle.preprocessing_summary,
        "limitations": _dataset_limitations(bundle.name),
    }


def save_dataset_manifest(bundle: DatasetBundle, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest = dataset_profile(bundle)
    manifest["split_indices"] = {
        "train": bundle.train_indices,
        "calibration": bundle.calib_indices,
        "test": bundle.test_indices,
    }
    path = output_path / "dataset_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _dataset_limitations(name: str) -> list[str]:
    if name == "pima":
        return [
            "Small benchmark dataset.",
            "All records are from women at least 21 years old of Pima Indian heritage.",
            "Several clinical zero values are treated as invalid/missing.",
        ]
    if name == "cdc":
        return [
            "Survey-derived BRFSS indicators, not direct clinical hospital records.",
            "Age is bucketed rather than exact.",
            "The binary target may combine prediabetes and diabetes depending on source file.",
        ]
    if name == "early_stage":
        return [
            "Small symptom-survey dataset.",
            "Categorical self-reported features limit clinical detail.",
        ]
    return ["Synthetic data is for smoke testing only and must not be used for thesis claims."]


def _find_column(frame: pd.DataFrame, name_fragment: str) -> str | None:
    needle = name_fragment.lower()
    for column in frame.columns:
        if needle in str(column).lower():
            return str(column)
    return None


def partition_clients(
    y: np.ndarray,
    n_clients: int,
    strategy: Literal["iid", "non_iid"] = "iid",
    alpha: float = 0.5,
    seed: int = 42,
) -> list[np.ndarray]:
    y = np.asarray(y, dtype=int)
    if n_clients < 2:
        raise ValueError("n_clients must be at least 2.")
    if strategy == "iid":
        return _iid_partition(y, n_clients=n_clients, seed=seed)
    if strategy == "non_iid":
        return _dirichlet_partition(y, n_clients=n_clients, alpha=alpha, seed=seed)
    raise ValueError(f"Unknown partition strategy: {strategy}")


def _iid_partition(y: np.ndarray, n_clients: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    clients: list[list[int]] = [[] for _ in range(n_clients)]
    for label in np.unique(y):
        label_indices = np.flatnonzero(y == label)
        rng.shuffle(label_indices)
        for client_id, split in enumerate(np.array_split(label_indices, n_clients)):
            clients[client_id].extend(split.tolist())
    return _shuffle_clients(clients, rng)


def _dirichlet_partition(y: np.ndarray, n_clients: int, alpha: float, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    alpha = max(float(alpha), 1e-3)
    best_clients: list[list[int]] | None = None

    for _ in range(100):
        clients = [[] for _ in range(n_clients)]
        for label in np.unique(y):
            label_indices = np.flatnonzero(y == label)
            rng.shuffle(label_indices)
            proportions = rng.dirichlet(np.repeat(alpha, n_clients))
            counts = rng.multinomial(len(label_indices), proportions)
            start = 0
            for client_id, count in enumerate(counts):
                clients[client_id].extend(label_indices[start : start + count].tolist())
                start += count
        best_clients = clients
        if min(len(client) for client in clients) >= 5:
            break
    assert best_clients is not None
    return _shuffle_clients(best_clients, rng)


def _shuffle_clients(clients: list[list[int]], rng: np.random.Generator) -> list[np.ndarray]:
    output = []
    for indices in clients:
        values = np.asarray(indices, dtype=int)
        rng.shuffle(values)
        output.append(values)
    return output
