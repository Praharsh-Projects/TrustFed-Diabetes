from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"


UCI_STATIC_URLS = {
    "cdc": "https://archive.ics.uci.edu/static/public/891/cdc+diabetes+health+indicators.zip",
    "early_stage": "https://archive.ics.uci.edu/static/public/529/early+stage+diabetes+risk+prediction+dataset.zip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public datasets when sources allow direct access.")
    parser.add_argument("--dataset", choices=["pima", "cdc", "early_stage", "all"], required=True)
    parser.add_argument("--output-dir", default=str(DATA_RAW))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = ["pima", "cdc", "early_stage"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        if dataset == "pima":
            _download_pima(output_dir)
        elif dataset in UCI_STATIC_URLS:
            _download_uci_zip(dataset, output_dir)


def _download_pima(output_dir: Path) -> None:
    try:
        from sklearn.datasets import fetch_openml

        frame = fetch_openml(name="diabetes", version=1, as_frame=True).frame
        target_col = "class" if "class" in frame.columns else frame.columns[-1]
        frame = frame.rename(columns={target_col: "Outcome"})
        frame["Outcome"] = frame["Outcome"].astype(str).str.lower().map(
            {"tested_positive": 1, "positive": 1, "1": 1, "tested_negative": 0, "negative": 0, "0": 0}
        ).fillna(frame["Outcome"])
        path = output_dir / "pima_diabetes.csv"
        frame.to_csv(path, index=False)
        print(f"Downloaded Pima/OpenML dataset to {path}")
    except Exception as exc:
        print(
            "Could not download Pima automatically. Place a public Pima CSV at "
            f"{output_dir / 'pima_diabetes.csv'}. Error: {exc}",
            file=sys.stderr,
        )


def _download_uci_zip(dataset: str, output_dir: Path) -> None:
    if _download_with_ucimlrepo(dataset, output_dir):
        return
    url = UCI_STATIC_URLS[dataset]
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / f"{dataset}.zip"
        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(tmp)
        except Exception as exc:
            print(f"Could not download {dataset} from {url}. Error: {exc}", file=sys.stderr)
            return

        csv_files = list(Path(tmp).rglob("*.csv"))
        if not csv_files:
            print(f"No CSV found inside {url}.", file=sys.stderr)
            return
        source = csv_files[0]
        destination = output_dir / _default_filename(dataset)
        shutil.copy2(source, destination)
        print(f"Downloaded {dataset} dataset to {destination}")


def _download_with_ucimlrepo(dataset: str, output_dir: Path) -> bool:
    dataset_ids = {"cdc": 891, "early_stage": 529}
    try:
        from ucimlrepo import fetch_ucirepo
    except ModuleNotFoundError:
        return False
    try:
        fetched = fetch_ucirepo(id=dataset_ids[dataset])
        features = fetched.data.features
        targets = fetched.data.targets
        frame = features.copy()
        if targets is not None:
            for column in targets.columns:
                frame[column] = targets[column]
        destination = output_dir / _default_filename(dataset)
        frame.to_csv(destination, index=False)
        print(f"Downloaded {dataset} dataset to {destination}")
        return True
    except Exception as exc:
        print(f"ucimlrepo download failed for {dataset}: {exc}", file=sys.stderr)
        return False


def _default_filename(dataset: str) -> str:
    if dataset == "cdc":
        return "cdc_diabetes_health_indicators.csv"
    if dataset == "early_stage":
        return "early_stage_diabetes_risk_prediction.csv"
    return f"{dataset}.csv"


if __name__ == "__main__":
    main()
