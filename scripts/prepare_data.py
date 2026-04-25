from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fl_diabetes.config import load_json_config
from fl_diabetes.data import dataset_profile, load_dataset, save_dataset_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and profile configured public datasets.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="results/prepared")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = []
    for dataset_cfg in config.get("datasets", [{"dataset": config.get("dataset", "synthetic")}]):
        try:
            bundle = load_dataset(
                dataset=dataset_cfg["dataset"],
                data_path=dataset_cfg.get("data_path"),
                seed=int(config.get("seed", 42)),
                max_rows=dataset_cfg.get("max_rows"),
            )
        except FileNotFoundError as exc:
            if config.get("skip_missing_datasets", True):
                print(f"Skipping missing dataset: {exc}")
                continue
            raise
        dataset_dir = output_dir / bundle.name
        save_dataset_manifest(bundle, dataset_dir)
        profiles.append(dataset_profile(bundle))
    (output_dir / "dataset_profiles.json").write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    print(json.dumps({"prepared": [profile["dataset"] for profile in profiles], "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
