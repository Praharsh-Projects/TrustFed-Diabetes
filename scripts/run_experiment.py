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
from fl_diabetes.experiment import DEFAULT_EXPERIMENT, run_single_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one federated diabetes trustworthiness experiment.")
    parser.add_argument("--config", default=None, help="JSON config file. CLI options override file values.")
    parser.add_argument("--dataset", choices=["synthetic", "pima", "cdc", "early_stage"], default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--clients", type=int, default=None)
    parser.add_argument("--partition", choices=["iid", "non_iid"], default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--algorithm", choices=["fedavg", "fedprox"], default=None)
    parser.add_argument("--federated-model", choices=["logistic", "mlp"], default=None)
    parser.add_argument("--calibration", choices=["none", "sigmoid", "isotonic"], action="append", default=None)
    parser.add_argument("--decision-threshold-strategy", choices=["fixed_0p5", "calib_f1_optimal", "calib_youden_j"], default=None)
    parser.add_argument("--synthetic-samples", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--no-static-dashboard", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = dict(DEFAULT_EXPERIMENT)
    if args.config:
        config.update(load_json_config(args.config))

    overrides = {
        "dataset": args.dataset,
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "clients": args.clients,
        "partition": args.partition,
        "alpha": args.alpha,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "decision_threshold_strategy": args.decision_threshold_strategy,
        "synthetic_samples": args.synthetic_samples,
        "max_rows": args.max_rows,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if args.algorithm:
        config["algorithms"] = [args.algorithm]
    if args.federated_model:
        config["federated_models"] = [args.federated_model]
    if args.calibration:
        config["calibrations"] = args.calibration
    if args.no_static_dashboard:
        config["build_static_dashboard"] = False

    artifacts = run_single_experiment(config)
    print(json.dumps(artifacts.summary, indent=2))


if __name__ == "__main__":
    main()
