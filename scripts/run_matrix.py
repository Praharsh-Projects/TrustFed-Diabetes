from __future__ import annotations

import argparse
import itertools
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
    parser = argparse.ArgumentParser(description="Run a reproducible experiment matrix.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for quick validation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = load_json_config(args.config)
    base_output_dir = Path(matrix.get("base_output_dir", "results/runs"))
    run_summaries = []
    failures = []
    reused_existing = []

    combinations = list(_iter_matrix(matrix))
    if args.limit is not None:
        combinations = combinations[: args.limit]

    for run_config in combinations:
        output_dir = base_output_dir / _run_id(run_config)
        run_config["output_dir"] = str(output_dir)
        existing_summary = output_dir / "summary.json"
        if existing_summary.exists():
            run_summaries.append(json.loads(existing_summary.read_text(encoding="utf-8")))
            reused_existing.append(str(output_dir))
            print(f"reused existing {output_dir}", flush=True)
            continue
        try:
            artifacts = run_single_experiment(run_config)
            run_summaries.append(artifacts.summary)
            print(f"completed {output_dir}", flush=True)
        except FileNotFoundError as exc:
            if matrix.get("skip_missing_datasets", True):
                failures.append({"run": str(output_dir), "reason": str(exc)})
                print(f"skipped {output_dir}: {exc}", flush=True)
                continue
            raise

    summary = {
        "completed": len(run_summaries),
        "reused_existing": reused_existing,
        "skipped": failures,
        "runs": run_summaries,
    }
    base_output_dir.mkdir(parents=True, exist_ok=True)
    (base_output_dir / "matrix_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"completed": len(run_summaries), "reused_existing": len(reused_existing), "skipped": len(failures)}, indent=2))


def _iter_matrix(matrix: dict) -> list[dict]:
    base = {
        key: value
        for key, value in matrix.items()
        if key
        not in {
            "name",
            "base_output_dir",
            "datasets",
            "seeds",
            "clients",
            "partitions",
            "decision_threshold_strategies",
            "skip_missing_datasets",
        }
    }
    datasets = matrix.get("datasets", [{"dataset": matrix.get("dataset", "synthetic")}])
    seeds = _as_list(matrix.get("seeds", matrix.get("seed", 42)))
    clients = _as_list(matrix.get("clients", 5))
    partitions = _as_list(matrix.get("partitions", {"partition": matrix.get("partition", "iid"), "alpha": matrix.get("alpha", 0.5)}))
    threshold_strategies = _as_list(matrix.get("decision_threshold_strategies", matrix.get("decision_threshold_strategy", "fixed_0p5")))
    runs = []
    for dataset_cfg, seed, client_count, partition_cfg, threshold_strategy in itertools.product(
        datasets,
        seeds,
        clients,
        partitions,
        threshold_strategies,
    ):
        cfg = dict(DEFAULT_EXPERIMENT)
        cfg.update(base)
        cfg.update(dataset_cfg)
        cfg.update(partition_cfg)
        cfg["seed"] = seed
        cfg["clients"] = client_count
        cfg["decision_threshold_strategy"] = threshold_strategy
        runs.append(cfg)
    return runs


def _run_id(config: dict) -> str:
    partition = config["partition"]
    alpha = str(config.get("alpha", 0.5)).replace(".", "p")
    threshold = str(config.get("decision_threshold_strategy", "fixed_0p5")).replace(".", "p")
    return f"{config['dataset']}_seed{config['seed']}_c{config['clients']}_{partition}_a{alpha}_{threshold}"


def _as_list(value):
    return value if isinstance(value, list) else [value]


if __name__ == "__main__":
    main()
