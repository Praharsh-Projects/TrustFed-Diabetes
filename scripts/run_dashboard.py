from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fl_diabetes.dashboard_app import run_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the interactive Dash audit dashboard.")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--visual-results-dir", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_results = ROOT / "results" / "full_cdc_polished_summary"
    results_dir = args.results_dir or str(default_results)
    visual_results_dir = args.visual_results_dir
    run_dashboard(
        results_dir=results_dir,
        visual_results_dir=visual_results_dir,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
