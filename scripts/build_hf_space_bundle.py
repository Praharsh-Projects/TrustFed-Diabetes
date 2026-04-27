from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = ROOT / "deploy" / "huggingface-space"
SUMMARY_ROOT = ROOT / "results" / "full_cdc_polished_summary"

RUNTIME_FILES = [
    "dashboard_metrics.csv",
    "dashboard_calibration.csv",
    "dashboard_fairness.csv",
    "dashboard_rounds.csv",
    "dashboard_shap.csv",
    "dashboard_stability.csv",
    "dashboard_thresholds.csv.gz",
    "dashboard_local_explanations.csv",
    "dashboard_curves.csv",
    "dashboard_confusion.csv",
    "dashboard_showcase_metrics.csv",
    "dashboard_score_ceiling.csv",
]


SPACE_README = """---
title: TrustFed-Diabetes Live
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8050
pinned: false
---

# TrustFed-Diabetes Live

This Space hosts the final precomputed TrustFed-Diabetes dashboard.

- No model training runs at startup.
- The dashboard reads bundled summary files only.
- The live app mirrors the final no-training review experience from the GitHub project.
"""


SPACE_DOCKERIGNORE = """__pycache__
*.py[cod]
.pytest_cache
.mypy_cache
.ruff_cache
"""


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_bundle(bundle_root: Path = BUNDLE_ROOT) -> Path:
    _reset_dir(bundle_root)

    _copy_file(ROOT / "app.py", bundle_root / "app.py")
    _copy_file(ROOT / "Dockerfile", bundle_root / "Dockerfile")
    _copy_file(ROOT / "requirements.txt", bundle_root / "requirements.txt")

    (bundle_root / "README.md").write_text(SPACE_README, encoding="utf-8")
    (bundle_root / ".dockerignore").write_text(SPACE_DOCKERIGNORE, encoding="utf-8")

    shutil.copytree(
        ROOT / "src" / "fl_diabetes",
        bundle_root / "src" / "fl_diabetes",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for cache_dir in bundle_root.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)
    for pyc_file in bundle_root.rglob("*.pyc"):
        pyc_file.unlink(missing_ok=True)

    target_results = bundle_root / "results" / "full_cdc_polished_summary"
    target_results.mkdir(parents=True, exist_ok=True)
    for filename in RUNTIME_FILES:
        _copy_file(SUMMARY_ROOT / filename, target_results / filename)

    return bundle_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the deployable Hugging Face Space bundle.")
    parser.add_argument("--output-dir", default=str(BUNDLE_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_root = Path(args.output_dir).resolve()
    built = build_bundle(bundle_root)
    print(f"Hugging Face Space bundle created at {built}")


if __name__ == "__main__":
    main()
