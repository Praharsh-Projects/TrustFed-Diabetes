from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_hf_space_bundle import BUNDLE_ROOT, build_bundle


TOKEN_ENV_VARS = ["HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGING_FACE_HUB_TOKEN"]


def _load_token() -> str | None:
    for name in TOKEN_ENV_VARS:
        value = os.getenv(name)
        if value:
            return value
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or update the public Hugging Face Space deployment.")
    parser.add_argument("--space-id", required=True, help="Hugging Face Space id in the form owner/space-name")
    parser.add_argument("--bundle-dir", default=str(BUNDLE_ROOT))
    return parser.parse_args()


def main() -> None:
    token = _load_token()
    if token is None:
        raise SystemExit(
            "No Hugging Face token found. Set one of: HF_TOKEN, HUGGINGFACEHUB_API_TOKEN, HUGGING_FACE_HUB_TOKEN."
        )
    try:
        from huggingface_hub import HfApi, upload_folder
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Run: py -m pip install huggingface_hub"
        ) from exc

    args = parse_args()
    bundle_dir = Path(args.bundle_dir).resolve()
    build_bundle(bundle_dir)

    api = HfApi(token=token)
    api.create_repo(repo_id=args.space_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    upload_folder(
        repo_id=args.space_id,
        repo_type="space",
        folder_path=str(bundle_dir),
        token=token,
        commit_message="Deploy TrustFed-Diabetes live dashboard",
    )
    print(f"Published Space: https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
