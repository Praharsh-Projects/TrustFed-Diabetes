"""Configuration helpers for reproducible experiment runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")
    return config


def merge_config(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def config_snapshot(config: dict[str, Any]) -> str:
    return json.dumps(config, indent=2, sort_keys=True)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
