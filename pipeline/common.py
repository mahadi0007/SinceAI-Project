"""Shared helpers: repo root, config loading, out/ paths."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict[str, Any]:
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_path(rel: str) -> Path:
    return ROOT / rel


def out_path(cfg: dict[str, Any], name: str) -> Path:
    d = ROOT / cfg["paths"]["out_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d / name
