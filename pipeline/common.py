"""Shared helpers: repo root, config loading, out/ paths."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict[str, Any]:
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_path(rel: str) -> Path:
    return (ROOT / rel).resolve()


def out_dir(cfg: dict[str, Any]) -> Path:
    """Output folder: env VALMET_OUT overrides config paths.out_dir (keeps baseline/candidate runs apart)."""
    return ROOT / os.environ.get("VALMET_OUT", cfg["paths"]["out_dir"])


def out_path(cfg: dict[str, Any], name: str) -> Path:
    d = out_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    return d / name


DEFAULT_VISION_SOURCES = ("featherless_vision", "gemini_vision")


def is_vision_label(a: dict[str, Any], cfg: dict[str, Any], labels: Any = None, min_conf: float = 0.0) -> bool:
    """True if the object's label comes from a supported vision model (any provider) with enough
    confidence. Replaces hard-coded `label_source == "gemini_vision"` checks downstream."""
    srcs = cfg.get("label", {}).get("vision_sources", DEFAULT_VISION_SOURCES)
    if a.get("label_source") not in srcs:
        return False
    if labels is not None and a.get("label") not in labels:
        return False
    c = a.get("label_confidence")
    return isinstance(c, (int, float)) and c == c and 0.0 <= float(c) <= 1.0 and float(c) >= min_conf
