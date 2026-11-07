"""Shared scene state for review / repair: full-resolution Gaussians with provenance.

Gaussian g (row in the filtered splat, original PLY row = row_ids[g]) -> voxel point src_vox[g]
(if within member_dist) -> fine fragment frag[vox] and segmentation label labels[vox]
(label = index into label_ids, i.e. into the FULL asset list incl. structures).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.common import ROOT, repo_path
from pipeline.load_splat import load_splat

Cfg = dict[str, Any]


@dataclass
class Scene:
    xyz: np.ndarray
    rgb: np.ndarray
    scales: np.ndarray
    opacity: np.ndarray
    quats: np.ndarray          # (N,4) wxyz, normalised
    row_ids: np.ndarray
    vox_xyz: np.ndarray
    vox_frag: np.ndarray       # fine fragment per voxel point (-1 noise, -2 structure)
    vox_label: np.ndarray      # index into assets / label_ids
    g_vox: np.ndarray          # voxel per Gaussian, -1 if farther than member_dist
    assets: list[dict]
    label_ids: np.ndarray
    seg_dir: Path

    @property
    def g_label(self) -> np.ndarray:
        return np.where(self.g_vox >= 0, self.vox_label[np.maximum(self.g_vox, 0)], -1)

    @property
    def g_frag(self) -> np.ndarray:
        return np.where(self.g_vox >= 0, self.vox_frag[np.maximum(self.g_vox, 0)], -1)


def load_scene(cfg: Cfg, seg_dir: str | Path) -> Scene:
    seg_dir = ROOT / seg_dir
    lc = cfg["load"]
    d = load_splat(str(repo_path(cfg["paths"]["ply"])), lc["min_opacity"], lc["max_scale"], lc["crop_floaters"])
    s = np.load(seg_dir / "segment_points.npz")
    assert len(s["src_vox"]) == len(d["xyz"]), "segment_points.npz does not match the loaded splat"
    assert (s["src_rows"] == d["row_ids"]).all(), "source row ids differ (different PLY or load filters)"
    g_vox = np.where(s["src_dist"] <= cfg["label"]["member_dist"], s["src_vox"], -1)
    assets = json.loads((seg_dir / "assets.json").read_text(encoding="utf-8"))
    p = seg_dir / "assets_labeled.json"
    if p.exists():                       # labelled copy only if it belongs to THIS segmentation version
        lab = json.loads(p.read_text(encoding="utf-8"))
        if [a["id"] for a in lab] == list(s["label_ids"]) and len(lab) == len(assets):
            assets = lab
    assert [a["id"] for a in assets] == list(s["label_ids"]), "assets do not match segment_points label_ids"
    return Scene(d["xyz"].astype(np.float64), d["rgb"], d["scales"], d["opacity"], d["quats_wxyz"], d["row_ids"],
                 s["xyz"].astype(np.float64), s["frag"], s["labels"], g_vox, assets, s["label_ids"], seg_dir)


def seg_revision(seg_dir: str | Path) -> str:
    """Segmentation revision: hash of the label array + asset ids (OBJ ids mean nothing across revisions)."""
    import hashlib
    s = np.load(ROOT / seg_dir / "segment_points.npz")
    h = hashlib.sha256(s["labels"].tobytes())
    h.update("|".join(map(str, s["label_ids"])).encode())
    return h.hexdigest()[:16]
