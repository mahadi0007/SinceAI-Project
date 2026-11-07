"""Version stamps for derived BIM / edit artifacts, so the viewer never mixes scene versions.

A scene revision = source PLY checksum (segment_manifest.json) + segmentation revision (hash of the label array,
the same one navgrid writes into path.json) + object count. Every derived artifact (scene.ifc, removable.json,
scene_edits.json) records the revision it was built from; the viewer rejects an artifact whose revision differs.

Coordinates: the scene is -Y up in scan units; IFC is +Z up in metres:  ifc = units_to_m * (x, z, -y).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.common import out_dir, out_path

Cfg = dict[str, Any]
COORDS = "ifc = units_to_m * (x, z, -y); scene -Y up (scan units), IFC +Z up (metres)"


class ArtifactMismatch(RuntimeError):
    """A derived artifact was built from another scene version (or is incomplete)."""


def label_revision(labels: np.ndarray) -> str:
    """Segmentation revision: OBJ ids are only meaningful within one label array (see navgrid)."""
    return hashlib.sha1(np.ascontiguousarray(labels).tobytes()).hexdigest()[:12]


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scene_revision(cfg: Cfg, labels: np.ndarray | None = None) -> dict[str, Any]:
    man = json.loads(out_path(cfg, "segment_manifest.json").read_text(encoding="utf-8"))
    if labels is None:
        labels = np.load(out_path(cfg, "segment_points.npz"))["labels"]
    sg = json.loads(out_path(cfg, "scene_graph.json").read_text(encoding="utf-8"))
    return {"ply_sha256": man["ply_sha256"], "segmentation": label_revision(labels),
            "n_objects": len(sg["objects"]), "units_to_m": cfg["scene"]["units_to_m"],
            "up": cfg["scene"]["up_direction"]}


def compare(have: dict[str, Any] | None, want: dict[str, Any], what: str) -> list[str]:
    """Differences between an artifact's recorded revision and the current scene's ([] = compatible)."""
    if not have:
        return [f"{what}: no scene revision recorded (built before versioning; regenerate it)"]
    out = []
    for k in ("ply_sha256", "segmentation", "n_objects", "units_to_m", "up"):
        if have.get(k) != want.get(k):
            out.append(f"{what}: {k} {str(have.get(k))[:16]} != current {str(want.get(k))[:16]}")
    return out


def backup(cfg: Cfg, names: list[str], tag: str) -> Path | None:
    """Copy existing outputs to out/_backup/<tag>_<time>/ before they are regenerated."""
    have = [n for n in names if (out_dir(cfg) / n).exists()]
    if not have:
        return None
    dst = out_dir(cfg) / "_backup" / f"{tag}_{time.strftime('%Y%m%d-%H%M%S')}"
    dst.mkdir(parents=True, exist_ok=True)
    for n in have:
        shutil.copy2(out_dir(cfg) / n, dst / n)
    print(f"[backup] {have} -> {dst}")
    return dst


def write_bim_manifest(cfg: Cfg, rev: dict[str, Any], object_ids: list[str], stats: dict[str, Any]) -> None:
    files = {n: file_sha256(out_path(cfg, n)) for n in ("scene.ifc", "deviation.npz")}
    out_path(cfg, "bim_manifest.json").write_text(json.dumps(
        {"version": 1, "kind": "bim", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "revision": rev,
         "coords": COORDS, "files": files, "object_ids": object_ids, "stats": stats}, indent=1), encoding="utf-8")


def check_bim(cfg: Cfg, rev: dict[str, Any], hash_files: bool = True) -> dict[str, Any]:
    """bim_manifest.json for the current scene, or ArtifactMismatch with every reason."""
    p = out_path(cfg, "bim_manifest.json")
    if not out_path(cfg, "scene.ifc").exists():
        raise ArtifactMismatch(f"no scene.ifc in {out_dir(cfg).name}/ (run python -m pipeline.export_ifc)")
    if not p.exists():
        raise ArtifactMismatch(f"scene.ifc in {out_dir(cfg).name}/ has no bim_manifest.json: unknown scene version; "
                               "regenerate with python -m pipeline.export_ifc")
    m = json.loads(p.read_text(encoding="utf-8"))
    errs = compare(m.get("revision"), rev, "scene.ifc")
    if hash_files and not errs:
        for n, h in m.get("files", {}).items():
            if not out_path(cfg, n).exists() or file_sha256(out_path(cfg, n)) != h:
                errs.append(f"{n} changed after bim_manifest.json was written")
    if errs:
        raise ArtifactMismatch("; ".join(errs))
    return m
