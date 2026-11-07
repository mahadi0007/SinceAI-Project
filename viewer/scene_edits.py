"""Shared, versioned object-layout edits for viewer.app (out/scene_edits.json) + adapters for the two older formats.

One rigid transform per shown instance, in the SCENE frame (-y up, scan units):
    p_shown = R(wxyz) @ (p - pivot) + position          pivot = the source object's scene-graph centre
An untouched original has position == pivot and identity rotation. Splat piece, BIM mesh, bounds and picking
all use this one transform. Copies are instances of a source object; they never carry its asset identity.

Older formats (read only; never rewritten):
  * out/viewer_edits.json (viewer.app before v1): {name: {oid, position, wxyz, removed}}, same frame and pivot.
  * out/edits.json (+ edits_pieces.npz) (viewer.bim_view): IFC frame (+z up, metres), keyed by object id,
    "position" = where the object's IFC point centroid goes, rotation about that centroid. Moves, removals and
    copies convert exactly; groups / detached / smart-selection pieces have no viewer.app equivalent and are kept
    verbatim under "bim_only" (reported, not applied).
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from typing import Any

import numpy as np
from viser.transforms import SO3

from pipeline.bim_provenance import ArtifactMismatch, backup, compare
from pipeline.common import out_path
from pipeline.export_ifc import AXES, rot_from_ifc, rot_to_ifc

Cfg = dict[str, Any]
VERSION = 1
FILE = "scene_edits.json"
FRAME = "scene: -y up, scan units; p_shown = R(wxyz) @ (p - pivot) + position"
MAX_HISTORY = 200
IDENTITY = (1.0, 0.0, 0.0, 0.0)


# ---------- transforms ----------

def rigid(position: Any, wxyz: Any, pivot: Any) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) with p_shown = R @ p + t."""
    R = SO3(np.asarray(wxyz, np.float64)).as_matrix()
    return R, np.asarray(position, np.float64) - R @ np.asarray(pivot, np.float64)


def apply(points: np.ndarray, position: Any, wxyz: Any, pivot: Any) -> np.ndarray:
    R, t = rigid(position, wxyz, pivot)
    return np.asarray(points, np.float64) @ R.T + t


def to_bim_pose(position: Any, wxyz: Any, pivot: Any, centre_ifc: Any, u2m: float) -> tuple[np.ndarray, np.ndarray]:
    """Scene instance pose -> bim_view pose (IFC frame, rotation about the object's IFC centroid)."""
    R, t = rigid(position, wxyz, pivot)
    Ri, ti = rot_to_ifc(R), u2m * AXES @ t
    return ti + Ri @ np.asarray(centre_ifc, np.float64), SO3.from_matrix(Ri).wxyz


def from_bim_pose(pos_ifc: Any, wxyz_ifc: Any, centre_ifc: Any, pivot: Any, u2m: float) -> tuple[np.ndarray, np.ndarray]:
    """bim_view pose -> scene instance pose (inverse of to_bim_pose)."""
    Ri = SO3(np.asarray(wxyz_ifc, np.float64)).as_matrix()
    ti = np.asarray(pos_ifc, np.float64) - Ri @ np.asarray(centre_ifc, np.float64)
    R, t = rot_from_ifc(Ri), AXES.T @ ti / u2m
    return t + R @ np.asarray(pivot, np.float64), SO3.from_matrix(R).wxyz


# ---------- state ----------

def instance(source: str, pivot: Any, copy_of: bool = False, position: Any = None, wxyz: Any = IDENTITY,
             removed: bool = False) -> dict[str, Any]:
    return {"source": source, "copy": copy_of, "position": [float(x) for x in (pivot if position is None else position)],
            "wxyz": [float(x) for x in wxyz], "removed": bool(removed)}


def originals(pivots: dict[str, Any]) -> dict[str, dict]:
    return {oid: instance(oid, p) for oid, p in pivots.items()}


def layout_hash(instances: dict[str, dict], pivots: dict[str, Any]) -> str:
    """Hash of what the layout does to the scene (untouched originals don't count)."""
    rows = []
    for name, it in sorted(instances.items()):
        p = np.asarray(it["position"], float)
        q = np.asarray(it["wxyz"], float)
        q = q * np.sign(q[0] or 1.0)
        if not it["copy"] and not it["removed"] and np.allclose(p, pivots[it["source"]], atol=1e-4) \
                and np.allclose(q, IDENTITY, atol=1e-5):
            continue
        rows.append([name, it["source"], it["copy"], it["removed"], np.round(p, 4).tolist(), np.round(q, 5).tolist()])
    return hashlib.sha1(json.dumps(rows).encode()).hexdigest()[:12]


def new_state(rev: dict[str, Any], pivots: dict[str, Any]) -> dict[str, Any]:
    return {"version": VERSION, "frame": FRAME, "revision": rev,
            "pivots": {k: [float(x) for x in v] for k, v in pivots.items()},
            "instances": originals(pivots), "applied_layout": None, "history": [], "migrated_from": [],
            "bim_only": {}, "updated": None}


def push_history(state: dict[str, Any], action: str) -> None:
    """Remember the layout before `action` (undo restores it)."""
    state["history"].append({"at": time.strftime("%Y-%m-%d %H:%M:%S"), "action": action,
                             "instances": copy.deepcopy(state["instances"])})
    del state["history"][:-MAX_HISTORY]


def undo(state: dict[str, Any]) -> str | None:
    if not state["history"]:
        return None
    h = state["history"].pop()
    state["instances"] = h["instances"]
    return h["action"]


# ---------- adapters ----------

def from_viewer_edits(legacy: dict[str, Any], pivots: dict[str, Any]) -> tuple[dict[str, dict], list[str]]:
    """viewer_edits.json -> instances (same frame and pivot). Unknown objects are reported, not applied."""
    inst, skipped = originals(pivots), []
    for name, e in legacy.items():
        if e.get("oid") not in pivots:
            skipped.append(name)
            continue
        inst[name] = instance(e["oid"], pivots[e["oid"]], copy_of=name != e["oid"], position=e["position"],
                              wxyz=e["wxyz"], removed=e.get("removed", False))
    return inst, skipped


def from_bim_edits(edits: dict[str, Any], pivots: dict[str, Any], centres_ifc: dict[str, Any],
                   u2m: float) -> tuple[dict[str, dict], dict[str, Any]]:
    """bim_view edits.json -> instances; returns (instances, bim_only = what can't be represented here)."""
    inst = originals(pivots)
    bim_only: dict[str, Any] = {k: edits[k] for k in ("groups", "detached", "pieces") if edits.get(k)}
    for oid, m in edits.get("moved", {}).items():
        if oid in pivots and oid in centres_ifc:
            pos, q = from_bim_pose(m["position"], m["wxyz"], centres_ifc[oid], pivots[oid], u2m)
            inst[oid].update(position=pos.tolist(), wxyz=q.tolist())
        else:
            bim_only.setdefault("moved", {})[oid] = m
    for oid in edits.get("removed", []):
        if oid in inst:
            inst[oid]["removed"] = True
        else:
            bim_only.setdefault("removed", []).append(oid)
    for c in edits.get("copies", []):
        oid = c["of"]
        if oid not in pivots or oid not in centres_ifc:
            bim_only.setdefault("copies", []).append(c)
            continue
        k = 1 + sum(1 for n in inst if n.startswith(f"{oid}_copy"))
        pos, q = from_bim_pose(c["position"], c["wxyz"], centres_ifc[oid], pivots[oid], u2m)
        inst[f"{oid}_copy{k}"] = instance(oid, pivots[oid], True, pos, q, c.get("removed", False))
    return inst, bim_only


def to_bim_edits(instances: dict[str, dict], pivots: dict[str, Any], centres_ifc: dict[str, Any], u2m: float,
                 bim_only: dict[str, Any] | None = None) -> dict[str, Any]:
    """instances -> bim_view edits.json (IFC frame), plus anything kept from an earlier bim_view file."""
    out: dict[str, Any] = {"moved": {}, "removed": [], "copies": [], "groups": {}, "detached": [], "pieces": []}
    for name, it in instances.items():
        oid = it["source"]
        pos, q = to_bim_pose(it["position"], it["wxyz"], pivots[oid], centres_ifc[oid], u2m)
        rec = {"position": [round(float(x), 4) for x in pos], "wxyz": [round(float(x), 5) for x in q]}
        if it["copy"]:
            out["copies"].append({"of": oid, **rec, "removed": it["removed"]})
            continue
        if it["removed"]:
            out["removed"].append(oid)
        if not (np.allclose(it["position"], pivots[oid], atol=1e-4) and np.allclose(it["wxyz"], IDENTITY, atol=1e-5)):
            out["moved"][oid] = rec
    for k, v in (bim_only or {}).items():
        if isinstance(out.get(k), dict):
            out[k].update(v)
        elif isinstance(out.get(k), list):
            out[k] += v
    return out


# ---------- files ----------

def load(cfg: Cfg, rev: dict[str, Any], pivots: dict[str, Any],
         centres_ifc: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[str]]:
    """scene_edits.json for this scene version; else a migration of the older files (they are left as they are).
    Raises ArtifactMismatch when scene_edits.json belongs to another scene version."""
    notes: list[str] = []
    p = out_path(cfg, FILE)
    if p.exists():
        st = json.loads(p.read_text(encoding="utf-8"))
        if st.get("version") != VERSION:
            raise ArtifactMismatch(f"{FILE}: version {st.get('version')} is not supported (expected {VERSION})")
        errs = compare(st.get("revision"), rev, FILE)
        if errs:
            raise ArtifactMismatch("; ".join(errs))
        missing = sorted({it["source"] for it in st["instances"].values()} - set(pivots))
        if missing:
            raise ArtifactMismatch(f"{FILE}: objects {missing} are no longer editable (removable.json changed)")
        for oid, pv in pivots.items():                      # newly editable objects start untouched
            st["instances"].setdefault(oid, instance(oid, pv))
            st["pivots"].setdefault(oid, [float(x) for x in pv])
        return st, notes
    st = new_state(rev, pivots)
    legacy = out_path(cfg, "viewer_edits.json")
    if legacy.exists():
        inst, skipped = from_viewer_edits(json.loads(legacy.read_text(encoding="utf-8")), pivots)
        st["instances"].update(inst)
        st["migrated_from"].append({"file": "viewer_edits.json", "skipped": skipped,
                                    "note": "unversioned legacy file; objects matched by id within this scene"})
        notes.append(f"migrated viewer_edits.json ({len(inst)} instances, skipped {skipped})")
    bim = out_path(cfg, "edits.json")
    if bim.exists() and centres_ifc is not None:
        inst, bim_only = from_bim_edits(json.loads(bim.read_text(encoding="utf-8")), pivots, centres_ifc,
                                        cfg["scene"]["units_to_m"])
        changed = {k: v for k, v in inst.items() if v != st["instances"].get(k)}
        st["instances"].update(changed)
        st["bim_only"] = bim_only
        st["migrated_from"].append({"file": "edits.json", "applied": sorted(changed), "bim_only": sorted(bim_only)})
        notes.append(f"migrated edits.json ({len(changed)} instances; kept, not applied: {sorted(bim_only) or 'none'})")
    return st, notes


def save(cfg: Cfg, state: dict[str, Any]) -> None:
    p = out_path(cfg, FILE)
    if p.exists() and not getattr(save, "_backed_up", False):
        backup(cfg, [FILE], "scene_edits")                  # once per session, before the first overwrite
        save._backed_up = True                              # type: ignore[attr-defined]
    state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(p)
