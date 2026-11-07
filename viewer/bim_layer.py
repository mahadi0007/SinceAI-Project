"""out/scene.ifc as scene-frame meshes for viewer.app (Scan / BIM / Overlay display modes).

Tessellation is viewer.bim_view.tessellate; vertices go from IFC (+z up, metres) to the scene (-y up, units) with
export_ifc.from_ifc. Elements are associated with scene objects through Pset_ScanExtraction.ObjectId, and only when
their SegmentationRevision is the current one; any element from another version rejects the whole model.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from pipeline.bim_provenance import ArtifactMismatch, check_bim
from pipeline.common import out_path
from pipeline.export_ifc import from_ifc

Cfg = dict[str, Any]

CLASS_COLORS = {                                       # same palette as viewer.bim_view
    "IfcPipeSegment": (70, 130, 220),
    "IfcPipeFitting": (240, 170, 40),
    "IfcPump": (220, 60, 60),
    "IfcValve": (60, 190, 90),
    "IfcElectricMotor": (170, 90, 210),
    "IfcSensor": (40, 200, 200),
    "IfcBuildingElementProxy": (150, 150, 150),
}


def ray_hits(verts: np.ndarray, faces: np.ndarray, o: np.ndarray, d: np.ndarray) -> float:
    """Nearest ray-triangle hit distance (Moeller-Trumbore, both sides); inf if none."""
    if not len(faces):
        return np.inf
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    e1, e2 = b - a, c - a
    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = o - a
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = (q @ d) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
    return float(t[hit].min()) if hit.any() else np.inf


def ray_box(lo: np.ndarray, hi: np.ndarray, o: np.ndarray, d: np.ndarray) -> bool:
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - o) / d, (hi - o) / d
    tmin = np.nanmax(np.minimum(t1, t2))
    tmax = np.nanmin(np.maximum(t1, t2))
    return bool(tmax >= max(tmin, 0.0))


class BimLayer:
    """Per scene object: IFC class, colour and a triangle mesh in scene coordinates."""

    def __init__(self, cfg: Cfg, nodes: list[dict], rev: dict[str, Any]) -> None:
        import ifcopenshell
        from viewer.bim_view import tessellate
        t0 = time.time()
        self.manifest = check_bim(cfg, rev)
        f = ifcopenshell.open(str(out_path(cfg, "scene.ifc")))
        prov = next((ps for ps in (self._psets(f.by_type("IfcProject")[0]) or {}).values()
                     if "SegmentationRevision" in ps), {})
        if prov.get("SegmentationRevision") != rev["segmentation"]:
            raise ArtifactMismatch(f"scene.ifc project revision {prov.get('SegmentationRevision')} != current "
                                   f"{rev['segmentation']}")
        u2m = cfg["scene"]["units_to_m"]
        by_id = {n["id"]: n for n in nodes}
        self.mesh: dict[str, dict[str, Any]] = {}
        bad, offs = [], []
        for e in tessellate(f):
            ps = e["psets"].get("Pset_ScanExtraction", {})
            oid = ps.get("ObjectId")
            if ps.get("SegmentationRevision") != rev["segmentation"] or oid not in by_id:
                bad.append(f"{e['el'].GlobalId}:{oid}@{ps.get('SegmentationRevision')}")
                continue
            v = from_ifc(e["verts"], u2m)
            cls = e["el"].is_a()
            self.mesh[oid] = {"verts": v, "faces": e["faces"].astype(np.int64), "cls": cls,
                              "color": CLASS_COLORS.get(cls, CLASS_COLORS["IfcBuildingElementProxy"]),
                              "lo": v.min(0), "hi": v.max(0), "name": e["el"].Name, "guid": e["el"].GlobalId,
                              "model_source": ps.get("ModelSource", "")}
            n = by_id[oid]
            ext = max(float(np.linalg.norm(n["extent"])), 0.05)
            offs.append(float(np.linalg.norm((v.min(0) + v.max(0)) / 2 - np.array(n["center"]))) / ext)
        if bad:
            raise ArtifactMismatch(f"{len(bad)} IFC elements do not belong to this scene version, e.g. {bad[:3]}")
        med = float(np.median(offs)) if offs else np.inf
        if med > 0.25:                                    # wrong axes / units put every element off its object
            raise ArtifactMismatch(f"IFC geometry does not line up with the scan objects (median centre offset "
                                   f"{med:.2f} x object size): coordinate convention mismatch")
        self.centre_offset = med
        print(f"[bim] {len(self.mesh)} IFC elements matched to scene objects in {time.time() - t0:.1f}s "
              f"(median centre offset {med:.3f} x size)")

    @staticmethod
    def _psets(el: Any) -> dict:
        import ifcopenshell.util.element as ue
        return ue.get_psets(el)

    def local(self, oids: list[str], pivot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Merged mesh of these objects relative to pivot (for an instance frame)."""
        ms = [self.mesh[o] for o in oids if o in self.mesh]
        if not ms:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64)
        base = np.cumsum([0] + [len(m["verts"]) for m in ms])[:-1]
        return (np.concatenate([m["verts"] for m in ms]) - pivot).astype(np.float32), \
            np.concatenate([m["faces"] + b for m, b in zip(ms, base)])

    def pick(self, o: np.ndarray, d: np.ndarray, oids: Any) -> tuple[float, str | None]:
        """Nearest of these objects' meshes hit by the ray (scene frame)."""
        best = (np.inf, None)
        for oid in oids:
            m = self.mesh.get(oid)
            if m is None or not ray_box(m["lo"] - 1e-3, m["hi"] + 1e-3, o, d):
                continue
            t = ray_hits(m["verts"], m["faces"], o, d)
            if t < best[0]:
                best = (t, oid)
        return best
