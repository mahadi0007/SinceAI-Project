"""Explicit export of an edited layout as a separate IFC: out/scene.ifc + out/scene_edits.json -> out/scene_edited.ifc

scene.ifc is never modified. Supported: moving / turning (any rotation) an editable object (its element and its
attached parts' elements get a new ObjectPlacement = layout transform x old placement), removing it (elements
deleted), copies (new elements with new GlobalIds, Pset_ValmetAsset removed, Pset_ScanExtraction.ObjectId = copy
name and CopyOf = source). Not supported: bim_view groups / smart-selection pieces (no viewer.app equivalent).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from pipeline.bim_provenance import ArtifactMismatch, backup, check_bim, compare, scene_revision
from pipeline.common import out_path
from pipeline.export_ifc import AXES, rot_to_ifc

Cfg = dict[str, Any]


def layout_matrix_ifc(it: dict, pivot: Any, u2m: float) -> np.ndarray:
    """4x4 IFC-frame (metres) matrix of an instance's scene-frame layout transform."""
    from viewer.scene_edits import rigid
    R, t = rigid(it["position"], it["wxyz"], pivot)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rot_to_ifc(R), u2m * AXES @ t
    return T


def export_edited(cfg: Cfg, state: dict[str, Any], members: dict[str, list[str]],
                  name: str = "scene_edited.ifc") -> dict[str, Any]:
    import ifcopenshell
    import ifcopenshell.api
    import ifcopenshell.util.element as ue
    import ifcopenshell.util.placement as up
    from viewer.scene_edits import layout_hash

    rev = scene_revision(cfg)
    m = check_bim(cfg, rev)
    errs = compare(state.get("revision"), rev, "scene_edits.json")
    if errs:
        raise ArtifactMismatch("; ".join(errs))
    u2m = cfg["scene"]["units_to_m"]
    f = ifcopenshell.open(str(out_path(cfg, "scene.ifc")))
    el_of = {ue.get_psets(el).get("Pset_ScanExtraction", {}).get("ObjectId"): el for el in f.by_type("IfcElement")}
    stats = {"moved": 0, "removed": 0, "copies": 0, "elements": 0}

    def placed(el: Any, T: np.ndarray) -> None:
        M = up.get_local_placement(el.ObjectPlacement)
        ifcopenshell.api.run("geometry.edit_object_placement", f, product=el, matrix=T @ M, is_si=True)

    for iname, it in sorted(state["instances"].items(), key=lambda kv: kv[1]["copy"]):   # originals first
        src = it["source"]
        els = [el_of[x] for x in members.get(src, [src]) if x in el_of]
        T = layout_matrix_ifc(it, state["pivots"][src], u2m)
        identity = np.allclose(T, np.eye(4), atol=1e-6)
        if not it["copy"]:
            if it["removed"]:
                for el in els:
                    ifcopenshell.api.run("root.remove_product", f, product=el)
                stats["removed"] += 1
            elif not identity:
                for el in els:
                    placed(el, T)
                stats["moved"] += 1
            continue
        if it["removed"]:
            continue
        for el in els:
            c = ifcopenshell.api.run("root.copy_class", f, product=el)
            if c.Representation is None and el.Representation is not None:   # copy_class leaves the shape out
                c.Representation = ue.copy_deep(f, el.Representation)
            placed(c, T)
            ps = ue.get_psets(c)
            if "Pset_ValmetAsset" in ps:                                 # a copy never carries an asset identity
                ifcopenshell.api.run("pset.remove_pset", f, product=c, pset=f.by_id(ps["Pset_ValmetAsset"]["id"]))
            sx = ps.get("Pset_ScanExtraction")
            if sx:
                ifcopenshell.api.run("pset.edit_pset", f, pset=f.by_id(sx["id"]),
                                     properties={"ObjectId": iname, "CopyOf": sx["ObjectId"]})
            c.Name = f"{iname} (copy of {el.Name})"
            stats["elements"] += 1
        stats["copies"] += 1
    proj = f.by_type("IfcProject")[0]
    prov = ue.get_psets(proj).get("Pset_ScanProvenance")
    if prov:
        ifcopenshell.api.run("pset.edit_pset", f, pset=f.by_id(prov["id"]), properties={
            "EditedLayout": layout_hash(state["instances"], state["pivots"]),
            "EditedFromSha256": m["files"]["scene.ifc"]})
    backup(cfg, [name], "edited_ifc")
    f.write(str(out_path(cfg, name)))
    return {"file": name, **stats}
