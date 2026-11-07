"""PoC viewer for out/scene.ifc plus the photoreal splat, split into one piece per scan object.

Click an object to select it; drag the gizmo to move / turn it (yaw only), or delete / duplicate it. Splat
piece and BIM element move together. Small objects enclosed by a larger one (and the noise inside it) are parts
of it; "Click adds parts" attaches more, "Ungroup" detaches them. Smart select renders the clicked view, runs SAM 2.1
at the click and selects the Gaussians inside the mask (shift-click adds to the selection). Edits persist in
out/edits.json (+ edits_pieces.npz).

Click an element to see its property sets. Colour by scan colours (from the IFC styles / colour maps), IFC class
or mean deviation from the scan;
the heat map colours every scan point by its distance to the model (out/deviation.npz).
Run: python -m viewer.bim_view  (port viewer.bim_port, default 8090)
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from itertools import product
from typing import Any

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as ue
import numpy as np
import trimesh
import viser
from scipy.spatial import cKDTree
from viser.transforms import SO3

from pipeline.common import load_config, out_path, repo_path
from pipeline.export_ifc import to_ifc
from pipeline.load_splat import covariances, load_splat
from pipeline.propose import sam_model
from pipeline.views import Camera, render
from viewer.nudge import add_nudge_gui, gizmo_scale

Cfg = dict[str, Any]

CLASS_COLORS = {
    "IfcPipeSegment": (70, 130, 220),
    "IfcPipeFitting": (240, 170, 40),
    "IfcPump": (220, 60, 60),
    "IfcValve": (60, 190, 90),
    "IfcElectricMotor": (170, 90, 210),
    "IfcSensor": (40, 200, 200),
    "IfcBuildingElementProxy": (150, 150, 150),
}


DEV_GOOD_MM, DEV_BAD_MM = 10, 60


def deviation_colors(mm: np.ndarray) -> np.ndarray:
    """Green (<= DEV_GOOD_MM) -> yellow -> red (>= DEV_BAD_MM); grey where unknown."""
    t = np.clip((np.nan_to_num(mm, nan=-1) - DEV_GOOD_MM) / (DEV_BAD_MM - DEV_GOOD_MM), 0, 1)
    rgb = np.where(t[:, None] < 0.5, np.c_[60 + 2 * t * 180, 190 + 0 * t, 70 + 0 * t],
                   np.c_[240 + 0 * t, 190 - (2 * t - 1) * 140, 70 + 0 * t])
    rgb[~np.isfinite(mm)] = 150
    return rgb.astype(np.uint8)


def read_faceset(fs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IfcTriangulatedFaceSet read directly, with its per-face IfcIndexedColourMap (the geometry iterator
    drops indexed colours). Elements holding face sets are placed at the origin, so coordinates are world."""
    v = np.array(fs.Coordinates.CoordList, np.float32)
    t = np.array(fs.CoordIndex, np.uint32) - 1
    col = np.full((len(t), 3), 0.7)
    if fs.HasColours:
        cm = fs.HasColours[0]
        col = np.array(cm.Colours.ColourList)[np.array(cm.ColourIndex) - 1]
    return v, t, (col * 255).astype(np.uint8)


def tessellate(f: Any) -> list[dict[str, Any]]:
    out, swept = [], []
    for el in f.by_type("IfcElement"):
        items = [i for r in (el.Representation.Representations if el.Representation else []) for i in r.Items]
        if items and all(i.is_a("IfcTriangulatedFaceSet") for i in items):
            parts = [read_faceset(i) for i in items]
            offs = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
            out.append({"el": el, "psets": ue.get_psets(el), "verts": np.concatenate([p[0] for p in parts]),
                        "faces": np.concatenate([p[1] + o for p, o in zip(parts, offs)]).astype(np.uint32),
                        "face_rgb": np.concatenate([p[2] for p in parts])})
        elif items:
            swept.append(el)
    if not swept:
        return out
    s = ifcopenshell.geom.settings()
    s.set("use-world-coords", True)
    it = ifcopenshell.geom.iterator(s, f, os.cpu_count() or 1, include=swept)
    it.initialize()
    while True:
        sh = it.get()
        el = f.by_id(sh.id)
        g = sh.geometry
        mats = np.array([[m.diffuse.r(), m.diffuse.g(), m.diffuse.b()] for m in g.materials] or [[0.7] * 3])
        ids = np.maximum(np.array(g.material_ids, int), 0)
        out.append({"el": el, "psets": ue.get_psets(el),
                    "verts": np.array(g.verts, np.float32).reshape(-1, 3),
                    "faces": np.array(g.faces, np.uint32).reshape(-1, 3),
                    "face_rgb": (mats[ids] * 255).astype(np.uint8)})
        if not it.next():
            return out


def props_md(e: dict[str, Any]) -> str:
    el = e["el"]
    lines = [f"**{el.Name}**  `{el.is_a()}`  ", f"GlobalId `{el.GlobalId}`"]
    for pname, props in e["psets"].items():
        lines.append(f"\n**{pname}**  ")
        lines += [f"{k}: {v}  " for k, v in props.items() if k != "id"]
    return "\n".join(lines)


class BimView:
    """Splat + BIM viewer with object editing. Every scan object is its own splat piece (Gaussians within
    label.member_dist of its segmented points) and BIM element; move / delete / duplicate act on both.
    Edits are kept in out/edits.json (the IFC and pipeline outputs are not changed)."""

    def __init__(self, cfg: Cfg, apply_edits: bool = True) -> None:
        self.cfg, u2m = cfg, cfg["scene"]["units_to_m"]
        t = time.time()
        self.f = ifcopenshell.open(str(out_path(cfg, "scene.ifc")))
        self.elems = tessellate(self.f)
        print(f"[ifc] {len(self.elems)} elements tessellated in {time.time() - t:.1f}s")
        with open(out_path(cfg, "scene_graph.json"), encoding="utf-8") as fh:
            self.objs = json.load(fh)["objects"]
        index = {o["id"]: k for k, o in enumerate(self.objs)}
        self.elem_of = {index[e["psets"]["Pset_ScanExtraction"]["ObjectId"]]: e for e in self.elems}

        seg = np.load(out_path(cfg, "segment_points.npz"))
        self.pts = to_ifc(seg["xyz"], u2m).astype(np.float32)
        self.seg_labels = seg["labels"]
        self.pt_lab = seg["labels"].astype(np.int64)
        self.pt_lab[np.isin(self.pt_lab, [k for k, o in enumerate(self.objs) if o["kind"] != "object"])] = -1   # walls etc.
        self.splat = self._split_splat(seg, u2m)
        self.centre0 = {k: self.pts[self.pt_lab == k].mean(0) for k in self.elem_of if (self.pt_lab == k).any()}
        self.centre = dict(self.centre0)

        self.src: dict[int, int] = {k: k for k in self.centre}             # key -> scene object it shows
        self.leader: dict[int, int] = {}                                   # part object -> object it belongs to
        self.manual: dict[int, set[int]] = {}                              # parts attached by hand
        self.detached: set[int] = set()                                    # never auto-attached (ungrouped)
        self.pieces: dict[int, dict[str, Any]] = {}                        # smart selections: rows, objs, name, centre
        self.taken = np.zeros(len(self.splat["xyz"]), bool)                # Gaussians held by a smart selection
        self.last_sam: dict[str, Any] | None = None
        self.busy = threading.Lock()                                       # clicks arriving mid-edit are dropped
        rpath = out_path(cfg, "removable.json")
        self.allowed: set[int] | None = None                               # selectable objects (removable.json)
        if apply_edits and rpath.exists():
            ids = set(json.loads(rpath.read_text(encoding="utf-8"))["removable"])
            self.allowed = {k for k, o in enumerate(self.objs) if o["id"] in ids}
        self.pose: dict[int, tuple[np.ndarray, np.ndarray]] = {}           # key -> (gizmo position, wxyz)
        self.removed: set[int] = set()
        self.selected: int | None = None
        self.mesh_h: dict[int, Any] = {}
        self.splat_h: dict[int, Any] = {}
        self.gizmo: Any = None
        self.server = viser.ViserServer(port=cfg["viewer"].get("bim_port", 8090))
        self.server.scene.set_up_direction("+z")
        self._build_gui(seg)
        if apply_edits:
            self._load_edits()
        else:
            self._init_groups()
        self._build_splats()
        self._build_meshes()
        self.server.scene.on_click()(self._on_click)
        threading.Thread(target=sam_model, args=(cfg["propose"]["sam_model"],), daemon=True).start()

    # --- data ---
    def _split_splat(self, seg: Any, u2m: float) -> dict[str, Any]:
        t = time.time()
        d = load_splat(str(repo_path(self.cfg["paths"]["ply"])))
        lab, near_noise, nearest = self._own(seg, d)
        self.g_nearest = nearest
        M = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])                  # scene -> IFC axes (see to_ifc)
        cov = covariances(d["scales"], d["quats_wxyz"]).astype(np.float64)
        order = np.argsort(lab, kind="stable")
        keys, starts = np.unique(lab[order], return_index=True)
        print(f"[splat] {len(lab)} Gaussians, {int((lab >= 0).sum())} on objects, split in {time.time() - t:.1f}s")
        return {"xyz": to_ifc(d["xyz"], u2m).astype(np.float32), "scale": (u2m * d["scales"]).astype(np.float32),
                "lab": lab, "row_ids": d["row_ids"],
                "cov": (u2m ** 2 * M @ cov @ M.T).astype(np.float32),
                "rgb": d["rgb"].astype(np.float32), "op": d["opacity"].astype(np.float32)[:, None],
                "rows": {int(k): order[a:b] for k, a, b in zip(keys, starts, list(starts[1:]) + [len(lab)])},
                "noise": np.flatnonzero((lab == -1) & near_noise)}

    def _own(self, seg: Any, d: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gaussian -> object (or -1): distance-weighted vote of the nearest voxels within member_dist + the
        Gaussian's own size. Noise voxels don't vote, so object fringe the segmentation dropped follows its object;
        a Gaussian whose nearest voter is a wall stays in the background. Also returns 'nearest voxel is noise' and the nearest voxel."""
        oc = self.cfg["viewer"]["splat_owner"]
        vote = np.where(seg["labels"] == -1, -3, self.pt_lab)          # -3 noise (no vote), -1 walls etc.
        reach = self.cfg["label"]["member_dist"] + np.minimum(d["scales"].max(1), oc["max_size"])
        dist, j = cKDTree(seg["xyz"]).query(d["xyz"], k=oc["k"], distance_upper_bound=float(reach.max()), workers=-1)
        lab = vote[np.minimum(j, len(vote) - 1)]
        w = np.where((dist <= reach[:, None]) & (lab != -3), 1.0 / (dist + oc["eps"]), 0.0)
        score = np.einsum("nk,nkc->nc", w, lab[:, :, None] == lab[:, None, :])
        best = score.argmax(1)
        rows = np.arange(len(lab))
        first = lab[rows, (w > 0).argmax(1)]                               # nearest voter: a wall keeps it
        return (np.where((score[rows, best] > 0) & (first != -1), lab[rows, best], -1),
                (lab[:, 0] == -3) & (dist[:, 0] <= reach), np.minimum(j[:, 0], len(vote) - 1))

    # --- groups ---
    off = np.array(list(product((-1, 0, 1), repeat=3)))                   # one-cell dilation

    def _vox(self, p: np.ndarray) -> np.ndarray:
        return np.floor(p / self.cfg["viewer"]["absorb"]["voxel"]).astype(np.int64)

    @staticmethod
    def _code(v: np.ndarray) -> np.ndarray:
        v = v + 2 ** 20
        return (v[:, 0] << 42) | (v[:, 1] << 21) | v[:, 2]

    def _can_lead(self, k: int, n_pts: int) -> bool:
        ac = self.cfg["viewer"]["absorb"]
        return n_pts >= ac["min_leader_points"] and self.objs[k].get("label") not in ac["leader_skip"]

    def _auto_groups(self) -> dict[int, int]:
        """Small object -> larger non-pipe object whose (one voxel dilated) occupancy holds most of its points."""
        ac = self.cfg["viewer"]["absorb"]
        n = np.bincount(self.pt_lab[self.pt_lab >= 0], minlength=len(self.objs))
        cover: dict[int, list[int]] = {}
        objects = [k for k in self.src if k not in self.pieces]
        for k in objects:
            if self._can_lead(k, n[k]):
                v = np.unique(self._vox(self.pts[self.pt_lab == k]), axis=0)
                for c in np.unique(self._code((v[:, None] + self.off).reshape(-1, 3))).tolist():
                    cover.setdefault(c, []).append(k)
        out = {}
        for k in objects:
            if k in self.detached or not 0 < n[k] <= ac["max_part_points"]:
                continue
            codes = self._code(self._vox(self.pts[self.pt_lab == k])).tolist()
            hits = Counter(lead for c in codes for lead in cover.get(c, ()) if n[lead] >= 3 * n[k])
            if hits:
                lead, h = hits.most_common(1)[0]
                if h >= ac["inside_frac"] * len(codes):
                    out[k] = lead
        return out

    def _regroup(self) -> None:
        """Groups -> point labels, centres and the noise Gaussians within a cell of each group."""
        self.parts: dict[int, list[int]] = {}
        for m, lead in self.leader.items():
            self.parts.setdefault(lead, []).append(m)
        lut = np.arange(len(self.objs))
        lut[list(self.leader)] = list(self.leader.values())
        self.grp_lab = np.where(self.pt_lab >= 0, lut[np.maximum(self.pt_lab, 0)], -1)
        n = np.bincount(self.grp_lab[self.grp_lab >= 0], minlength=len(self.objs))
        self.centre = dict(self.centre0)
        self.centre.update({P: pc["centre"] for P, pc in self.pieces.items()})
        owner: dict[int, int] = {}
        for k in self.src:
            if k == self.src[k] and k not in self.pieces and (k in self.parts or self._can_lead(k, n[k])):
                g = self.grp_lab == k
                if k in self.parts:
                    self.centre[k] = self.pts[g].mean(0)
                v = np.unique(self._vox(self.pts[g]), axis=0)
                for c in np.unique(self._code((v[:, None] + self.off).reshape(-1, 3))).tolist():
                    owner.setdefault(c, k)
        sp = self.splat
        noise = sp["noise"]
        take = np.array([owner.get(c, -1) for c in self._code(self._vox(sp["xyz"][noise])).tolist()], np.int64)
        order = np.argsort(take, kind="stable")
        keys, starts = np.unique(take[order], return_index=True)
        self.noise_of = {int(k): noise[order[a:b]] for k, a, b in zip(keys, starts, list(starts[1:]) + [len(take)])
                         if k >= 0}
        self.bg_rows = np.setdiff1d(sp["rows"].get(-1, np.zeros(0, np.int64)), noise[take >= 0])
        self.g_owner = np.where(sp["lab"] >= 0, lut[np.maximum(sp["lab"], 0)], -1)    # Gaussian -> object key
        for k, r in self.noise_of.items():
            self.g_owner[r] = k

    def _init_groups(self) -> None:
        self.leader = self._auto_groups()
        for lead in self.manual:
            self.leader.pop(lead, None)
        for lead, ms in self.manual.items():
            for m in ms:
                self.leader[m] = lead
        for m in self.leader:
            self.src.pop(m, None)
        self._regroup()

    # --- scene ---
    def _rows(self, key: int) -> np.ndarray:
        if key < 0:
            return self.bg_rows[~self.taken[self.bg_rows]]
        k = self.src[key]
        if k in self.pieces:
            return self.pieces[k]["rows"]
        r = [self.splat["rows"].get(x) for x in [k, *self.parts.get(k, [])]] + [self.noise_of.get(k)]
        r = np.concatenate([x for x in r if x is not None] or [np.zeros(0, np.int64)])
        return r[~self.taken[r]]

    def _add_splat(self, key: int) -> None:
        if key in self.splat_h:
            self.splat_h.pop(key).remove()
        rows = self._rows(key)
        if not len(rows):
            return
        sp = self.splat
        self.splat_h[key] = self.server.scene.add_gaussian_splats(
            f"/splat/{key}", centers=sp["xyz"][rows], covariances=sp["cov"][rows], rgbs=sp["rgb"][rows],
            opacities=sp["op"][rows], visible=self.show_splat.value and key not in self.removed)

    def _build_splats(self) -> None:
        self._add_splat(-1)                                                # background
        for key in self.src:
            self._add_splat(key)
            self._apply_pose(key)

    def _build_meshes(self) -> None:
        for h in self.mesh_h.values():
            h.remove()
        self.mesh_h.clear()
        if not self.show_bim.value:                                        # sent to the browser only when shown
            return
        for key in self.src:
            self._add_mesh(key)

    def _add_mesh(self, key: int) -> None:
        if key in self.mesh_h:
            self.mesh_h.pop(key).remove()
        k = self.src[key]
        ids = self.pieces[k]["objs"] if k in self.pieces else [k, *self.parts.get(k, [])]
        es = [self.elem_of[x] for x in ids if x in self.elem_of]
        if not self.show_bim.value or not es or (not self.show_proxy.value
                                                  and es[0]["el"].is_a() == "IfcBuildingElementProxy"):
            return
        e = es[0]
        base = np.cumsum([0] + [len(x["verts"]) for x in es])[:-1]
        verts = np.concatenate([x["verts"] for x in es])
        faces = np.concatenate([x["faces"] + b for x, b in zip(es, base)])
        if self.mode.value == "Scan colours":
            v = verts[faces].reshape(-1, 3)                                # unshared, so faces keep their colour
            face_rgb = np.concatenate([x["face_rgb"] for x in es])
            rgba = np.c_[np.repeat(face_rgb, 3, 0), np.full(len(v), int(self.opacity.value * 255))]
            m = trimesh.Trimesh(v, np.arange(len(v)).reshape(-1, 3), vertex_colors=rgba.astype(np.uint8),
                                process=False)
            h = self.server.scene.add_mesh_trimesh(f"/ifc/{key}", m)
        else:
            h = self.server.scene.add_mesh_simple(f"/ifc/{key}", verts, faces, color=self._color_of(e),
                                                  opacity=self.opacity.value if self.opacity.value < 1 else None)
        h.visible = key not in self.removed
        self.mesh_h[key] = h
        self._apply_pose(key)

    def _color_of(self, e: dict[str, Any]) -> tuple[int, int, int]:
        if self.mode.value == "IFC class":
            return CLASS_COLORS.get(e["el"].is_a(), CLASS_COLORS["IfcBuildingElementProxy"])
        mm = e["psets"].get("Pset_ScanExtraction", {}).get("DeviationMeanMm")
        return tuple(int(v) for v in deviation_colors(np.array([np.nan if mm is None else mm]))[0])

    def _apply_pose(self, key: int) -> None:
        """Rotate about / move the object's centre: node = gizmo pose composed with -centre."""
        c = self.centre[self.src[key]]
        pos, wxyz = self.pose.get(key, (c, np.array([1.0, 0, 0, 0])))
        node_pos = np.asarray(pos) - SO3(np.asarray(wxyz)).as_matrix() @ c
        for h in (self.splat_h.get(key), self.mesh_h.get(key)):
            if h is not None:
                h.position, h.wxyz = tuple(node_pos), tuple(wxyz)

    # --- picking / selection ---
    def _key_points(self, key: int) -> np.ndarray:
        k = self.src[key]
        c = self.centre[k]
        pos, wxyz = self.pose.get(key, (c, np.array([1.0, 0, 0, 0])))
        if k in self.pieces:
            r = self.pieces[k]["rows"]
            p = self.splat["xyz"][r[::max(1, len(r) // 5000)]]
        else:
            p = self.pts[self.grp_lab == k]
        return (p - c) @ SO3(np.asarray(wxyz)).as_matrix().T + pos

    def _on_click(self, ev: Any) -> None:
        if not self.busy.acquire(blocking=False):
            return
        try:
            self._click(ev)
        finally:
            self.busy.release()

    def _click(self, ev: Any) -> None:
        o, d = np.array(ev.ray_origin), np.array(ev.ray_direction)
        d /= np.linalg.norm(d) + 1e-12
        best, best_t = None, np.inf
        for key in self.src:
            if key in self.removed or (not self.show_bim.value and not self.show_splat.value):
                continue
            if self.allowed is not None and self.src[key] not in self.allowed and self.src[key] not in self.pieces:
                continue
            c = self.pose.get(key, (self.centre[self.src[key]],))[0]
            if np.linalg.norm(np.cross(c - o, d)) > 3.0:                      # cheap ray-to-centre cull
                continue
            v = self._key_points(key) - o
            t = v @ d
            hit = (t > 0) & (np.linalg.norm(v - np.outer(t, d), axis=1) < 0.03 + 0.004 * np.maximum(t, 0))
            if hit.any() and t[hit].min() < best_t:
                best, best_t = key, float(t[hit].min())
        add = ev.modifier is not None and "shift" in str(ev.modifier).lower()
        if self.allowed is None and self.smart.value and self.show_splat.value and not self.add_parts.value and (add or best is None or best not in self.pieces):
            self.smart_select(ev, add)
            return
        if best is None:
            return
        sel = self.selected
        if self.add_parts.value and sel is not None and best != sel and sel == self.src[sel] and best == self.src[best]:
            self.attach(sel, best)
        else:
            self.select(best)

    def select(self, key: int | None) -> None:
        self.selected = key
        if self.gizmo is not None:
            self.gizmo.remove()
            self.gizmo = None
        if key is None:
            self.md.content = "Click an object to select it."
            return
        k = self.src[key]
        tag = f" (copy of {self._name(k)})" if key != k else ""
        if k in self.pieces:
            pc = self.pieces[k]
            whole = ", ".join(self.objs[x]["id"] for x in pc["objs"][:12]) + (" ..." if len(pc["objs"]) > 12 else "")
            self.md.content = (f"**{pc['name']}**{tag}  smart selection  \n{len(pc['rows'])} Gaussians, "
                               f"{len(pc['objs'])} whole objects: {whole or 'none'}  \n"
                               "Ungroup = undo the selection")
        else:
            parts = self.parts.get(k, [])
            if parts or self.noise_of.get(k) is not None:
                tag += f" + {len(parts)} parts, {len(self.noise_of.get(k, []))} loose Gaussians"
            self.md.content = props_md(self.elem_of[k]).replace("\n", f"{tag}\n", 1)
        c = self.centre[self.src[key]]
        pos, wxyz = self.pose.get(key, (c, np.array([1.0, 0, 0, 0])))
        extent = float(np.ptp(self._key_points(key), 0).max())
        self.gizmo = self.server.scene.add_transform_controls(
            "/gizmo", scale=gizmo_scale(extent=extent, vc=self.cfg["viewer"]),
            line_width=self.cfg["viewer"].get("gizmo_line_width", 5.0), position=tuple(pos), wxyz=tuple(wxyz),
            depth_test=False, active_axes=(True, True, False),             # drag on the floor; Lift buttons go up / down
            rotation_limits=((0, 0), (0, 0), (-1000, 1000)))               # yaw only

        @self.gizmo.on_update
        def _(_ev: Any) -> None:
            self.pose[key] = (np.array(self.gizmo.position), np.array(self.gizmo.wxyz))
            self._apply_pose(key)
            self._save_edits()

    # --- edits ---
    def nudge(self, dp: np.ndarray, yaw: float) -> None:
        """Step the selected object (Move / Lift / Turn buttons)."""
        key = self.selected
        if key is None:
            return
        pos, wxyz = self.pose.get(key, (self.centre[self.src[key]], np.array([1.0, 0, 0, 0])))
        self.pose[key] = (np.asarray(pos) + dp, (SO3.exp(np.array([0, 0, yaw])) @ SO3(np.asarray(wxyz))).wxyz)
        if self.gizmo is not None:
            self.gizmo.position, self.gizmo.wxyz = tuple(self.pose[key][0]), tuple(self.pose[key][1])
        self._apply_pose(key)
        self._save_edits()

    def delete(self) -> None:
        if self.selected is None:
            return
        key = self.selected
        self.removed.add(key)
        for h in (self.splat_h.get(key), self.mesh_h.get(key)):
            if h is not None:
                h.visible = False
        self.select(None)
        self._save_edits()

    def duplicate(self) -> None:
        if self.selected is None:
            return
        key = self.selected
        new = max(max(self.src) + 1, len(self.objs))
        self.src[new] = self.src[key]
        c = self.centre[self.src[key]]
        pos, wxyz = self.pose.get(key, (c, np.array([1.0, 0, 0, 0])))
        self.pose[new] = (np.asarray(pos) + np.array([0.5, 0.5, 0]), np.asarray(wxyz))
        self._add_splat(new)
        self._build_meshes()
        self._apply_pose(new)
        self.select(new)
        self._save_edits()

    def reset(self) -> None:
        if self.selected is None:
            return
        key = self.selected
        if key != self.src[key]:                                             # a copy: reset = remove it
            self.delete()
            return
        self.pose.pop(key, None)
        self._apply_pose(key)
        self.select(key)
        self._save_edits()

    def _refresh(self, lead: int) -> None:
        self._add_splat(-1)
        for key in [k for k in self.src if self.src[k] == lead]:
            self._add_splat(key)
            self._add_mesh(key)
            self._apply_pose(key)

    def attach(self, lead: int, m: int) -> None:
        """Make object m (with its own parts) a part of lead."""
        if lead in self.pieces or m in self.pieces:
            self._extend(lead, self._rows(m), [m] if m in self.pieces else [m, *self.parts.get(m, [])])
            return
        for x in [m, *self.parts.get(m, [])]:
            for s in self.manual.values():
                s.discard(x)
            self.manual.setdefault(lead, set()).add(x)
            self.leader[x] = lead
            self.detached.discard(x)
        self.manual.pop(m, None)
        for h in (self.splat_h.pop(m, None), self.mesh_h.pop(m, None)):
            if h is not None:
                h.remove()
        del self.src[m]
        self.pose.pop(m, None)
        self.removed.discard(m)
        self._regroup()
        self._refresh(lead)
        self.select(lead)
        self._save_edits()

    def ungroup(self) -> None:
        lead = self.selected
        if lead is not None and lead in self.pieces:
            self._dissolve(lead)
            self.select(None)
            self._save_edits()
            return
        if lead is None or lead != self.src[lead] or not self.parts.get(lead):
            return
        for m in self.parts[lead]:
            del self.leader[m]
            self.detached.add(m)
            self.src[m] = m
        self.manual.pop(lead, None)
        members = self.parts[lead]
        self._regroup()
        self._refresh(lead)
        for m in members:
            self._add_splat(m)
            self._add_mesh(m)
            self._apply_pose(m)
        self.select(lead)
        self._save_edits()

    def restore_all(self) -> None:
        for h in [*self.splat_h.values(), *self.mesh_h.values()]:
            h.remove()
        self.splat_h.clear()
        self.mesh_h.clear()
        self.src = {k: k for k in self.centre0}
        self.pose.clear()
        self.removed.clear()
        self.manual.clear()
        self.detached.clear()
        self.pieces.clear()
        self.taken[:] = False
        self.last_sam = None
        self.select(None)
        self._init_groups()
        self._build_splats()
        self._build_meshes()
        self._save_edits()

    def _name(self, k: int) -> str:
        return self.pieces[k]["name"] if k in self.pieces else self.objs[k]["id"]

    def _save_edits(self) -> None:
        oid = lambda k: self._name(self.src[k])                          # noqa: E731
        data = {"moved": {oid(k): {"position": [round(float(x), 4) for x in p], "wxyz": [round(float(x), 5) for x in q]}
                          for k, (p, q) in self.pose.items() if k == self.src[k]},
                "removed": sorted(oid(k) for k in self.removed if k == self.src[k]),
                "copies": [{"of": oid(k), "position": [round(float(x), 4) for x in self.pose[k][0]],
                            "wxyz": [round(float(x), 5) for x in self.pose[k][1]], "removed": k in self.removed}
                           for k in self.src if k != self.src[k]],
                "groups": {self.objs[lead]["id"]: sorted(self.objs[m]["id"] for m in ms)
                           for lead, ms in self.manual.items() if ms},
                "detached": sorted(self.objs[m]["id"] for m in self.detached),
                "pieces": [{"name": pc["name"], "objs": [self.objs[x]["id"] for x in pc["objs"]]}
                           for pc in self.pieces.values()]}
        np.savez_compressed(out_path(self.cfg, "edits_pieces.npz"),
                            **{pc["name"]: self.splat["row_ids"][pc["rows"]] for pc in self.pieces.values()})
        with open(out_path(self.cfg, "edits.json"), "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)
        n_copies = sum(not c["removed"] for c in data["copies"])
        self.md_edits.content = (f"Edits: {len(data['moved'])} moved, {len(data['removed'])} removed, "
                                 f"{n_copies} copies, {len(data['groups'])} grouped, "
                                 f"{len(data['pieces'])} smart selections (out/edits.json)")

    def _load_edits(self) -> None:
        path = out_path(self.cfg, "edits.json")
        data = {}
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        index = {o["id"]: k for k, o in enumerate(self.objs)}
        self.detached = {index[o] for o in data.get("detached", []) if index.get(o) in self.src}
        self.manual = {index[lead]: {index[m] for m in ms if index.get(m) in self.src}
                       for lead, ms in data.get("groups", {}).items() if index.get(lead) in self.src}
        self._init_groups()
        npz = out_path(self.cfg, "edits_pieces.npz")
        saved = dict(np.load(npz)) if npz.exists() else {}
        for pc in data.get("pieces", []):
            if pc["name"] in saved:
                rows = np.searchsorted(self.splat["row_ids"], saved[pc["name"]])
                objs = [index[o] for o in pc["objs"] if o in index]
                self._make_piece(rows, objs, pc["name"])
        index.update({pc["name"]: P for P, pc in self.pieces.items()})
        for oid, m in data.get("moved", {}).items():
            if index.get(oid) in self.src:
                self.pose[index[oid]] = (np.array(m["position"]), np.array(m["wxyz"]))
        self.removed |= {index[o] for o in data.get("removed", []) if index.get(o) in self.src}
        for c in data.get("copies", []):
            if index.get(c["of"]) in self.src:
                new = max(max(self.src) + 1, len(self.objs))
                self.src[new] = index[c["of"]]
                self.pose[new] = (np.array(c["position"]), np.array(c["wxyz"]))
                if c.get("removed"):
                    self.removed.add(new)

    # --- smart selections (SAM) ---
    def _make_piece(self, rows: np.ndarray, objs: list[int], name: str | None = None) -> int:
        """New selection holding these Gaussians and whole objects; the objects leave the scene as their own keys."""
        P = max(max(self.src) + 1, len(self.objs))
        rows = np.unique(rows)
        if name is None:
            used = {pc["name"] for pc in self.pieces.values()}
            name = next(f"SEL-{i}" for i in range(1, 10 ** 6) if f"SEL-{i}" not in used)
        self.pieces[P] = {"rows": rows, "objs": sorted(set(objs)), "name": name,
                          "centre": self.splat["xyz"][rows].mean(0)}
        self.taken[rows] = True
        for k in objs:
            for key in [x for x in self.src if self.src[x] == k]:
                for h in (self.splat_h.pop(key, None), self.mesh_h.pop(key, None)):
                    if h is not None:
                        h.remove()
                del self.src[key]
                self.pose.pop(key, None)
                self.removed.discard(key)
        self.src[P] = P
        self.centre[P] = self.pieces[P]["centre"]
        return P

    def _refresh_rows(self, rows: np.ndarray) -> None:
        """Rebuild the splat pieces that own (or owned) these Gaussians."""
        self._add_splat(-1)
        owners = set(np.unique(self.g_owner[rows]).tolist())
        for key in [k for k in self.src if self.src[k] in owners and self.src[k] not in self.pieces]:
            self._add_splat(key)
            self._apply_pose(key)

    def _show_piece(self, P: int) -> None:
        self._add_splat(P)
        self._add_mesh(P)
        self._apply_pose(P)

    def _dissolve(self, P: int) -> tuple[np.ndarray, list[int]]:
        pc = self.pieces.pop(P)
        for key in [x for x in self.src if self.src[x] == P]:
            for h in (self.splat_h.pop(key, None), self.mesh_h.pop(key, None)):
                if h is not None:
                    h.remove()
            del self.src[key]
            self.pose.pop(key, None)
            self.removed.discard(key)
        self.taken[pc["rows"]] = False
        for k in pc["objs"]:
            if k not in self.leader and k in self.centre0:
                self.src[k] = k
                self._add_splat(k)
                self._add_mesh(k)
                self._apply_pose(k)
        self._refresh_rows(pc["rows"])
        return pc["rows"], pc["objs"]

    def _extend(self, key: int, rows: np.ndarray, objs: list[int]) -> None:
        """Add Gaussians / objects to the selected object or selection (an object becomes a selection)."""
        k = self.src[key]
        if k in self.pieces:
            pc = self.pieces.pop(k)
            self.taken[pc["rows"]] = False
            for h in (self.splat_h.pop(k, None), self.mesh_h.pop(k, None)):
                if h is not None:
                    h.remove()
            del self.src[k]
            base_rows, base_objs, name = pc["rows"], pc["objs"], pc["name"]
        else:
            base_rows, base_objs, name = self._rows(k), [k, *self.parts.get(k, [])], None
        objs = [o for o in objs if o in self.centre0 or o in self.pieces]
        for o in [o for o in objs if o in self.pieces]:                   # merge other selections too
            pc = self.pieces.pop(o)
            self.taken[pc["rows"]] = False
            for h in (self.splat_h.pop(o, None), self.mesh_h.pop(o, None)):
                if h is not None:
                    h.remove()
            del self.src[o]
            objs = [*objs, *pc["objs"]]
        P = self._make_piece(np.concatenate([base_rows, rows]), [*base_objs, *[o for o in objs if o < len(self.objs)]],
                             name)
        self._refresh_rows(self.pieces[P]["rows"])
        self._show_piece(P)
        self.select(P)
        self._save_edits()

    def _shown(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gaussians as currently shown (originals at their pose + background): splat rows, positions, owner key."""
        rows, xyz, key = [self._rows(-1)], [self.splat["xyz"][self._rows(-1)]], [np.full(len(self._rows(-1)), -1)]
        for k in self.src:
            if k in self.removed or k != self.src[k]:
                continue
            r = self._rows(k)
            c = self.centre[k]
            pos, wxyz = self.pose.get(k, (c, np.array([1.0, 0, 0, 0])))
            rows.append(r)
            xyz.append((self.splat["xyz"][r] - c) @ SO3(np.asarray(wxyz)).as_matrix().T + pos)
            key.append(np.full(len(r), k))
        return np.concatenate(rows), np.concatenate(xyz).astype(np.float32), np.concatenate(key)

    def smart_select(self, ev: Any, add: bool) -> None:
        sc = self.cfg["viewer"]["sam"]
        self.md.content = "Segmenting ..."
        t = time.time()
        cam_h = ev.client.camera
        W = sc["width"]
        H = int(round(W / cam_h.aspect))
        f = 0.5 * H / np.tan(cam_h.fov / 2)
        cam = Camera(np.asarray(cam_h.position, float), SO3(np.asarray(cam_h.wxyz)).as_matrix().T, f, f,
                     (W - 1) / 2, (H - 1) / 2, W, H)
        u, v = ev.screen_pos[0] * (W - 1), ev.screen_pos[1] * (H - 1)
        ray = cam.R.T @ np.array([(u - cam.cx) / f, (v - cam.cy) / f, 1.0])
        if np.degrees(np.arccos(np.clip(ray @ np.asarray(ev.ray_direction) / np.linalg.norm(ray), -1, 1))) > 3:
            print("[sam] camera does not match the click ray; check the camera convention", flush=True)
        rows, xyz, key = self._shown()
        sp = self.splat
        view = render(cam, xyz, sp["rgb"][rows], sp["scale"][rows], sp["op"][rows, 0])   # pixel -> Gaussian
        base = None
        if add and self.selected is not None and self.selected == self.src[self.selected]:
            base = self.selected
        self.select(None)                                                  # no gizmo in the captured frame
        try:                                                               # SAM sees what the browser shows
            img = np.asarray(ev.client.get_render(height=H, width=W, transport_format="png",
                                                  timeout=sc["capture_timeout"]))[..., :3]
        except Exception as e:  # noqa: BLE001
            print(f"[sam] no browser frame ({e}); using the approximate render", flush=True)
            img = view.rgb
        import torch
        from PIL import Image
        proc, model = sam_model(self.cfg["propose"]["sam_model"])
        inp = proc(images=Image.fromarray(img), input_points=[[[[u, v]]]], input_labels=[[[1]]],
                   return_tensors="pt")
        with torch.no_grad():
            out = model(**inp, multimask_output=True)
        m = proc.post_process_masks(out.pred_masks, inp["original_sizes"])[0][0].numpy().astype(bool)
        score = out.iou_scores[0, 0].numpy()
        ui, vi = int(np.clip(round(u), 0, W - 1)), int(np.clip(round(v), 0, H - 1))
        g = view.gid[vi, ui]
        foot = (view.gid >= 0) & (key[np.maximum(view.gid, 0)] == key[g]) if g >= 0 and key[g] >= 0 else None
        cands = []                                                         # (area, score, mask, covers click)
        for x, q in zip(m, score):
            a = int(x.sum())
            if not x[vi, ui] or a < sc["min_mask_px"] or a > sc["max_mask_frac"] * W * H:
                continue
            if foot is not None and (x & ~foot).sum() > sc["max_spill"] * max(int(foot.sum()), 1):
                continue                                                   # far beyond the clicked segment
            covers = foot is not None and (x & foot).sum() >= sc["member_frac"] * foot.sum()
            cands.append((a, float(q), x, covers))
        if not cands:
            self.md.content = "Nothing segmented there; try another spot or angle."
            return
        cands.sort(key=lambda c: c[0])
        masks = [c[2] for c in cands]
        good = [i for i, c in enumerate(cands) if c[3] and c[1] >= sc["min_score"]]
        pick = good[0] if good else int(np.argmax([c[1] for c in cands]))   # smallest that holds the clicked segment
        self.last_sam = {"view": view, "rows": rows, "xyz": xyz, "key": key, "cam": cam, "masks": masks,
                         "i": pick, "base": base, "P": None}
        self._apply_mask()
        print(f"[sam] {time.time() - t:.1f}s, masks (px, score) {[(int(x.sum()), round(float(q), 2)) for x, q in zip(m, score)]}, "
              f"kept {[c[0] for c in cands]}, chose {cands[self.last_sam['i']][0]}", flush=True)
        Image.fromarray(np.where(masks[self.last_sam["i"]][..., None], (0.5 * img + [127, 0, 0]), img)
                        .astype(np.uint8)).save(out_path(self.cfg, "sam_last.png"))

    def resize_selection(self, step: int) -> None:
        ls = self.last_sam
        if ls is None or ls["P"] is None or self.selected != ls["P"] or ls["P"] not in self.pieces:
            return
        i = int(np.clip(ls["i"] + step, 0, len(ls["masks"]) - 1))
        if i == ls["i"]:
            return
        self._dissolve(ls["P"])
        if ls["base_piece"] is not None:                                   # restore the selection we extended
            P = self._make_piece(*ls["base_piece"])
            self._refresh_rows(self.pieces[P]["rows"])
            self._show_piece(P)
            ls["base"] = P
        ls["i"] = i
        self._apply_mask()

    def _apply_mask(self) -> None:
        """Mask -> whole objects mostly visible inside it + the other visible Gaussians in it (and the layer just
        behind them); depth-gated so background seen through gaps is not taken."""
        ls, sc = self.last_sam, self.cfg["viewer"]["sam"]
        view, cam, key = ls["view"], ls["cam"], ls["key"]
        mask = ls["masks"][ls["i"]] & (view.gid >= 0)
        if not mask.any():
            return
        front = np.percentile(view.depth[mask], 10)
        ys, xs = np.nonzero(mask)
        size = max(np.ptp(xs), np.ptp(ys)) * front / cam.fx
        far = front + max(sc["depth_gate_min"], sc["depth_gate_size"] * size)
        mask &= view.depth <= far
        g_all, g_in = view.gid[view.gid >= 0], view.gid[mask]
        tot, inm = Counter(key[g_all].tolist()), Counter(key[g_in].tolist())
        whole = [k for k, n in inm.items() if k >= 0 and k in self.src and n >= sc["member_frac"] * tot[k]
                 and tot[k] >= sc["min_visible_px"]]
        vis = np.unique(g_in[~np.isin(key[g_in], whole)])
        u, v, z = cam.project(ls["xyz"])
        ok = np.isfinite(u) & (z > 0) & (z <= far) & ~np.isin(key, whole)
        ui, vi = np.round(u[ok]).astype(int), np.round(v[ok]).astype(int)
        inside = (ui >= 0) & (ui < cam.W) & (vi >= 0) & (vi < cam.H)
        cand = np.flatnonzero(ok)[inside][mask[vi[inside], ui[inside]]]
        if len(vis):
            d, _ = cKDTree(ls["xyz"][vis]).query(ls["xyz"][cand], k=1, workers=-1)
            cand = cand[d <= sc["thickness"]]
        rows = ls["rows"][np.union1d(vis, cand)]
        objs: list[int] = []
        for k in whole:
            objs += self.pieces[k]["objs"] if k in self.pieces else [k, *self.parts.get(k, [])]
            rows = np.concatenate([rows, self._rows(k)])
        base = ls["base"]
        ls["base_piece"] = None
        if base is not None and base in self.src:
            if base in self.pieces:
                pc = self.pieces[base]
                ls["base_piece"] = (pc["rows"], pc["objs"], pc["name"])
            self._extend(base, rows, [*objs, *[k for k in whole if k in self.pieces]])
            ls["P"] = self.selected
            return
        for k in [k for k in whole if k in self.pieces]:                  # clicked into an existing selection
            pc = self.pieces.pop(k)
            self.taken[pc["rows"]] = False
            for h in (self.splat_h.pop(k, None), self.mesh_h.pop(k, None)):
                if h is not None:
                    h.remove()
            del self.src[k]
        P = self._make_piece(rows, [o for o in objs if o < len(self.objs)])
        self._refresh_rows(self.pieces[P]["rows"])
        self._show_piece(P)
        ls["P"] = P
        self.select(P)
        self._save_edits()

    # --- gui ---
    def _build_gui(self, seg: Any) -> None:
        sc, gui = self.server.scene, self.server.gui
        cloud = sc.add_point_cloud("/scan", self.pts, (np.clip(seg["rgb"], 0, 1) * 255).astype(np.uint8),
                                   point_size=0.01, visible=False)
        dev_mm = np.load(out_path(self.cfg, "deviation.npz"))["dist_m"] * 1000
        has = np.isfinite(dev_mm)
        heat = sc.add_point_cloud("/heatmap", self.pts[has], deviation_colors(dev_mm[has]), point_size=0.012,
                                  visible=False)
        self.md = gui.add_markdown("Click an object to select it.")
        with gui.add_folder("Edit selected"):
            b_dup, b_del = gui.add_button("Duplicate"), gui.add_button("Delete")
            b_reset, b_none = gui.add_button("Reset object"), gui.add_button("Deselect")
            add_nudge_gui(
                gui=gui,
                vc=self.cfg["viewer"],
                up=np.array([0.0, 0.0, 1.0]),
                units_to_m=1.0,
                on_nudge=self.nudge,
            )
            self.smart = gui.add_checkbox("Smart select (SAM)", False)
            b_bigger, b_smaller = gui.add_button("Selection: bigger"), gui.add_button("Selection: smaller")
            self.add_parts = gui.add_checkbox("Click adds parts", False)
            b_ungroup = gui.add_button("Ungroup")
            b_all = gui.add_button("Restore all")
            self.md_edits = gui.add_markdown("Edits: none")
        with gui.add_folder("View"):
            self.show_splat = gui.add_checkbox("Show splat (photoreal)", True)
            self.show_bim = gui.add_checkbox("Show BIM model", False)
            self.mode = gui.add_dropdown("BIM colour", ("Scan colours", "IFC class", "Deviation vs scan"),
                                         initial_value="Scan colours")
            self.show_proxy = gui.add_checkbox("BIM: show unclassified", True)
            self.opacity = gui.add_slider("BIM opacity", 0.1, 1.0, 0.05, 1.0)
            show_scan = gui.add_checkbox("Scan points", False)
            show_heat = gui.add_checkbox("Deviation heat map", False)
            show_rem = gui.add_checkbox("Show removable", False, disabled=self.allowed is None)
        gui.add_markdown("  \n".join(f"<span style='color:rgb{c}'>&#9632;</span> {k[3:]}"
                                      for k, c in CLASS_COLORS.items()))
        gui.add_markdown(f"Deviation: green <= {DEV_GOOD_MM} mm, red >= {DEV_BAD_MM} mm  \n"
                         f"Scan -> model median {np.median(dev_mm[has]):.0f} mm, "
                         f"p90 {np.percentile(dev_mm[has], 90):.0f} mm")

        b_dup.on_click(lambda _: self.duplicate())
        b_del.on_click(lambda _: self.delete())
        b_reset.on_click(lambda _: self.reset())
        b_none.on_click(lambda _: self.select(None))
        b_ungroup.on_click(lambda _: self.ungroup())
        b_bigger.on_click(lambda _: self.resize_selection(1))
        b_smaller.on_click(lambda _: self.resize_selection(-1))
        b_all.on_click(lambda _: self.restore_all())
        self.show_splat.on_update(lambda _: [setattr(h, "visible", self.show_splat.value and k not in self.removed)
                                             for k, h in self.splat_h.items()])
        self.show_bim.on_update(lambda _: self._build_meshes())
        for w in (self.mode, self.opacity, self.show_proxy):
            w.on_update(lambda _: self._build_meshes())
        show_scan.on_update(lambda _: setattr(cloud, "visible", show_scan.value))
        show_heat.on_update(lambda _: setattr(heat, "visible", show_heat.value))
        boxes = []
        for k in sorted(self.allowed or []):
            P = self.pts[self.pt_lab == k]
            boxes.append(sc.add_box(f"/removable/{k}", color=(0, 200, 255), dimensions=tuple(np.ptp(P, 0) + 0.04),
                                    wireframe=True, position=tuple((P.min(0) + P.max(0)) / 2), visible=False))
        show_rem.on_update(lambda _: [setattr(b, "visible", show_rem.value) for b in boxes])


def main() -> None:
    view = BimView(load_config())
    print(f"[bim] http://localhost:{view.server.get_port()}", flush=True)
    while True:
        time.sleep(10)


if __name__ == "__main__":
    main()
