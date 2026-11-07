"""Viser simulation: splat scene + 3D frames + walking avatar + proximity panel.

Inputs: data/*.ply (via load_splat), out/scene_graph.json, out/path.json.
Run:    python -m viewer.app      then open http://localhost:8080
"""
from __future__ import annotations

import base64
import json
import threading
import time
from typing import Any

import numpy as np
import viser
from scipy.spatial import cKDTree
from viser.transforms import SO3

from copilot.agent import STATUS_DOT, diagnose, to_markdown
from copilot.card import DETAIL_SECTIONS, asset_card, asset_details
from copilot.evidence import build_evidence, load_pack
from copilot.voice import briefing
from pipeline.common import ROOT, load_config, out_path, repo_path
from pipeline.load_splat import covariances, load_splat

Cfg = dict[str, Any]
# Viser renders markdown tables as Mantine tables without borders; this <style> (injected once through an
# HTML widget) gives every table in the panel cell borders, a shaded header row and compact padding.
TABLE_CSS = """<style>
.mantine-Table-table { border-collapse: collapse !important; width: 100%; max-width: 100%; margin: 4px 0 8px;
  font-size: 0.9em; }
.mantine-Table-table th, .mantine-Table-table td {
  border: 1px solid #c9ced6 !important; padding: 3px 5px !important; vertical-align: top;
  overflow-wrap: break-word; word-break: normal; hyphens: auto; }
.mantine-Table-table:has(th:nth-child(5)) { font-size: 0.84em; }   /* wide tables: smaller text so they fit */
.mantine-Table-table:has(th:nth-child(5)) th, .mantine-Table-table:has(th:nth-child(5)) td {
  padding: 3px 4px !important; }
.mantine-Table-table thead th { background: #eef1f5; font-weight: 600; text-align: left !important; }
.mantine-Table-table tbody tr:nth-child(even) td { background: #f8f9fb; }
[data-mantine-color-scheme="dark"] .mantine-Table-table th,
[data-mantine-color-scheme="dark"] .mantine-Table-table td { border-color: #4a4f57 !important; }
[data-mantine-color-scheme="dark"] .mantine-Table-table thead th { background: #2c2f34; }
[data-mantine-color-scheme="dark"] .mantine-Table-table tbody tr:nth-child(even) td { background: #25272b; }
</style>"""
EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


# ---------- data ----------

def load_scene(cfg: Cfg) -> tuple[dict, dict, dict | None]:
    """Scene graph, walk path, and the inspection tour (out/tour.json) if enabled and present.
    With a tour, the walk follows the tour points instead of the direct path."""
    with open(out_path(cfg, "scene_graph.json"), encoding="utf-8") as f:
        sg = json.load(f)
    with open(out_path(cfg, "path.json"), encoding="utf-8") as f:
        path = json.load(f)
    tour = None
    tp = out_path(cfg, "tour.json")
    if cfg["viewer"].get("use_tour", True) and tp.exists():
        with open(tp, encoding="utf-8") as f:
            tour = json.load(f)
        ids = {n["id"] for n in sg["objects"] if n["kind"] == "object"}
        if tour["stops"][-1]["object_id"] != path["target_object"] or any(
                st["object_id"] not in ids for st in tour["stops"]):
            print("[viewer] out/tour.json is stale (target/objects changed); re-run pipeline.tour. Ignoring it.")
            tour = None
        else:
            path = {**path, "points": tour["points"]}
    return sg, path, tour


def stop_title(st: dict) -> str:
    """One-line identity + status of a tour/go-to stop (handles objects without an asset record)."""
    if not st.get("has_asset", True):
        return f"{st['object_id']} · {st['label']} · no asset record"
    return f"{st['tag']} · {st['condition_state']} · health {st['health_score']} · {st['issue_code']}"


def goto_record(n: dict, sample_index: int) -> dict:
    """Stop record for a single "Go to equipment" destination (asset or plain scene object)."""
    a = n.get("asset")
    rec = {"kind": "goto", "object_id": n["id"], "label": n["label"], "center": n["center"],
           "sample_index": sample_index, "has_asset": bool(a)}
    if a:
        rec.update({"tag": a["tag"], "asset_class": a["asset_class"], "condition_state": a["condition_state"],
                    "health_score": a["health_score"], "issue_code": a["issue_code"]})
    else:
        rec.update({"tag": n["id"], "asset_class": n["label"], "condition_state": "no_record",
                    "health_score": None, "issue_code": "-"})
    return rec


def stop_text(stop: dict, i: int, n: int) -> str:
    return (f"**Stop {i}/{n}: {stop['tag']}** · {stop['asset_class']} · {stop['condition_state']} · "
            f"health **{stop['health_score']}** · {stop['issue_code']}"
            + ("  \n_Faulty target: copilot diagnosis below._" if stop["kind"] == "target" else ""))


def status_of(node: dict) -> str:
    return node["asset"]["condition_state"] if node["asset"] else "no_record"


def box_edges(lo: list[float], hi: list[float]) -> np.ndarray:
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return np.array([[c[a], c[b]] for a, b in EDGES], np.float32)


def edges_for(nodes: list[dict], colors: dict[str, list[int]]) -> tuple[np.ndarray, np.ndarray]:
    segs = [box_edges(n["bbox_min"], n["bbox_max"]) for n in nodes]
    cols = [np.tile(np.array(colors[status_of(n)], np.uint8), (12, 2, 1)) for n in nodes]
    if not segs:
        return np.zeros((0, 2, 3), np.float32), np.zeros((0, 2, 3), np.uint8)
    return np.concatenate(segs), np.concatenate(cols)


AVATAR_LENGTHS = ("hip_h", "hip_dx", "leg", "torso", "torso_h", "shoulder_h", "shoulder_dx",
                  "arm", "head_h", "head_r", "stride", "bob")


def scaled_avatar(av: dict) -> dict:
    """Apply avatar.scale to every length (sizes, heights, stride); angles and colours unchanged."""
    s = float(av.get("scale", 1.0))
    out = dict(av)
    for k in AVATAR_LENGTHS:
        v = av[k]
        out[k] = [x * s for x in v] if isinstance(v, list) else v * s
    return out


LABEL_MODES = ("Equipment", "All objects", "Nearby only", "Off")


def label_visible(mode: str, is_equipment: bool) -> bool:
    """Persistent label visibility: Equipment = mapped or non-pipe objects; All = everything."""
    return mode == "All objects" or (mode == "Equipment" and is_equipment)


def label_text(n: dict) -> str:
    """'OBJ-0018 · P-002 pump · attention_required · health 45' or 'OBJ-0142 · ladder/stairs 0.90'."""
    if n["asset"]:
        a = n["asset"]
        return f"{n['id']} · {a['tag']} {n['label']} · {a['condition_state']} · health {a['health_score']}"
    conf = n.get("label_confidence")
    return f"{n['id']} · {n['label']}" + (f" {conf:.2f}" if conf is not None else "")


def nearby_text(near: list[dict], others: list[dict], radius: float) -> str:
    rows = ["| ID | Tag | Name | Health | Status | Issue |", "|---|---|---|---|---|---|"]
    order = {"attention_required": 0, "degraded": 1, "watch": 2, "healthy": 3}
    for n in sorted(near, key=lambda n: order.get(status_of(n), 9)):
        a = n["asset"]
        rows.append(f"| {n['id']} | **{a['tag']}** | {n['label']} | {a['health_score']} | "
                    f"{a['condition_state']} | {a['issue_code']} |")
    counts: dict[str, int] = {}
    for n in others:
        counts[n["label"]] = counts.get(n["label"], 0) + 1
    other = ", ".join(f"{k} ×{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    head = f"**Within {radius:.1f} units:** {len(near)} mapped assets · {len(others)} other objects\n\n"
    body = "\n".join(rows) if near else "_no mapped assets nearby_"
    return head + body + (f"\n\n_No asset record:_ {other}" if other else "")


def selection_text(n: dict, u2m: float) -> str:
    e = np.array(n["extent"]) * u2m
    f = n.get("features", {})
    src = n.get("label_source", "geometry")
    conf = n.get("label_confidence")
    md = [f"### {n['id']} · {n['label']}",
          f"Name from **{src}**" + (f" (confidence {conf:.2f})" if conf is not None else "")
          + (f": _{n['label_reason']}_" if n.get("label_reason") else ""),
          "",
          f"Size **{e[0]:.2f} × {e[1]:.2f} × {e[2]:.2f} m** · bottom {f.get('height_bottom', 0) * u2m:.2f} m "
          f"above floor · {n['n_points']} points" + (" · overhead" if "overhead" in n.get("tags", []) else "")]
    if n.get("label_vlm"):
        v = n["label_vlm"]
        md += ["", f"_Vision model's low-confidence guess: {v.get('label')} ({v.get('confidence')})_"]
    rp = n.get("repair")
    if rp:
        if rp.get("family") == "context":
            md += ["", f"**Repaired boundary** (context, {rp.get('boundary_status', 'unreviewed')}): "
                       f"{rp.get('region')}/{rp.get('cid')} proposed as _{rp.get('cls')}_ ({rp.get('confidence')}) by "
                       f"{rp.get('model')} on {rp.get('renderer')} renders, refined by SAM 2.1 masks + weighted visible "
                       f"3D support. Boundary, class and asset identity are separate, all unverified; revert via "
                       f"`repair.context.disabled`."]
        else:
            md += ["", f"**Repaired boundary** (geometry): {len(rp.get('leaves', []))} pipe pieces joined by "
                       f"centreline/thickness/gap checks."]
    if n.get("fragments"):
        md += [f"_{len(n['fragments'])} fine fragment(s); baseline objects: "
               f"{', '.join(n.get('baseline_ids', [])[:6]) or '-'}_"]
    if n.get("assembly"):
        md += [f"_Assembly {n['assembly']['id']} ({n['assembly']['note']})_"]
    a = n["asset"]
    if a:
        md += ["", "| Tag | Class | Health | Status | Urgency |", "|---|---|---|---|---|",
               f"| **{a['tag']}** | {a['asset_class']} | **{a['health_score']}** | {a['condition_state']} | "
               f"{a['urgency']} |", "",
               f"Issue: **{a['issue_code']}**: {a['issue_summary']}  ",
               f"Next check: {a['recommended_next_check']}  ",
               f"_3D mapping: {a['mapping_method']} (confidence {a['mapping_confidence']}) · identity "
               f"{'VERIFIED' if a.get('identity_verified') else 'unverified'} ({a.get('identity_basis', '?')})_"]
    else:
        md += ["", "_No asset record in the Excel pack (not mapped)._"]
    return "\n".join(md)


# ---------- viewer ----------

class Sim:
    def __init__(self, cfg: Cfg) -> None:
        self.cfg, self.vc = cfg, cfg["viewer"]
        self.sg, path, self.tour = load_scene(cfg)
        self.nodes = [n for n in self.sg["objects"] if n["kind"] == "object"]
        self.pts = np.array(path["points"], np.float64)
        seg = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        self.s = np.concatenate([[0], np.cumsum(seg)])
        # distance along the walk at which each tour stop is reached (pauses happen there)
        self.stops = self.tour["stops"] if self.tour else []
        self.stop_s = [float(self.s[min(st["sample_index"], len(self.s) - 1)]) for st in self.stops]
        self.next_stop = 0
        self.pause_until = self.pause_left = 0.0
        self.face_target: np.ndarray | None = None   # avatar faces this point while stopped
        self._saved_cams: dict[int, tuple] = {}      # each browser's own camera, restored when POV is turned off
        self._pov_look: np.ndarray | None = None     # smoothed POV look-at point
        self.stop_active: dict | None = None         # tour stop currently highlighted
        self.walk_mode = "tour"                      # "tour" (ends at faulty target) | "goto" (one equipment)
        self.tour_route = (self.pts.copy(), list(self.stops))
        self.planner: Any = None                     # pipeline.tour.WalkPlanner, created on first "Go to"
        self.tree = cKDTree(np.array([n["center"] for n in self.nodes])[:, [0, 2]])
        self.max_half = max(float(np.hypot(n["extent"][0], n["extent"][2])) / 2 for n in self.nodes)
        self.target = next(n for n in self.nodes if n["id"] == self.sg["target"]["object_id"])
        self.dist = 0.0
        self.playing = False
        self.arrived = False
        self._diag: dict | None = None           # prefetched target diagnosis
        self._audio: bytes | None = None         # spoken briefing (mp3) for the target
        self._prefetching = False                # silent target diagnosis + voice running
        self._busy = False                       # a manual copilot job is running
        end_gap = self.box_gap(self.target, self.pts[-1])
        self.arrive_gap = max(float(cfg["viewer"]["voice"]["arrive_gap"]), end_gap + 0.05)
        if end_gap > cfg["viewer"]["voice"]["arrive_gap"]:
            print(f"[viewer] path ends {end_gap:.2f} u from the target box; arrive_gap raised to "
                  f"{self.arrive_gap:.2f}")
        seg = np.load(out_path(cfg, "segment_points.npz"))     # for click picking on real points
        self.pick_xyz, self.pick_lab = seg["xyz"].astype(np.float64), seg["labels"]
        self.selected: dict | None = None
        self.node_by_id = {n["id"]: n for n in self.nodes}
        self.movable: dict[str, dict] = {}           # objects in removable_rows.npz: own splat, can be edited
        self.inst: dict[str, dict] = {}              # shown instances (originals + copies) by name
        self.sel_inst: str | None = None
        self._picked_inst: str | None = None
        self.show_movable = False
        self.gizmo: Any = None
        self.server = viser.ViserServer(port=self.vc["port"])
        self.server.scene.set_up_direction(self.cfg["scene"]["up_direction"])
        self._labels: list[Any] = []
        self.av = scaled_avatar(self.vc["avatar"])
        self._build_scene()
        self._build_gui()
        self.server.scene.on_click()(self.on_scene_click)

    # --- splat display groups (ceiling / walls / room) ---
    def _load_splat_groups(self) -> None:
        """Load the splat once; tag every Gaussian as ceiling, wall or room, plus brightness/inside masks."""
        d = load_splat(str(repo_path(self.cfg["paths"]["ply"])))
        dc, sc_cfg = self.vc["display"], self.cfg["scene"]
        x = d["xyz"]
        self.sp = {"xyz": x, "rgb": d["rgb"], "opacity": d["opacity"][:, None]}
        if self.vc["render_splats"]:
            self.sp["cov"] = covariances(d["scales"], d["quats_wxyz"]).astype(np.float32)
        h = sc_cfg["floor_y"] - x[:, 1]                                   # height above floor (up = -y)
        walls = [p for p in self.cfg["segment"]["known_planes"] if p["label"] == "wall"]
        near_wall = np.zeros(len(x), bool)
        for p in walls:
            near_wall |= np.abs(x[:, p["axis"]] - p["value"]) < dc["wall_band"]
        xs = [p["value"] for p in walls if p["axis"] == 0]
        zs = [p["value"] for p in walls if p["axis"] == 2]
        m = dc["crop_margin"]
        inside = ((x[:, 0] > min(xs) - m) & (x[:, 0] < max(xs) + m)
                  & (x[:, 2] > min(zs) - m) & (x[:, 2] < max(zs) + m)) if xs and zs else np.ones(len(x), bool)
        self.sp_lum = d["rgb"] @ np.array([0.299, 0.587, 0.114])
        self.sp_inside = inside
        self.sp_group = np.where(h > dc["ceiling_cut_h"], "ceiling", np.where(near_wall, "walls", "room"))
        self.h_splat: dict[str, Any] = {}
        self.sp_movable = np.zeros(len(x), bool)
        if self.vc["render_splats"]:
            idx = {n["id"]: i for i, n in enumerate(self.sg["objects"])}
            for oid, r in self.movable_rows(d["row_ids"]).items():
                self.sp_movable[r] = True
                n = self.node_by_id[oid]
                c = np.array(n["center"])
                self.movable[oid] = {"c": c, "xyz": x[r] - c, "rgb": d["rgb"][r], "op": self.sp["opacity"][r],
                                     "cov": self.sp["cov"][r], "pick": self.pick_xyz[self.pick_lab == idx[oid]] - c,
                                     "box": box_edges(np.array(n["bbox_min"]) - c, np.array(n["bbox_max"]) - c)}

    def movable_rows(self, row_ids: np.ndarray) -> dict[str, np.ndarray]:
        p = out_path(self.cfg, "removable_rows.npz")
        if not p.exists():
            return {}
        z = np.load(p)
        return {oid: np.searchsorted(row_ids, z[oid]) for oid in z.files if oid in self.node_by_id}

    def rebuild_splats(self, dark_threshold: float, crop: bool) -> dict[str, int]:
        """(Re)send the three splat groups with the brightness / crop filter; keep their visibility."""
        sc = self.server.scene
        keep = self.sp_lum >= dark_threshold
        if crop:
            keep &= self.sp_inside
        counts = {}
        for g in ("room", "walls", "ceiling"):
            m = keep & (self.sp_group == g) & ~self.sp_movable
            vis = self.h_splat[g].visible if g in self.h_splat else self.vc["display"].get(f"show_{g}", True)
            if self.vc["render_splats"]:
                h = sc.add_gaussian_splats(f"/splat/{g}", centers=self.sp["xyz"][m], rgbs=self.sp["rgb"][m],
                                           opacities=self.sp["opacity"][m], covariances=self.sp["cov"][m])
            else:
                h = sc.add_point_cloud(f"/splat/{g}", self.sp["xyz"][m],
                                       (self.sp["rgb"][m] * 255).astype(np.uint8), point_size=self.vc["point_size"])
            h.visible = vis
            self.h_splat[g] = h
            counts[g] = int(m.sum())
        return counts

    # --- scene ---
    def _build_scene(self) -> None:
        sc = self.server.scene
        self._load_splat_groups()
        self.rebuild_splats(self.vc["display"]["dark_threshold"], self.vc["display"]["crop_outside_walls"])
        segs, cols = edges_for([n for n in self.nodes if n["id"] not in self.movable], self.vc["colors"])
        self.h_boxes = sc.add_line_segments("/boxes", segs, cols, thickness=self.vc["box_thickness"],
                                            thickness_units="screen")
        # before/after: baseline boxes (orange) and repaired objects (magenta = context, cyan = geometry)
        bp = ROOT / "out_baseline" / "assets.json"
        base = [a for a in json.loads(bp.read_text(encoding="utf-8")) if a["kind"] == "object"] if bp.exists() else []
        self.n_baseline = len(base)
        bs = np.concatenate([box_edges(a["bbox_min"], a["bbox_max"]) for a in base]) if base else np.zeros((0, 2, 3), np.float32)
        self.h_base = sc.add_line_segments("/baseline_boxes", bs, (255, 140, 0), thickness=1.5,
                                           thickness_units="screen", visible=False)
        ctx = [n for n in self.nodes if n.get("repair", {}).get("family") == "context"]
        geo = [n for n in self.nodes if n.get("repair", {}).get("family") == "geometry"]
        self.n_ctx, self.n_geo = len(ctx), len(geo)
        segs_r = [box_edges(n["bbox_min"], n["bbox_max"]) for n in ctx + geo]
        cols_r = [np.tile(np.array([255, 0, 200] if n in ctx else [0, 200, 255], np.uint8), (12, 2, 1)) for n in ctx + geo]
        self.h_rep = sc.add_line_segments("/repaired_boxes", np.concatenate(segs_r) if segs_r else np.zeros((0, 2, 3), np.float32),
                                          np.concatenate(cols_r) if cols_r else np.zeros((0, 2, 3), np.uint8),
                                          thickness=3.0, thickness_units="screen", visible=False)
        self.draw_path()
        self.h_tour_markers: list[Any] = []
        self.draw_tour_markers()
        t = self.target
        sc.add_label("/target_label", f"TARGET {t['asset']['tag']} · {t['asset']['issue_code']}",
                     position=(t["center"][0], t["bbox_min"][1] - 0.1, t["center"][2]),
                     anchor="bottom-center")
        self.h_highlight = sc.add_line_segments("/highlight", *edges_for([], self.vc["colors"]),
                                                thickness=self.vc["highlight_thickness"],
                                                thickness_units="screen")
        self.h_stop_box = sc.add_line_segments("/stop/box", *edges_for([], self.vc["colors"]),
                                               thickness=self.vc["stop_box_thickness"], thickness_units="screen")
        self.h_stop_fill: Any = None
        self.h_stop_label: Any = None
        self.h_selected = sc.add_line_segments("/selected", *edges_for([], self.vc["colors"]),
                                               thickness=self.vc["selected_thickness"], thickness_units="screen")
        # one persistent label per object; which ones are visible depends on the Labels menu
        self.is_equipment = [bool(n["asset"]) or n["label"] not in self.vc["near_label_skip"]
                             for n in self.nodes]
        mode = self.vc["label_mode"]
        for oid in self.movable:
            self.add_instance(oid, oid)
        self.h_all_labels = [sc.add_label(f"/movable/{n['id']}/label" if n["id"] in self.movable else
                                          f"/all_labels/{n['id']}", label_text(n),
                                          position=((0.0, n["bbox_min"][1] - n["center"][1], 0.0) if n["id"] in self.movable
                                                    else (n["center"][0], n["bbox_min"][1], n["center"][2])),
                                          anchor="bottom-center",
                                          font_screen_scale=(self.vc["equipment_label_scale"] if eq
                                                             else self.vc["all_label_scale"]),
                                          visible=label_visible(mode, eq))
                             for n, eq in zip(self.nodes, self.is_equipment)]
        self.load_edits()
        self._build_avatar()
        sc.add_icosphere("/avatar/ring", radius=1.0, color=tuple(self.vc["avatar"]["ring_color"]),
                         opacity=0.15,
                         scale=(self.vc["proximity_radius"], 0.01, self.vc["proximity_radius"]))

    def _build_avatar(self) -> None:
        """Articulated worker. Local frame: forward +z, left/right x, up = -y (native).
        Limbs hang from pivot frames (hips, shoulders) that swing about local x."""
        sc, av = self.server.scene, self.av
        self.h_avatar = sc.add_frame("/avatar", show_axes=False)
        self.h_body = sc.add_frame("/avatar/body", show_axes=False)       # bobs while walking
        col = {k: tuple(v) for k, v in av["colors"].items()}
        sc.add_box("/avatar/body/torso", col["vest"], tuple(av["torso"]),
                   position=(0, -av["torso_h"], 0))
        sc.add_box("/avatar/body/belt", col["trousers"], (av["torso"][0] * 1.02, 0.06, av["torso"][2] * 1.05),
                   position=(0, -(av["hip_h"] + 0.03), 0))
        sc.add_icosphere("/avatar/body/head", radius=av["head_r"], color=col["skin"],
                         position=(0, -av["head_h"], 0))
        sc.add_icosphere("/avatar/body/helmet", radius=av["head_r"] * 1.15, color=col["helmet"],
                         position=(0, -(av["head_h"] + av["head_r"] * 0.35), 0), scale=(1, 0.6, 1))
        self.h_limbs: dict[str, Any] = {}
        for side, sx in (("l", 1), ("r", -1)):
            hip = sc.add_frame(f"/avatar/body/hip_{side}", show_axes=False,
                               position=(sx * av["hip_dx"], -av["hip_h"], 0))
            sc.add_box(f"/avatar/body/hip_{side}/leg", col["trousers"], tuple(av["leg"]),
                       position=(0, av["leg"][1] / 2, 0))
            sc.add_box(f"/avatar/body/hip_{side}/boot", col["boots"], (av["leg"][0] * 1.1, 0.06, 0.2),
                       position=(0, av["leg"][1] - 0.03, 0.04))
            sh = sc.add_frame(f"/avatar/body/shoulder_{side}", show_axes=False,
                              position=(sx * av["shoulder_dx"], -av["shoulder_h"], 0))
            sc.add_box(f"/avatar/body/shoulder_{side}/arm", col["vest"], tuple(av["arm"]),
                       position=(0, av["arm"][1] / 2, 0))
            sc.add_icosphere(f"/avatar/body/shoulder_{side}/hand", radius=av["arm"][0] * 0.6,
                             color=col["skin"], position=(0, av["arm"][1] + 0.02, 0))
            self.h_limbs[f"hip_{side}"], self.h_limbs[f"shoulder_{side}"] = hip, sh

    def pose_avatar(self, p: np.ndarray, moving: bool) -> None:
        av = self.av
        if self.face_target is not None:                                # at a tour stop: face the equipment
            tangent = self.face_target - p
        elif self.arrived:                                              # face the target
            tangent = np.array(self.target["center"]) - p
        else:
            tangent = self.tangent()
        yaw = np.arctan2(tangent[0], tangent[2])                        # local +z -> facing direction
        self._facing = np.array([np.sin(yaw), 0.0, np.cos(yaw)])         # used by the worker POV camera
        self.h_avatar.position = tuple(p)
        self.h_avatar.wxyz = (np.cos(yaw / 2), 0.0, np.sin(yaw / 2), 0.0)
        phase = 2 * np.pi * self.dist / av["stride"]
        swing = av["leg_swing"] * np.sin(phase) if moving else 0.0
        arm = -av["arm_swing"] * np.sin(phase) if moving else 0.0
        bob = -av["bob"] * abs(np.cos(phase)) if moving else 0.0         # up = -y

        def rot_x(a: float) -> tuple[float, float, float, float]:
            return (float(np.cos(a / 2)), float(np.sin(a / 2)), 0.0, 0.0)

        self.h_body.position = (0.0, bob, 0.0)
        self.h_limbs["hip_l"].wxyz, self.h_limbs["hip_r"].wxyz = rot_x(swing), rot_x(-swing)
        self.h_limbs["shoulder_l"].wxyz, self.h_limbs["shoulder_r"].wxyz = rot_x(arm), rot_x(-arm)

    def tangent(self) -> np.ndarray:
        i = int(np.clip(np.searchsorted(self.s, self.dist, side="right") - 1, 0, len(self.pts) - 2))
        t = self.pts[i + 1] - self.pts[i]
        return t / (np.linalg.norm(t) + 1e-9)

    # --- gui ---
    def _build_gui(self) -> None:
        """Tabs: Equipment (pick → walk → ask copilot; the only thing a field user needs),
        Inspection tour, More (view settings, clicked-object inspector, boundary-repair review)."""
        g = self.server.gui
        g.configure_theme(control_width="large")         # room for the asset card's tables
        g.add_html(TABLE_CSS)                             # grid lines for every markdown table in the panel
        g.add_markdown("## Valmet Field Copilot")
        tabs = g.add_tab_group()
        with tabs.add_tab("Equipment", icon="tool"):
            self._build_equipment_tab(g)
        with tabs.add_tab("Ask the MCP", icon="messages"):
            self._build_chat_gui(g)
        with tabs.add_tab("Inspection tour", icon="route"):
            self._build_tour_tab(g)
        if self.movable:
            with tabs.add_tab("Edit scene", icon="arrows-move"):
                self._build_edit_tab(g)
        with tabs.add_tab("More", icon="settings"):
            self._build_more_tab(g)

    # Equipment tab: the main flow
    def _equipment_options(self) -> None:
        order = {"attention_required": 0, "degraded": 1, "watch": 2, "healthy": 3}
        assets = sorted((n for n in self.nodes if n["asset"]),
                        key=lambda n: (order.get(status_of(n), 9), n["asset"]["tag"]))
        self.equip_opt: dict[str, str] = {}                 # tag -> dropdown text
        self.equip_node: dict[str, dict] = {}               # dropdown text -> node
        for n in assets:
            a = n["asset"]
            opt = f"{STATUS_DOT.get(a['condition_state'], '⚪')} {a['tag']} · {a['asset_class'].replace('_', ' ').lower()}"
            self.equip_opt[a["tag"]], self.equip_node[opt] = opt, n

    def selected_equipment(self) -> dict:
        return self.equip_node[self.dd_equip.value]

    def _build_equipment_tab(self, g: Any) -> None:
        self._equipment_options()
        self.dd_equip = g.add_dropdown("Equipment", list(self.equip_node),
                                       initial_value=self.equip_opt[self.target["asset"]["tag"]],
                                       hint="Sorted by condition: most urgent first. Clicking equipment in the "
                                            "3D view selects it here too.")
        b_walk = g.add_button("Walk to equipment", color="blue", icon="walk")
        self.b_ask = g.add_button("Ask what you want to know", color="green", icon="message-chatbot")
        self.md_walk = g.add_markdown("")
        self.md_loading = g.add_markdown("", visible=False)
        self.pb = g.add_progress_bar(0.0, animated=True, visible=False, color="blue")
        self.h_audio = g.add_html("", visible=False)          # spoken briefing player
        self.md_copilot = g.add_markdown("")
        icons = dict(zip(DETAIL_SECTIONS, ("📇", "📈", "🔔", "🔧", "🗂️", "📄", "⚠️", "📍")))
        self.md_details: dict[str, Any] = {}
        for name in DETAIL_SECTIONS:                              # full data, one collapsible table each
            with g.add_folder(f"{icons[name]} {name}", expand_by_default=False):
                self.md_details[name] = g.add_markdown("")
        self._load_token = 0
        self.load_equipment()

        @self.dd_equip.on_update
        def _(_e: Any) -> None:
            self.load_equipment()

        @b_walk.on_click
        def _(_e: Any) -> None:
            self.go_to(self.selected_equipment())

        @self.b_ask.on_click
        def _(e: Any) -> None:
            self.open_question_dialog(e.client)

    def choose_equipment(self, tag: str) -> None:
        """Select equipment in the Equipment tab (from a click, a walk or the tour) and load its data."""
        opt = self.equip_opt.get(tag)
        if opt and self.dd_equip.value != opt:
            self.dd_equip.value = opt
            self.load_equipment()

    def load_equipment(self, speak: bool = True) -> None:
        """Show the selected equipment's status + copilot diagnosis automatically, without spending credits:
        the saved AI answer when one exists for the same evidence, else the template from the Excel evidence.
        Then the copilot reads its summary aloud (ElevenLabs; cached per text, so repeats are free)."""
        n = self.selected_equipment()
        tag = n["asset"]["tag"]
        if self.arrived and n is self.target:                 # the tour's spoken briefing owns the panel
            self.fill_details(n)
            self.show_diagnosis()
            return
        self._load_token += 1
        token = self._load_token
        self.md_copilot.content = f"_Loading {tag}…_"
        self.set_audio(None)                                   # stop the previous equipment's summary
        self.fill_details(n)

        def current() -> bool:
            return token == self._load_token and not (self.arrived and self.selected_equipment() is self.target)

        def work() -> None:
            d = None
            try:
                d = diagnose(tag, None, self.cfg, allow_paid=False)
                text = self.equipment_view(tag, d)
            except Exception as e:  # noqa: BLE001 - keep the panel usable
                text = f"⚠️ _Could not load {tag}: {type(e).__name__}: {e}_"
            if not current():
                return
            self.md_copilot.content = text
            if speak and d is not None:
                _, audio = briefing(d, tag, self.cfg, opener=f"Copilot summary for {tag}.")
                if current():
                    self.set_audio(audio, f"🔊 Copilot summary · {tag}")
        threading.Thread(target=work, daemon=True).start()

    def fill_details(self, n: dict) -> None:
        """Full Excel + 3D record of the equipment in the collapsible tables (fast: Excel is cached)."""
        try:
            xl = load_pack(str(repo_path(self.cfg["paths"]["xlsx"])))
            det = asset_details(build_evidence(n["asset"]["tag"], self.cfg), xl, n, self.cfg["scene"]["units_to_m"])
        except Exception as e:  # noqa: BLE001 - keep the panel usable
            det = {k: f"⚠️ _Could not load: {type(e).__name__}: {e}_" for k in DETAIL_SECTIONS}
        for k, md in self.md_details.items():
            md.content = det.get(k, "")

    def set_audio(self, audio: bytes | None, caption: str = "") -> None:
        """Show (and autoplay) an mp3 in the Equipment tab, or remove the player (stops playback)."""
        if not audio:
            self.h_audio.visible = False
            self.h_audio.content = ""
            return
        b64 = base64.b64encode(audio).decode()
        self.h_audio.content = (f'<div style="font-weight:600;margin:4px 0">{caption}</div>'
                                f'<audio controls autoplay style="width:100%" '
                                f'src="data:audio/mpeg;base64,{b64}"></audio>')
        self.h_audio.visible = True

    def equipment_view(self, tag: str, d: dict) -> str:
        """Asset card from the Excel evidence; a saved AI answer (if any) goes on top of it."""
        card = asset_card(build_evidence(tag, self.cfg), compact=True)
        if d.get("meta", {}).get("source", "").startswith("template_fallback"):
            return card
        return self.answer_header(d) + to_markdown(d) + "\n\n---\n\n" + card

    @staticmethod
    def answer_header(d: dict) -> str:
        src = d.get("meta", {}).get("source", "")
        if src.startswith("template_fallback"):
            return "📋 _Built from the Excel evidence (no AI call). **Ask what you want to know** for an AI answer._\n\n"
        return f"🤖 _AI answer · {src}_\n\n"

    def open_question_dialog(self, client: Any) -> None:
        """Small dialog: question about the selected equipment -> live copilot answer (1 paid request)."""
        if self._busy:
            return
        tag = self.selected_equipment()["asset"]["tag"]
        with client.gui.add_modal(f"Ask the copilot about {tag}") as modal:
            client.gui.add_markdown("Uses one live Featherless request (repo budget: "
                                    f"{self.cfg['copilot']['budget']['max_paid_requests']} in total). "
                                    "The same question about the same data is answered from the saved copy.")
            tx = client.gui.add_text("Question", "What should I check first, and why?")
            b_go = client.gui.add_button("Ask", color="green", icon="send")
            b_cancel = client.gui.add_button("Cancel")

            @b_go.on_click
            def _(_e: Any) -> None:
                q = tx.value.strip()
                modal.close()
                if q:
                    self.run_job(tag, q, lambda d: setattr(self.md_copilot, "content",
                                                           self.answer_header(d) + to_markdown(d)))

            @b_cancel.on_click
            def _(_e: Any) -> None:
                modal.close()

    # Inspection tour tab
    def _build_tour_tab(self, g: Any) -> None:
        if self.tour:
            route = " → ".join(["door"] + [s["tag"] for s in self.stops])
            mode = "from Excel status" if self.tour["mode"] == "auto_excel" else "manual list"
            skipped = ", ".join(f"{s['tag']}" for s in self.tour["skipped"]) or "none"
            g.add_markdown(f"Walks past the warning assets and ends at the faulty **{self.target['asset']['tag']}**, "
                           f"where the copilot briefs the worker (spoken).  \nRoute ({mode}): {route} · "
                           f"{self.tour['length'] * self.cfg['scene']['units_to_m']:.0f} m · skipped: {skipped}")
        b_play = g.add_button("Play / Pause", color="blue", icon="player-play")
        b_reset = g.add_button("Reset", icon="refresh")
        self.sl_pos = g.add_slider("Position", 0.0, float(self.s[-1]), 0.01, 0.0)
        self.md_stop = g.add_markdown("_Press Play to start the tour._" if self.tour
                                      else "_No tour: run `python -m pipeline.tour`._")
        with g.add_folder("Nearby assets", expand_by_default=False):
            self.md_near = g.add_markdown("")

        @b_play.on_click
        def _(_e: Any) -> None:
            if self.walk_mode != "tour":
                self.set_walk_mode("tour")
            if self.dist >= self.s[-1]:
                self.dist, self.next_stop = 0.0, 0
                self.clear_stop_highlight()
            self.playing = not self.playing
            now = time.time()
            if self.playing:                    # resume the stop pause where it was frozen
                self.pause_until = now + self.pause_left if self.stop_active is not None else 0.0
                self.prefetch_diagnosis()
            else:
                self.pause_left = max(0.0, self.pause_until - now)
                self.blink_stop(0.0)            # freeze the stop highlight in its visible state

        @b_reset.on_click
        def _(_e: Any) -> None:
            if self.walk_mode != "tour":
                self.set_walk_mode("tour")
            self.playing, self.dist = False, 0.0
            self.leave_target()
            self.next_stop, self.pause_until = 0, 0.0
            self.clear_stop_highlight()
            if self.tour:
                self.md_stop.content = "_Press Play to start the tour._"
            self.update()

        @self.sl_pos.on_update
        def _(_e: Any) -> None:
            if not self.playing:
                self.dist = float(self.sl_pos.value)
                self.next_stop = sum(s <= self.dist - 1e-6 for s in self.stop_s)   # scrubbing skips pauses
                self.pause_until = self.pause_left = 0.0
                self.clear_stop_highlight()
                self.update()

    # More tab: view settings, clicked-object inspector, boundary-repair review
    def _build_more_tab(self, g: Any) -> None:
        c = self.sg["counts"]
        g.add_markdown(f"**{c['objects']} objects** extracted · {c['structures']} structural planes · "
                       f"**{c['mapped']} mapped** to Excel")
        with g.add_folder("View"):
            self.cb_pov = g.add_checkbox("Worker view (POV)", False,
                                         hint="See the room through the worker's eyes; turns with the worker. "
                                              "Untick to get your own camera back.")
            self.sl_speed = g.add_slider("Walk speed (u/s)", 0.1, 2.0, 0.1, self.vc["walk_speed"])
            self.sl_radius = g.add_slider("Proximity radius", 0.5, 3.0, 0.1, self.vc["proximity_radius"])
            self.dd_labels = g.add_dropdown("Labels", list(LABEL_MODES), initial_value=self.vc["label_mode"])
            self.cb_boxes = g.add_checkbox("Show all frames", True)
        dc = self.vc["display"]
        with g.add_folder("Scene display", expand_by_default=False):
            cb_ceiling = g.add_checkbox("Show ceiling", dc["show_ceiling"])
            cb_walls = g.add_checkbox("Show walls", dc["show_walls"])
            cb_crop = g.add_checkbox("Crop outside walls", dc["crop_outside_walls"])
            sl_dark = g.add_slider("Hide dark splats below", 0.0, 0.2, 0.01, dc["dark_threshold"])
            b_apply = g.add_button("Apply filter")
            md_disp = g.add_markdown("")
        with g.add_folder("Clicked object", expand_by_default=False):
            self.md_sel = g.add_markdown("_Click any object in the 3D view to inspect it._")
            self.img_sel = g.add_image(np.full((8, 24, 3), 235, np.uint8), visible=False)
            b_walk_click = g.add_button("Walk to this object")
            self.b_identify = g.add_button("Ask the AI to know this", hint="Uses 1 paid request from the team's AI budget")
            self.md_identify = g.add_markdown("")
        with g.add_folder("Boundary repair: before / after", expand_by_default=False):
            g.add_markdown(f"After (white frames): **{len(self.nodes)}** objects · **{self.n_ctx}** context repairs "
                           f"(magenta) · **{self.n_geo}** geometry pipe joins (cyan)  \nBefore: **{self.n_baseline}** "
                           f"baseline objects (orange). Fewer boxes is not the goal; check whole components.")
            self.cb_base = g.add_checkbox("Show baseline boxes (before)", False)
            self.cb_rep = g.add_checkbox("Highlight repaired objects", True)
            self.h_rep.visible = True
            reg = self.review_regions()
            self.dd_region = g.add_dropdown("Review region", ["-"] + list(reg), initial_value="-")
            self.md_region = g.add_markdown("_Pick a region: the camera flies to its calibrated view; the image shows "
                                            "baseline | geometry-only | geometry+context (white = reviewed reference)._")
            self.img_region = g.add_image(np.full((8, 24, 3), 235, np.uint8), visible=False)
            self.md_render = g.add_markdown("", visible=False)
            self.img_render = g.add_image(np.full((8, 24, 3), 235, np.uint8), visible=False)

        @self.cb_pov.on_update
        def _(_e: Any) -> None:
            self.set_pov(self.cb_pov.value)

        @self.dd_labels.on_update
        def _(_e: Any) -> None:
            mode = self.dd_labels.value
            for h, eq in zip(self.h_all_labels, self.is_equipment):
                h.visible = label_visible(mode, eq)
            self.update()

        @self.cb_boxes.on_update
        def _(_e: Any) -> None:
            self.h_boxes.visible = self.cb_boxes.value
            for it in self.inst.values():
                it["box"].visible = self.cb_boxes.value

        @self.sl_radius.on_update
        def _(_e: Any) -> None:
            r = float(self.sl_radius.value)
            self.server.scene.add_icosphere("/avatar/ring", radius=1.0,
                                            color=tuple(self.vc["avatar"]["ring_color"]),
                                            opacity=0.15, scale=(r, 0.01, r))
            self.update()

        @cb_ceiling.on_update
        def _(_e: Any) -> None:
            self.h_splat["ceiling"].visible = cb_ceiling.value

        @cb_walls.on_update
        def _(_e: Any) -> None:
            self.h_splat["walls"].visible = cb_walls.value

        @b_apply.on_click
        def _(_e: Any) -> None:
            md_disp.content = "_Updating splats…_"
            c = self.rebuild_splats(float(sl_dark.value), bool(cb_crop.value))
            md_disp.content = (f"room {c['room']:,} · walls {c['walls']:,} · ceiling {c['ceiling']:,} splats "
                               f"(of {len(self.sp_lum):,})")

        @b_walk_click.on_click
        def _(_e: Any) -> None:
            if self.selected is None:
                self.md_identify.content = "⚠️ Click an object in the 3D view first."
                return
            self.go_to(self.selected)

        @self.b_identify.on_click
        def _(_e: Any) -> None:
            if self.selected is None:
                return
            n = self.selected
            self.md_identify.content = f"_Asking {self.cfg['copilot']['model']} about {n['id']}…_"

            def work() -> None:
                from pipeline.label import identify_object
                self.md_identify.content = identify_object(n, self.cfg)
            threading.Thread(target=work, daemon=True).start()

        @self.cb_base.on_update
        def _(_e: Any) -> None:
            self.h_base.visible = self.cb_base.value

        @self.cb_rep.on_update
        def _(_e: Any) -> None:
            self.h_rep.visible = self.cb_rep.value

        @self.dd_region.on_update
        def _(e: Any) -> None:
            rid = self.dd_region.value
            if rid == "-":
                return
            r = self.review_regions()[rid]
            v = r["views"][0]
            for cl in self.server.get_clients().values():
                cl.camera.position = tuple(v["eye"])
                cl.camera.look_at = tuple(v["target"])
            refs = ", ".join(f"{x['id']} {x['cls']}" for x in r["refs"]) or "none (qualitative)"
            self.md_region.content = f"**{rid}** {r['name']}  \nReferences: {refs}"
            img = ROOT / "out_review" / "compare" / f"{rid}_v0.png"
            if img.exists():
                from PIL import Image
                im = Image.open(img).convert("RGB")
                im.thumbnail((1400, 400))
                self.img_region.image = np.asarray(im)
                self.img_region.visible = True
            rc = ROOT / "out_review" / "render_compare" / f"{rid}_v0.png"
            if rc.exists():                  # same calibrated camera: disc baseline vs EWA Gaussian splatting
                from PIL import Image
                im = Image.open(rc).convert("RGB")
                im.thumbnail((900, 1400))
                self.md_render.content = ("**Rendering** (same camera): left = disc baseline, right = EWA Gaussians; "
                                          "rows = clean RGB | depth | visibility | fragment overlay")
                self.img_render.image = np.asarray(im)
                self.md_render.visible = self.img_render.visible = True

    @staticmethod
    def review_regions() -> dict:
        import yaml
        p = ROOT / "review" / "regions.yaml"
        if not p.exists():
            return {}
        return {r["id"]: r for r in yaml.safe_load(p.read_text(encoding="utf-8"))["regions"]}

    # --- per-frame ---
    def position(self) -> np.ndarray:
        return np.array([np.interp(self.dist, self.s, self.pts[:, k]) for k in range(3)])

    def update(self) -> None:
        p = self.position()
        if self.walk_mode == "tour":            # "Go to equipment" never auto-diagnoses (paid budget)
            at_target = (self.dist >= self.s[-1] - 1e-6 and self.box_gap(self.target, p) <= self.arrive_gap
                         and not self.warning_stop_due())
            if at_target and not self.arrived:
                self.arrived = True
                self.on_arrival()
            elif not at_target and self.arrived:
                self.leave_target()
        self.pose_avatar(p, moving=self.playing and self.dist < self.s[-1])
        r = float(self.sl_radius.value)
        idx = self.tree.query_ball_point(p[[0, 2]], r + self.max_half)
        near_all = [self.nodes[i] for i in idx if self.box_gap(self.nodes[i], p) <= r]
        mapped = [n for n in near_all if n["asset"]]
        others = [n for n in near_all if not n["asset"]]
        self.h_highlight.points, self.h_highlight.colors = edges_for(near_all, self.vc["highlight_colors"])
        for h in self._labels:
            h.remove()
        show = []                                        # persistent labels cover the other modes
        if self.dd_labels.value == "Nearby only":
            show = mapped + [n for n in others if n["label"] not in self.vc["near_label_skip"]]
        self._labels = [self.server.scene.add_label(
            f"/near/{n['id']}", label_text(n),
            position=(n["center"][0], n["bbox_min"][1], n["center"][2]), anchor="bottom-center",
            font_screen_scale=1.0 if n["asset"] else self.vc["other_label_scale"])
            for n in show]
        self.md_near.content = nearby_text(mapped, others, r)

    def warning_stop_due(self) -> bool:
        """True while a tour warning stop is shown, or (during playback) still due at this spot.
        The last warning stop can sit at the path end next to the target; the target briefing waits
        until that stop's pause is over, so the worker isn't briefed on P-002 while facing another asset."""
        if self.stop_active is not None and self.stop_active["kind"] != "target":
            return True
        return self.playing and any(st["kind"] != "target" and self.stop_s[k] >= self.dist - 1e-6
                                    for k, st in enumerate(self.stops) if k >= self.next_stop)

    @staticmethod
    def box_gap(n: dict, p: np.ndarray) -> float:
        """x-z distance from the avatar to the object's box (0 when inside/touching)."""
        dx = max(n["bbox_min"][0] - p[0], 0.0, p[0] - n["bbox_max"][0])
        dz = max(n["bbox_min"][2] - p[2], 0.0, p[2] - n["bbox_max"][2])
        return float(np.hypot(dx, dz))

    # --- click to inspect ---
    def pick(self, origin: np.ndarray, direction: np.ndarray) -> dict | None:
        """Object whose real (segmented) points the click ray hits first."""
        d = direction / (np.linalg.norm(direction) + 1e-12)
        v = self.pick_xyz - origin
        t = v @ d
        perp = np.linalg.norm(v - np.outer(t, d), axis=1)
        tol = self.vc["pick_radius"] + self.vc["pick_radius_per_unit"] * np.maximum(t, 0)
        objs = self.sg["objects"]
        hit = (t > 0) & (perp < tol) & (self.pick_lab >= 0)
        hit &= np.array([objs[i]["kind"] == "object" for i in range(len(objs))])[np.maximum(self.pick_lab, 0)]
        hit &= ~np.isin(self.pick_lab, [i for i, o in enumerate(objs) if o["id"] in self.movable])
        best = (float(t[hit].min()), objs[int(self.pick_lab[np.where(hit)[0][np.argmin(t[hit])]])], None) \
            if hit.any() else (np.inf, None, None)
        for name, it in self.inst.items():                    # movable objects where they are now
            if it["removed"]:
                continue
            R = SO3(np.asarray(it["h"].wxyz)).as_matrix()
            o, dl = R.T @ (origin - np.asarray(it["h"].position)), R.T @ d
            v = self.movable[it["oid"]]["pick"] - o
            t = v @ dl
            perp = np.linalg.norm(v - np.outer(t, dl), axis=1)
            ok = (t > 0) & (perp < self.vc["pick_radius"] + self.vc["pick_radius_per_unit"] * np.maximum(t, 0))
            if ok.any() and float(t[ok].min()) < best[0]:
                best = (float(t[ok].min()), self.node_by_id[it["oid"]], name)
        self._picked_inst = best[2]
        return best[1]

    def on_scene_click(self, e: Any) -> None:
        n = self.pick(np.array(e.ray_origin), np.array(e.ray_direction))
        if n is not None:
            self.select(n, inst=self._picked_inst)

    def select(self, n: dict, inst: str | None = None) -> None:
        self.selected = n
        self.sel_inst = inst if inst is not None or n["id"] not in self.movable else n["id"]
        self.hide_gizmo()
        if self.movable:
            self.md_edit.content = (f"**{self.sel_inst}** selected." if self.sel_inst else
                                    "_This object can't be edited; pick one of the movable objects._")
        segs, _ = edges_for([n], self.vc["colors"])
        if self.sel_inst:
            segs = self.inst_box(self.sel_inst)
        self.h_selected.points = segs
        self.h_selected.colors = np.tile(np.array(self.vc["selected_color"], np.uint8), (len(segs), 2, 1))
        self.md_sel.content = selection_text(n, self.cfg["scene"]["units_to_m"])
        crop = out_path(self.cfg, "crops") / f"{n['id']}.png"
        if crop.exists():
            from PIL import Image
            self.img_sel.image = np.asarray(Image.open(crop).convert("RGB"))
            self.img_sel.visible = True
        else:
            self.img_sel.visible = False
        self.md_identify.content = ""
        if n["asset"]:                                               # clicked equipment -> Equipment tab
            self.choose_equipment(n["asset"]["tag"])

    # --- edit scene: move / remove / duplicate the objects that cut out cleanly (viewer.removable) ---
    def _build_edit_tab(self, g: Any) -> None:
        g.add_markdown(f"**{len(self.movable)} objects** cut out cleanly and can be moved, removed or duplicated. "
                       "Click one in the 3D view or pick it below.")
        dd = g.add_dropdown("Movable object", ["-", *self.movable], initial_value="-")
        cb_show = g.add_checkbox("Show movable objects", self.show_movable)
        self.md_edit = g.add_markdown("_Nothing selected._")
        b_move = g.add_button("Move", icon="arrows-move")
        b_dup = g.add_button("Duplicate", icon="copy")
        b_remove = g.add_button("Remove", icon="trash", color="red")
        b_restore = g.add_button("Restore object", hint="Put it back where it was and delete its copies")
        b_restore_all = g.add_button("Restore all")

        @dd.on_update
        def _(e: Any) -> None:
            if dd.value != "-":
                self.select(self.node_by_id[dd.value])
                self.fly_to(self.node_by_id[dd.value], e.client)
                dd.value = "-"                                          # the same object can be picked again

        @cb_show.on_update
        def _(_e: Any) -> None:
            self.show_movable = cb_show.value
            for it in self.inst.values():
                it["show"].visible = self.show_movable

        @b_move.on_click
        def _(_e: Any) -> None:
            if self.sel_inst and not self.inst[self.sel_inst]["removed"]:
                self.show_gizmo(self.sel_inst)

        @b_dup.on_click
        def _(_e: Any) -> None:
            if self.sel_inst:
                src = self.inst[self.sel_inst]
                oid = src["oid"]
                k = 1 + max([int(x.rsplit("_copy", 1)[1]) for x in self.inst if x.startswith(oid + "_copy")] or [0])
                name = f"{oid}_copy{k}"
                self.add_instance(name, oid, np.asarray(src["h"].position) + np.array(self.vc["duplicate_offset"]),
                                  src["h"].wxyz)
                self.select(self.node_by_id[oid], inst=name)
                self.show_gizmo(name)
                self.save_edits()

        @b_remove.on_click
        def _(_e: Any) -> None:
            if self.sel_inst:
                if self.sel_inst == self.inst[self.sel_inst]["oid"]:
                    self.inst[self.sel_inst]["removed"] = True
                    self.inst[self.sel_inst]["h"].visible = False
                else:                                                   # a copy is deleted for good
                    self.inst.pop(self.sel_inst)["h"].remove()
                self.md_edit.content = f"**{self.sel_inst}** removed."
                self.sel_inst = None
                self.hide_gizmo()
                self.h_selected.points, self.h_selected.colors = edges_for([], self.vc["colors"])
                self.save_edits()

        @b_restore.on_click
        def _(_e: Any) -> None:
            if self.selected is not None and self.selected["id"] in self.movable:
                self.restore(self.selected["id"])
                self.select(self.selected)
                self.save_edits()

        @b_restore_all.on_click
        def _(_e: Any) -> None:
            for oid in self.movable:
                self.restore(oid)
            self.hide_gizmo()
            self.save_edits()

    def add_instance(self, name: str, oid: str, pos: Any = None, wxyz: Any = (1.0, 0.0, 0.0, 0.0)) -> None:
        """Frame at the object's pose holding its splat, box and label, so they move together."""
        m, sc = self.movable[oid], self.server.scene
        h = sc.add_frame(f"/movable/{name}", show_axes=False, position=tuple(m["c"] if pos is None else pos),
                         wxyz=tuple(wxyz))
        sc.add_gaussian_splats(f"/movable/{name}/splat", centers=m["xyz"], rgbs=m["rgb"], opacities=m["op"],
                               covariances=m["cov"])
        box = sc.add_line_segments(f"/movable/{name}/box", m["box"],
                                   tuple(self.vc["colors"][status_of(self.node_by_id[oid])]),
                                   thickness=self.vc["box_thickness"], thickness_units="screen",
                                   visible=self.cb_boxes.value if hasattr(self, "cb_boxes") else True)
        show = sc.add_line_segments(f"/movable/{name}/show", m["box"], tuple(self.vc["movable_color"]),
                                    thickness=self.vc["selected_thickness"], thickness_units="screen",
                                    visible=self.show_movable)
        self.inst[name] = {"oid": oid, "h": h, "box": box, "show": show, "removed": False}

    def inst_box(self, name: str) -> np.ndarray:
        h = self.inst[name]["h"]
        R = SO3(np.asarray(h.wxyz)).as_matrix()
        return (self.movable[self.inst[name]["oid"]]["box"] @ R.T + np.asarray(h.position)).astype(np.float32)

    def show_gizmo(self, name: str) -> None:
        h = self.inst[name]["h"]
        self.hide_gizmo()
        self.gizmo = self.server.scene.add_transform_controls(
            "/gizmo", scale=0.6, position=h.position, wxyz=h.wxyz, depth_test=False,
            rotation_limits=((0, 0), (-1000, 1000), (0, 0)))               # yaw only (up = -y)

        @self.gizmo.on_update
        def _(_e: Any) -> None:
            h.position, h.wxyz = self.gizmo.position, self.gizmo.wxyz
            self.h_selected.points = self.inst_box(name)
            self.save_edits()

    def hide_gizmo(self) -> None:
        if self.gizmo is not None:
            self.gizmo.remove()
            self.gizmo = None

    def restore(self, oid: str) -> None:
        for name in [x for x in self.inst if x.startswith(oid + "_copy")]:
            self.inst.pop(name)["h"].remove()
        it = self.inst[oid]
        it["h"].position, it["h"].wxyz, it["h"].visible = tuple(self.movable[oid]["c"]), (1.0, 0.0, 0.0, 0.0), True
        it["removed"] = False

    def save_edits(self) -> None:
        out_path(self.cfg, "viewer_edits.json").write_text(json.dumps(
            {name: {"oid": it["oid"], "position": [float(x) for x in it["h"].position],
                    "wxyz": [float(x) for x in it["h"].wxyz], "removed": it["removed"]}
             for name, it in self.inst.items()}, indent=1), encoding="utf-8")

    def load_edits(self) -> None:
        p = out_path(self.cfg, "viewer_edits.json")
        if not p.exists():
            return
        for name, e in json.loads(p.read_text(encoding="utf-8")).items():
            if e["oid"] not in self.movable:
                continue
            if name not in self.inst:
                self.add_instance(name, e["oid"])
            it = self.inst[name]
            it["h"].position, it["h"].wxyz = tuple(e["position"]), tuple(e["wxyz"])
            it["removed"] = e["removed"]
            it["h"].visible = not e["removed"]

    # --- worker point of view ---
    def set_pov(self, on: bool) -> None:
        """Enter/leave the worker's view: hide the body (camera sits in the head), remember and later
        restore each browser's own camera."""
        self.h_body.visible = not on
        clients = self.server.get_clients().values()
        if on:
            self._pov_look = None
            self._saved_cams = {c.client_id: (c.camera.position, c.camera.look_at, c.camera.fov)
                                for c in clients}
            self.update_pov()
            return
        for c in clients:
            saved = self._saved_cams.get(c.client_id)
            if saved:
                with c.atomic():
                    c.camera.position, c.camera.look_at, c.camera.fov = saved
        self._saved_cams = {}

    def update_pov(self) -> None:
        """Camera at the worker's eyes, looking where the worker walks; at a stop or the target it looks
        at the equipment. Smoothed so turns at corners and stops are a pan, not a jump."""
        pc = self.vc["pov"]
        p = self.position()
        up = np.array([0.0, -1.0, 0.0])                                 # scene up = -y
        f = getattr(self, "_facing", self.tangent())
        eye = p + up * (self.av["head_h"] + float(self.h_body.position[1])) + f * pc["eye_forward"]
        focus = self.face_target if self.face_target is not None else (
            np.array(self.target["center"]) if self.arrived else None)
        if focus is not None:
            # turn to the equipment, but tilt the head only as far as a person would: low or overhead
            # equipment (box centre near the floor / ceiling) is looked at straight ahead, slightly down/up
            to = np.asarray(focus, np.float64) - eye
            horiz = to - up * float(to @ up)
            dist = float(np.linalg.norm(horiz))
            hdir = horiz / dist if dist > 1e-6 else f
            pitch = np.arctan2(float(to @ up), max(dist, 1e-6))          # + = up, - = down
            pitch = float(np.clip(pitch, -np.radians(pc["max_look_down_deg"]), np.radians(pc["max_look_up_deg"])))
            look = eye + pc["look_ahead"] * (hdir + up * np.tan(pitch))
        else:
            look = eye + f * pc["look_ahead"] - up * pc["look_down"]
        prev = getattr(self, "_pov_look", None)
        look = look if prev is None else prev + pc["smoothing"] * (look - prev)
        self._pov_look = look
        fov = float(np.radians(pc["fov_deg"]))
        for c in self.server.get_clients().values():
            with c.atomic():
                c.camera.up_direction = tuple(up)
                c.camera.fov = fov
                c.camera.position = tuple(eye)
                c.camera.look_at = tuple(look)

    def fly_to(self, n: dict, client: Any) -> None:
        c = np.array(n["center"])
        dist = max(self.vc["fly_min_dist"], self.vc["fly_dist_factor"] * float(np.linalg.norm(n["extent"])))
        pos = c + np.array([0.0, -self.vc["fly_height"], dist])   # in front (+z, room side), slightly above (-y)
        client.camera.position = tuple(pos)
        client.camera.look_at = tuple(c)

    # --- copilot jobs (one at a time, with a visible loading state) ---
    def set_loading(self, on: bool, text: str = "", frac: float = 0.0) -> None:
        self.md_loading.visible = self.pb.visible = on
        self.md_loading.content = text
        self.pb.value = frac
        self.b_ask.disabled = on
        self.b_ask.label = "Copilot is working…" if on else "Ask what you want to know"

    def run_job(self, tag: str, question: str | None, on_done: Any, allow_paid: bool = True) -> None:
        """Run diagnose() in a thread; show progress bar + elapsed seconds until it returns."""
        if self._busy:
            return
        self._busy = True
        t0, expected = time.time(), self.cfg["copilot"]["expected_latency_s"]
        q = f" · _{question}_" if question else ""

        def tick() -> None:
            while self._busy:
                el = time.time() - t0
                self.set_loading(True, f"⏳ **Copilot is analysing {tag}**{q}\n\n"
                                       f"Evidence pack → model → citation check · **{el:.0f} s** "
                                       f"(usually ~{expected} s)", min(95.0, 100 * el / expected))
                time.sleep(0.5)

        def work() -> None:
            try:
                d = diagnose(tag, question, self.cfg, allow_paid=allow_paid)
            finally:
                self._busy = False
                self.set_loading(False)
            on_done(d)

        threading.Thread(target=tick, daemon=True).start()
        threading.Thread(target=work, daemon=True).start()

    def prefetch_diagnosis(self) -> None:
        """Silently prepare the target diagnosis + spoken briefing while the avatar walks
        (cached/template answer: instant; live Featherless if auto_copilot_paid: ~60 s; TTS a few s).
        Nothing is shown until the worker stands at the target; if it is still running on arrival,
        the panel says so and the briefing plays when ready."""
        if self._diag is not None or self._prefetching:
            return
        self._prefetching = True
        tag = self.target["asset"]["tag"]

        def work() -> None:
            try:
                # automatic prefetch never spends LLM credits: cached answer for the same evidence,
                # else template (viewer.auto_copilot_paid: true allows a live Featherless call)
                d = diagnose(tag, None, self.cfg, allow_paid=self.vc.get("auto_copilot_paid", False))
                _, audio = briefing(d, tag, self.cfg)
                self._diag, self._audio = d, audio
            except Exception as e:  # noqa: BLE001 - show it instead of hanging on "preparing"
                print(f"[viewer] briefing failed: {type(e).__name__}: {e}")
                if self.arrived:
                    self.md_copilot.content = (self.arrival_header() + "⚠️ _Briefing failed; use "
                                               "**Ask what you want to know** below._")
                return
            finally:
                self._prefetching = False
            if self.arrived:
                self.show_diagnosis()
        threading.Thread(target=work, daemon=True).start()

    def arrival_header(self) -> str:
        a = self.target["asset"]
        return (f"#### 📍 Arrived at {a['tag']}\n"
                f"| Health | State | Urgency | Issue |\n|---|---|---|---|\n"
                f"| **{a['health_score']}** | {a['condition_state']} | **{a['urgency']}** | {a['issue_code']} |\n\n"
                f"_3D mapping: {a['mapping_method']} (confidence {a['mapping_confidence']})_\n\n---\n")

    def show_diagnosis(self) -> None:
        if not self.arrived or self._diag is None:
            return
        tag = self.target["asset"]["tag"]
        self.md_copilot.content = f"#### 📍 Arrived at {tag}\n\n" + self.equipment_view(tag, self._diag)
        if self._audio:
            self.set_audio(self._audio, f"🔊 Copilot briefing · {tag}")

    def on_arrival(self) -> None:
        self.choose_equipment(self.target["asset"]["tag"])
        if self._diag is not None:
            self.show_diagnosis()
            return
        self.md_copilot.content = self.arrival_header() + "_🎧 The copilot is preparing your briefing…_"
        self.prefetch_diagnosis()

    def leave_target(self) -> None:
        """Worker moved away from the target: stop the briefing and hide it until the next arrival."""
        self.arrived = False
        self.h_audio.visible = False
        self.h_audio.content = ""                                       # removes the <audio>, stops playback
        if not self._busy:
            self.load_equipment(speak=False)

    def run(self) -> None:
        self.update()
        print(f"[viewer] http://localhost:{self.vc['port']}  ({len(self.nodes)} objects, "
              f"path {self.s[-1]:.2f} u)")
        dt = 1.0 / self.vc["fps"]
        self._init_remote()
        while True:
            now = time.time()
            self.poll_remote(now)
            if self.playing and now < self.pause_until and self.stop_active is not None:
                self.blink_stop(now)                                   # pausing at a stop
            elif self.playing and self.stop_active is not None and self.stop_active["kind"] != "target":
                self.clear_stop_highlight()                            # walking on to the next stop
            if self.playing and now >= self.pause_until:
                nxt = self.stop_s[self.next_stop] if self.next_stop < len(self.stops) else float(self.s[-1])
                self.dist = min(self.dist + float(self.sl_speed.value) * dt, nxt, float(self.s[-1]))
                self.sl_pos.value = round(self.dist, 2)
                self.update()
                if self.next_stop < len(self.stops) and self.dist >= self.stop_s[self.next_stop] - 1e-6:
                    self.reach_stop(self.next_stop)
                if self.dist >= self.s[-1] and self.next_stop >= len(self.stops):
                    self.playing = False                               # end reached and every stop visited
            if self.cb_pov.value:                                      # every tick, so turns finish smoothly
                self.update_pov()                                      # even while paused at a stop
            time.sleep(dt)

    def reach_stop(self, k: int) -> None:
        """Pause at tour stop k and show its Excel status (the last stop is the faulty target)."""
        st = self.stops[k]
        self.next_stop = k + 1
        self.highlight_stop(st)
        if st["kind"] == "goto":                # single destination: status + offer the copilot (button)
            self.md_walk.content = (f"✅ **Arrived at {stop_title(st)}**" + (
                "  \nIts data is below; **Ask what you want to know** for more." if st["has_asset"]
                else "  \n_No asset record: this object is not in the Excel pack._"))
            return
        self.md_stop.content = stop_text(st, k + 1, len(self.stops))
        if st["kind"] != "target":
            self.pause_until = time.time() + float(self.tour["pause_s"])

    # --- routes: inspection tour vs. go to one equipment ---
    def _build_chat_gui(self, g: Any) -> None:
        """Chat: quick commands (instant, free) -> Claude Code headless with our MCP tools."""
        with g.add_folder("Ask the MCP"):
            self.tx_chat = g.add_text("Question", "Which assets need attention?")
            b_chat = g.add_button("Ask", color="violet")
            self.md_chat_status = g.add_markdown("", visible=False)
            self.md_claude = g.add_markdown("_Checking Claude Code…_")
            self.md_chat = g.add_markdown("_Type a question or a command (e.g. `take me to M-006`, `start tour`), "
                                          "then press **Ask**. Type `help` for the quick commands._")
        self._chat_busy = False

        def check() -> None:                          # free login check, in the background
            from copilot.chat import claude_status
            st = claude_status(self.cfg["chat"])
            self.cfg["chat"]["_status"] = st
            self.md_claude.content = st["message"]
        threading.Thread(target=check, daemon=True).start()

        @b_chat.on_click
        def _(_e: Any) -> None:
            q = self.tx_chat.value.strip()
            if q and not self._chat_busy:
                threading.Thread(target=self.chat, args=(q,), daemon=True).start()

    def chat_act(self, cmd: str, target: str) -> str:
        """Viewer actions for quick commands (same handlers as the MCP remote commands)."""
        if cmd == "tour":
            res = self.handle_command({"cmd": "tour", "stops": [t for t in target.split(",") if t]})
        else:
            res = self.handle_command({"cmd": cmd, "target": target})
        return ("" if res.get("ok") else "⚠️ ") + res.get("message", "")

    def chat(self, question: str) -> None:
        from copilot.chat import answer, parse_quick
        self._chat_busy = True
        labels = sorted({n["label"] for n in self.nodes})
        quick = self.cfg["chat"].get("quick_commands", True) and parse_quick(question, labels) is not None
        t0 = time.time()

        def tick() -> None:
            while self._chat_busy:
                self.md_chat_status.content = (f"⏳ **Claude is working** (copilot tools via MCP) · "
                                               f"{time.time() - t0:.0f} s")
                time.sleep(0.5)

        self.md_chat_status.visible = not quick
        if not quick:
            threading.Thread(target=tick, daemon=True).start()
        try:
            text, route = answer(question, self.cfg["chat"], self.chat_act)
        except Exception as e:  # noqa: BLE001 - chat must never break the viewer
            text, route = f"⚠️ {type(e).__name__}: {e}", "error"
        finally:
            self._chat_busy = False
            self.md_chat_status.visible = False
        badge = "⚡ quick command" if route == "quick" else f"🤖 Claude · {time.time() - t0:.0f} s"
        self.md_chat.content = f"> **You:** {question}\n\n{text}\n\n_{badge}_"

    def draw_tour_markers(self) -> None:
        """Numbered floor markers for the current tour stops (replaces any previous markers)."""
        for h in self.h_tour_markers:
            h.remove()
        self.h_tour_markers = []
        p = self.pts + np.array([0, -self.vc["path_lift"], 0])            # up = -y
        sc = self.server.scene
        for k, st in enumerate(self.stops, 1):
            q = p[min(st["sample_index"], len(p) - 1)]
            col = (220, 30, 30) if st["kind"] == "target" else (255, 150, 0)
            self.h_tour_markers.append(sc.add_icosphere(f"/tour/stop_{k}", radius=0.07, color=col, position=tuple(q)))
            self.h_tour_markers.append(sc.add_label(f"/tour/label_{k}", f"{k}. {st['tag']}",
                                                    position=(q[0], q[1] - 0.12, q[2]),
                                                    anchor="bottom-center", font_screen_scale=0.8))

    # --- remote control (MCP server -> out/viewer_cmd.jsonl -> here) ---
    def _init_remote(self) -> None:
        self.cmd_path = out_path(self.cfg, "viewer_cmd.jsonl")
        self.cmd_offset = self.cmd_path.stat().st_size if self.cmd_path.exists() else 0   # ignore old commands
        self.next_poll = 0.0

    def poll_remote(self, now: float) -> None:
        """Heartbeat + execute new commands (show / walk / tour) written by copilot.tools."""
        if now < self.next_poll:
            return
        self.next_poll = now + self.vc.get("remote_poll_s", 0.5)
        out_path(self.cfg, "viewer_heartbeat").write_text(str(now), encoding="utf-8")
        if not self.cmd_path.exists() or self.cmd_path.stat().st_size <= self.cmd_offset:
            return
        with open(self.cmd_path, encoding="utf-8") as f:
            f.seek(self.cmd_offset)
            lines = f.read().splitlines()
            self.cmd_offset = f.tell()
        for line in lines:
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                res = self.handle_command(c)
            except Exception as e:  # noqa: BLE001 - a bad command must never stop the viewer
                res = {"ok": False, "message": f"{type(e).__name__}: {e}"}
            with open(out_path(self.cfg, "viewer_cmd_results.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": c.get("id"), **res}) + "\n")

    def find_node(self, key: str) -> dict | None:
        k = str(key).strip().upper()
        for n in self.nodes:
            a = n.get("asset")
            if n["id"].upper() == k or (a and k in (a["tag"].upper(), a["asset_id"].upper())):
                return n
        return None

    def nearest_by_label(self, query: str, reachable: bool) -> tuple[dict | None, str]:
        """Object whose vision label matches `query` (e.g. 'tank' -> 'tank/vessel'): the reachable one with the
        shortest walk from the worker (reachable=True) or the most confident one (False)."""
        from copilot.tools import label_matches
        cands = sorted((n for n in self.nodes if label_matches(n["label"], query)),
                       key=lambda n: -(n.get("label_confidence") or 0))[: self.vc.get("label_candidates", 10)]
        if not cands:
            return None, f"No asset or detected object called '{query}'."
        if not reachable:
            return cands[0], f"{len(cands)} '{cands[0]['label']}' object(s); showing the most confident."
        if self.planner is None:
            from pipeline.tour import WalkPlanner
            self.planner = WalkPlanner(self.cfg)
        p, best = self.position(), None
        for n in cands:
            plan = self.planner.to_object(float(p[0]), float(p[2]), n)
            if plan["ok"] and (best is None or plan["length"] < best[1]):
                best = (n, plan["length"])
        if best is None:
            return None, f"None of the {len(cands)} '{cands[0]['label']}' objects is reachable from here."
        return best[0], f"nearest reachable of {len(cands)} '{best[0]['label']}' object(s)"

    def handle_command(self, c: dict) -> dict:
        cmd = c.get("cmd")
        if cmd in ("show", "walk"):
            n, why = self.find_node(c.get("target", "")), ""
            if n is None:                                    # not a tag / id: try a vision label ("tank")
                n, why = self.nearest_by_label(c.get("target", ""), reachable=cmd == "walk")
                if n is None:
                    return {"ok": False, "message": why}
            name = n["asset"]["tag"] if n.get("asset") else f"{n['id']} ({n['label']})"
            if cmd == "show":
                self.select(n)
                for client in self.server.get_clients().values():
                    self.fly_to(n, client)
                return {"ok": True, "message": f"Showing {name} in the viewer (selected, camera moved)"
                                               + (f"; {why.rstrip('.')}." if why else ".")}
            self.go_to(n)
            msg = self.md_walk.content.replace("**", "").replace("⚠️ ", "").replace("🚶 ", "")
            if why and self.playing:
                msg = msg.split("  \n")[0] + f" ({why})."
                self.md_walk.content += f"  \n_{why}_"
            return {"ok": self.playing, "message": msg}
        if cmd == "tour":
            from pipeline.tour import plan_tour
            t = plan_tour(self.cfg, list(c.get("stops") or []))
            self.tour, self.tour_route = t, (np.asarray(t["points"], np.float64), t["stops"])
            self.walk_mode = "goto"                       # force set_walk_mode to rebuild the tour route
            self.set_walk_mode("tour")
            self.draw_tour_markers()
            self.playing = True
            route = " → ".join(["door"] + [s["tag"] for s in t["stops"]])
            skipped = "; ".join(f"{s['tag']}: {s['reason']}" for s in t["skipped"]) or "none"
            return {"ok": True, "message": f"Tour started ({t['mode']}): {route}, "
                                           f"{t['length'] * self.cfg['scene']['units_to_m']:.1f} m. Skipped: {skipped}"}
        return {"ok": False, "message": f"Unknown command '{cmd}'."}

    def draw_path(self) -> None:
        p = self.pts + np.array([0, -self.vc["path_lift"], 0])            # up = -y
        self.h_path = self.server.scene.add_line_segments(
            "/path", np.stack([p[:-1], p[1:]], 1).astype(np.float32), tuple(self.vc["path_color"]),
            thickness=4.0, thickness_units="screen")

    def set_route(self, points: Any, stops: list[dict]) -> None:
        """Replace the walked route (and its stops), start from its beginning, redraw the path."""
        self.pts = np.asarray(points, np.float64)
        self.s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(self.pts, axis=0), axis=1))])
        self.stops = stops
        self.stop_s = [float(self.s[min(st["sample_index"], len(self.s) - 1)]) for st in stops]
        self.next_stop, self.pause_until, self.dist = 0, 0.0, 0.0
        self.clear_stop_highlight()
        self.sl_pos.max = float(self.s[-1])
        self.sl_pos.value = 0.0
        self.draw_path()

    def set_walk_mode(self, mode: str) -> None:
        self.playing = False
        if self.arrived:
            self.leave_target()
        self.walk_mode = mode
        for h in self.h_tour_markers:
            h.visible = mode == "tour"
        if mode == "tour":
            self.set_route(*self.tour_route)
            self.md_stop.content = "_Press Play to start the tour._"
            self.md_walk.content = ""
        self.update()

    def go_to(self, node: dict) -> None:
        """Plan from the worker's current position to one object and start walking (Go to mode)."""
        if self.walk_mode != "goto":
            self.set_walk_mode("goto")
        if self.planner is None:
            from pipeline.tour import WalkPlanner
            self.planner = WalkPlanner(self.cfg)
        p = self.position()
        plan = self.planner.to_object(float(p[0]), float(p[2]), node)
        if not plan["ok"]:
            self.md_walk.content = f"⚠️ {plan['message']}"
            return
        self.set_route(plan["points"], [goto_record(node, len(plan["points"]) - 1)])
        dest = node["asset"]["tag"] if node.get("asset") else f"{node['id']} ({node['label']})"
        note = "" if node.get("asset") else "  \n_No asset record: this object is not in the Excel pack._"
        if node.get("asset"):
            self.choose_equipment(node["asset"]["tag"])
        self.md_walk.content = (f"🚶 Walking to **{dest}**: {plan['length'] * self.cfg['scene']['units_to_m']:.1f} m, "
                                f"min clearance {plan['min_clearance']:.2f}{note}")
        self.playing = True

    def highlight_stop(self, st: dict) -> None:
        """Thick coloured frame + translucent fill + big label on the stop's equipment; worker faces it."""
        n = next(x for x in self.nodes if x["id"] == st["object_id"])
        col = self.vc["highlight_colors"].get(st["condition_state"], [255, 0, 0])
        segs, _ = edges_for([n], self.vc["colors"])
        self.h_stop_box.points = segs
        self.h_stop_box.colors = np.tile(np.array(col, np.uint8), (len(segs), 2, 1))
        self.h_stop_box.visible = True
        lo, hi = np.array(n["bbox_min"]), np.array(n["bbox_max"])
        pad = self.vc["stop_fill_pad"]
        self.h_stop_fill = self.server.scene.add_box(
            "/stop/fill", tuple(col), tuple(hi - lo + 2 * pad), position=tuple((lo + hi) / 2),
            opacity=self.vc["stop_fill_opacity"], cast_shadow=False, receive_shadow=False)
        self.h_stop_label = self.server.scene.add_label(
            "/stop/label", stop_title(st),
            position=(float(n["center"][0]), float(lo[1]) - 0.15, float(n["center"][2])),
            anchor="bottom-center", font_screen_scale=self.vc["stop_label_scale"])
        self.stop_active = st
        self.face_target = np.array(n["center"], np.float64)
        self.pose_avatar(self.position(), moving=False)

    def blink_stop(self, now: float) -> None:
        on = int(now * 2 * self.vc["stop_blink_hz"]) % 2 == 0
        self.h_stop_box.visible = on
        if self.h_stop_fill is not None:
            self.h_stop_fill.visible = on

    def clear_stop_highlight(self) -> None:
        self.h_stop_box.points, self.h_stop_box.colors = edges_for([], self.vc["colors"])
        for h in (self.h_stop_fill, self.h_stop_label):
            if h is not None:
                h.remove()
        self.h_stop_fill = self.h_stop_label = None
        self.stop_active = None
        self.face_target = None


if __name__ == "__main__":
    Sim(load_config()).run()
