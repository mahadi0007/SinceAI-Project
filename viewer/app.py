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

from copilot.agent import diagnose, to_markdown
from copilot.voice import briefing
from pipeline.common import ROOT, load_config, out_path, repo_path
from pipeline.load_splat import covariances, load_splat

Cfg = dict[str, Any]
IDLE_TEXT = "_Press Play: the copilot briefs the worker once they stand in front of the target._"
EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


# ---------- data ----------

def load_scene(cfg: Cfg) -> tuple[dict, dict, dict | None]:
    """Scene graph, walk path, and the inspection tour (out/tour.json) if enabled and present.
    With a tour, the walk follows the tour points instead of the direct path."""
    from pipeline.correct import downstream_stale
    for name in ("scene_graph.json", "path.json"):
        why = downstream_stale(out_path(cfg, name).parent, name)
        if why:
            raise SystemExit(f"[viewer] {why}")
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
        from api.annotate import free_port
        self.port = free_port(self.vc["port"])          # another viewer may already hold the configured port
        if self.port != self.vc["port"]:
            print(f"[viewer] port {self.vc['port']} is in use; using {self.port}")
        self.server = viser.ViserServer(port=self.port)
        self.server.scene.set_up_direction(self.cfg["scene"]["up_direction"])
        self._labels: list[Any] = []
        self.av = scaled_avatar(self.vc["avatar"])
        self._build_scene()
        self._build_gui()
        self.h_corr: list[Any] = []                  # boundary-correction preview layers (api/annotate.py)
        self.h_corr_acc: list[Any] = []
        self.annot: Any = None
        try:
            from api.annotate import start_in_thread
            self.annot = start_in_thread(cfg, sink=self)
            self.annot.push_accepted()
            self.md_correct.content = self.correct_link(self.dd_region.value)
        except Exception as e:  # noqa: BLE001  (viewer still works without the tool)
            print(f"[viewer] boundary correction tool unavailable: {e}")
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

    def rebuild_splats(self, dark_threshold: float, crop: bool) -> dict[str, int]:
        """(Re)send the three splat groups with the brightness / crop filter; keep their visibility."""
        sc = self.server.scene
        keep = self.sp_lum >= dark_threshold
        if crop:
            keep &= self.sp_inside
        counts = {}
        for g in ("room", "walls", "ceiling"):
            m = keep & (self.sp_group == g)
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
        segs, cols = edges_for(self.nodes, self.vc["colors"])
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
        lift = np.array([0, -self.vc["path_lift"], 0])                # up = -y
        p = self.pts + lift
        self.draw_path()
        self.h_tour_markers: list[Any] = []
        for k, st in enumerate(self.stops, 1):              # numbered tour stop markers on the floor
            q = p[min(st["sample_index"], len(p) - 1)]
            col = (220, 30, 30) if st["kind"] == "target" else (255, 150, 0)
            self.h_tour_markers.append(sc.add_icosphere(f"/tour/stop_{k}", radius=0.07, color=col, position=tuple(q)))
            self.h_tour_markers.append(sc.add_label(f"/tour/label_{k}", f"{k}. {st['tag']}",
                                                    position=(q[0], q[1] - 0.12, q[2]),
                                                    anchor="bottom-center", font_screen_scale=0.8))
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
        self.h_all_labels = [sc.add_label(f"/all_labels/{n['id']}", label_text(n),
                                          position=(n["center"][0], n["bbox_min"][1], n["center"][2]),
                                          anchor="bottom-center",
                                          font_screen_scale=(self.vc["equipment_label_scale"] if eq
                                                             else self.vc["all_label_scale"]),
                                          visible=label_visible(mode, eq))
                             for n, eq in zip(self.nodes, self.is_equipment)]
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
        g = self.server.gui
        c = self.sg["counts"]
        g.add_markdown(f"## Valmet Field Copilot\n**{c['objects']} assets extracted** · "
                       f"{c['structures']} structural planes · **{c['mapped']} mapped** to Excel\n\n"
                       f"Target: **{self.sg['target']['tag']}** · path {self.sg['path']['length']:.2f} u")
        b_play = g.add_button("Play / Pause", icon=None)
        b_reset = g.add_button("Reset")
        self.sl_speed = g.add_slider("Speed (u/s)", 0.1, 2.0, 0.1, self.vc["walk_speed"])
        self.sl_radius = g.add_slider("Proximity radius", 0.5, 3.0, 0.1, self.vc["proximity_radius"])
        self.sl_pos = g.add_slider("Position", 0.0, float(self.s[-1]), 0.01, 0.0)
        self._build_walk_mode_gui(g)
        self.cb_boxes = g.add_checkbox("Show all frames", True)
        dc = self.vc["display"]
        with g.add_folder("Scene display", expand_by_default=False):
            cb_ceiling = g.add_checkbox("Show ceiling", dc["show_ceiling"])
            cb_walls = g.add_checkbox("Show walls", dc["show_walls"])
            cb_crop = g.add_checkbox("Crop outside walls", dc["crop_outside_walls"])
            sl_dark = g.add_slider("Hide dark splats below", 0.0, 0.2, 0.01, dc["dark_threshold"])
            b_apply = g.add_button("Apply filter")
            md_disp = g.add_markdown("")

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
        self.dd_labels = g.add_dropdown("Labels", list(LABEL_MODES), initial_value=self.vc["label_mode"])
        with g.add_folder("Boundary repair: before / after"):
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
            self.md_correct = g.add_markdown(self.correct_link("-"))
            self.cb_corr = g.add_checkbox("Show correction preview", True)
            g.add_markdown("<small>Preview: green = selected (2+ views), orange = one view only, red = disputed "
                           "(excluded), yellow = unseen, left uncertain, blue = unseen, inferred with whole objects. "
                           "Magenta frames = accepted, cyan = applied.</small>")

        @self.cb_corr.on_update
        def _(_e: Any) -> None:
            for h in self.h_corr + self.h_corr_acc:
                h.visible = self.cb_corr.value

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
            self.md_correct.content = self.correct_link(rid)
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
        with g.add_folder("Selected object (click in the scene)"):
            self.md_sel = g.add_markdown("_Click any object in the 3D view to inspect it._")
            self.img_sel = g.add_image(np.full((8, 24, 3), 235, np.uint8), visible=False)
            b_fly = g.add_button("Fly to object", visible=True)
            self.b_identify = g.add_button("Ask vision model what this is (paid)")
            self.md_identify = g.add_markdown("")
        with g.add_folder("Inspection tour"):
            if self.tour:
                route = " → ".join(["door"] + [s["tag"] for s in self.stops])
                mode = "from Excel status" if self.tour["mode"] == "auto_excel" else "manual list"
                skipped = ", ".join(f"{s['tag']}" for s in self.tour["skipped"]) or "none"
                g.add_markdown(f"Tour {mode}: {route}  \n{self.tour['length']:.1f} u "
                               f"(~{self.tour['length'] * self.cfg['scene']['units_to_m']:.0f} m) · "
                               f"skipped (unreachable/limit): {skipped}")
            self.md_stop = g.add_markdown("_Press Play to start the tour._" if self.tour
                                          else "_No tour: run `python -m pipeline.tour`._")
        with g.add_folder("Nearby assets"):
            self.md_near = g.add_markdown("")

        @self.dd_labels.on_update
        def _(_e: Any) -> None:
            mode = self.dd_labels.value
            for h, eq in zip(self.h_all_labels, self.is_equipment):
                h.visible = label_visible(mode, eq)
            self.update()

        @b_fly.on_click
        def _(e: Any) -> None:
            if self.selected is not None:
                self.fly_to(self.selected, e.client)

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
        with g.add_folder("Copilot"):
            tags = sorted(n["asset"]["tag"] for n in self.nodes if n["asset"])
            self.dd_asset = g.add_dropdown("Asset", tags, initial_value=self.target["asset"]["tag"])
            self.tx_question = g.add_text("Question", "Why is this asset flagged and what should I check first?")
            self.b_ask = g.add_button("Ask copilot", color="blue")
            self.md_loading = g.add_markdown("", visible=False)
            self.pb = g.add_progress_bar(0.0, animated=True, visible=False, color="blue")
            self.h_audio = g.add_html("", visible=False)          # spoken briefing player
            self.md_copilot = g.add_markdown(IDLE_TEXT)

        @b_play.on_click
        def _(_e: Any) -> None:
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

        @self.b_ask.on_click
        def _(_e: Any) -> None:
            tag, q = self.dd_asset.value, self.tx_question.value
            self.run_job(tag, q, lambda d: setattr(self.md_copilot, "content", to_markdown(d)))

        @b_reset.on_click
        def _(_e: Any) -> None:
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

        @self.cb_boxes.on_update
        def _(_e: Any) -> None:
            self.h_boxes.visible = self.cb_boxes.value

        @self.sl_radius.on_update
        def _(_e: Any) -> None:
            r = float(self.sl_radius.value)
            self.server.scene.add_icosphere("/avatar/ring", radius=1.0,
                                            color=tuple(self.vc["avatar"]["ring_color"]),
                                            opacity=0.15, scale=(r, 0.01, r))
            self.update()

    def correct_link(self, rid: str) -> str:
        if getattr(self, "annot", None) is None:
            return "_Boundary correction tool not running yet._"
        port = self.annot.port
        q = f"?region={rid}" if rid != "-" else ""
        return (f"**Correct boundaries:** [open the correction tool](http://localhost:{port}/{q}) "
                f"(draw a box, include/exclude, SAM mask, second view, accept/undo; manual prompts only).")

    # --- boundary-correction preview sink (called from the annotation server thread) ---
    def clear_correction(self) -> None:
        for h in self.h_corr:
            h.remove()
        self.h_corr = []

    def show_correction(self, cid: str, layers: dict, colors: dict, bbox: Any, cam: Any) -> None:
        self.clear_correction()
        sc = self.server.scene
        vis = self.cb_corr.value
        for name, P in layers.items():
            if len(P):
                self.h_corr.append(sc.add_point_cloud(f"/correction/{name}", np.asarray(P, np.float32),
                                                      colors[name], point_size=0.014, visible=vis))
        if bbox:
            self.h_corr.append(sc.add_line_segments("/correction/box", box_edges(*bbox), (255, 255, 255),
                                                    thickness=3.0, thickness_units="screen", visible=vis))
            self.h_corr.append(sc.add_label("/correction/label", f"{cid} (preview)",
                                            position=((bbox[0][0] + bbox[1][0]) / 2, bbox[0][1] - 0.05,
                                                      (bbox[0][2] + bbox[1][2]) / 2), anchor="bottom-center"))
        if cam is not None and bbox:
            c = (np.array(bbox[0]) + np.array(bbox[1])) / 2
            for cl in self.server.get_clients().values():
                cl.camera.position = tuple(cam.C)
                cl.camera.look_at = tuple(c)

    def show_accepted(self, items: list[dict]) -> None:
        for h in self.h_corr_acc:
            h.remove()
        self.h_corr_acc = []
        sc = self.server.scene
        for it in items:
            col = (0, 220, 255) if it["status"] == "applied" else (255, 0, 200)
            bb = it["bbox"]
            self.h_corr_acc.append(sc.add_line_segments(f"/corrections/{it['cid']}", box_edges(*bb), col,
                                                        thickness=2.5, thickness_units="screen"))
            self.h_corr_acc.append(sc.add_label(f"/corrections/{it['cid']}_label",
                                                f"{it['cid']} {it['cls']} · {it['status']} · {it['evidence']}",
                                                position=((bb[0][0] + bb[1][0]) / 2, bb[0][1] - 0.05,
                                                          (bb[0][2] + bb[1][2]) / 2), anchor="bottom-center",
                                                font_screen_scale=0.7))

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
            at_target = self.dist >= self.s[-1] - 1e-6 and self.box_gap(self.target, p) <= self.arrive_gap
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
        if not hit.any():
            return None
        k = np.where(hit)[0][np.argmin(t[hit])]
        return objs[int(self.pick_lab[k])]

    def on_scene_click(self, e: Any) -> None:
        n = self.pick(np.array(e.ray_origin), np.array(e.ray_direction))
        if n is not None:
            self.select(n)

    def select(self, n: dict) -> None:
        self.selected = n
        segs, _ = edges_for([n], self.vc["colors"])
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
        if n["asset"]:
            self.dd_asset.value = n["asset"]["tag"]

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
        self.b_ask.label = "Copilot is working…" if on else "Ask copilot"

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
                                               "**Ask copilot** below._")
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
        self.md_copilot.content = self.arrival_header() + to_markdown(self._diag)
        if self._audio:
            b64 = base64.b64encode(self._audio).decode()
            self.h_audio.content = (
                '<div style="font-weight:600;margin:4px 0">🔊 Copilot briefing</div>'
                f'<audio controls autoplay style="width:100%" src="data:audio/mpeg;base64,{b64}"></audio>')
            self.h_audio.visible = True

    def on_arrival(self) -> None:
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
            self.md_copilot.content = IDLE_TEXT

    def run(self) -> None:
        self.update()
        print(f"[viewer] http://localhost:{self.port}  ({len(self.nodes)} objects, "
              f"path {self.s[-1]:.2f} u)")
        dt = 1.0 / self.vc["fps"]
        while True:
            now = time.time()
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
                if self.dist >= self.s[-1]:
                    self.playing = False
            time.sleep(dt)

    def reach_stop(self, k: int) -> None:
        """Pause at tour stop k and show its Excel status (the last stop is the faulty target)."""
        st = self.stops[k]
        self.next_stop = k + 1
        self.highlight_stop(st)
        if st["kind"] == "goto":                # single destination: status + offer the copilot (button)
            self.md_goto.content = (f"✅ **Arrived at {stop_title(st)}**" + (
                "  \nPress **Ask copilot** (Copilot panel) for a grounded diagnosis of this asset."
                if st["has_asset"] else "  \n_No asset record: this object is not in the Excel pack._"))
            if st["has_asset"]:
                self.dd_asset.value = st["tag"]
            return
        self.md_stop.content = stop_text(st, k + 1, len(self.stops))
        if st["kind"] != "target":
            self.pause_until = time.time() + float(self.tour["pause_s"])

    # --- routes: inspection tour vs. go to one equipment ---
    def _build_walk_mode_gui(self, g: Any) -> None:
        """Two clear mode buttons near the top: Inspection tour | Go to equipment."""
        with g.add_folder("Walk mode"):
            self.md_mode = g.add_markdown("**Mode: Inspection tour** (door → warning assets → faulty P-002)")
            b_tour = g.add_button("Inspection tour", color="blue")
            b_goto = g.add_button("Go to equipment", color="teal")
            goto_opts = [f"{n['asset']['tag']} · {n['label']} · {n['asset']['condition_state']}"
                         for n in sorted((n for n in self.nodes if n["asset"]), key=lambda n: n["asset"]["tag"])]
            self.dd_goto = g.add_dropdown("Equipment", goto_opts, initial_value=goto_opts[0], visible=False)
            b_goto_tag = g.add_button("Walk to selected equipment", visible=False)
            b_goto_click = g.add_button("Walk to clicked object", visible=False)
            self.md_goto = g.add_markdown("", visible=False)
        self.goto_widgets = (self.dd_goto, b_goto_tag, b_goto_click, self.md_goto)

        @b_tour.on_click
        def _(_e: Any) -> None:
            if self.walk_mode != "tour":
                self.set_walk_mode("tour")

        @b_goto.on_click
        def _(_e: Any) -> None:
            if self.walk_mode != "goto":
                self.set_walk_mode("goto")

        @b_goto_tag.on_click
        def _(_e: Any) -> None:
            tag = self.dd_goto.value.split(" · ")[0]
            self.go_to(next(n for n in self.nodes if n["asset"] and n["asset"]["tag"] == tag))

        @b_goto_click.on_click
        def _(_e: Any) -> None:
            if self.selected is None:
                self.md_goto.content = "⚠️ Click an object in the 3D view first, then press this button."
                return
            self.go_to(self.selected)

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
        for w in self.goto_widgets:
            w.visible = mode == "goto"
        self.md_mode.content = ("**Mode: Inspection tour** (door → warning assets → faulty P-002)" if mode == "tour"
                                else "**Mode: Go to equipment** (one asset, from where the worker stands)")
        if mode == "tour":
            self.set_route(*self.tour_route)
            self.md_stop.content = "_Press Play to start the tour._"
        else:
            self.md_goto.content = ("Pick equipment and press **Walk to selected equipment**, or click any object "
                                    "in the 3D view and press **Walk to clicked object**. The worker walks from "
                                    "where it stands now.")
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
            self.md_goto.content = f"⚠️ {plan['message']}"
            return
        self.set_route(plan["points"], [goto_record(node, len(plan["points"]) - 1)])
        dest = node["asset"]["tag"] if node.get("asset") else f"{node['id']} ({node['label']})"
        note = "" if node.get("asset") else "  \n_No asset record: this object is not in the Excel pack._"
        self.md_goto.content = (f"🚶 Walking to **{dest}**: {plan['length'] * self.cfg['scene']['units_to_m']:.1f} m, "
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
