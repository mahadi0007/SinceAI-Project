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
from pipeline.bim_provenance import ArtifactMismatch, compare, scene_revision
from pipeline.common import ROOT, load_config, out_path, repo_path
from pipeline.load_splat import covariances, load_splat
from viewer.nudge import add_nudge_gui, gizmo_scale, up_vector
from viewer import scene_edits as se
from viewer import ui

Cfg = dict[str, Any]
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


def audio_seconds(audio: bytes, cfg: Cfg) -> float:
    """Length of a constant-bitrate ElevenLabs mp3, from voice.output_format (e.g. mp3_44100_128 = 128 kbit/s)."""
    try:
        kbps = int(str(cfg["viewer"]["voice"]["output_format"]).rsplit("_", 1)[-1])
    except (KeyError, ValueError):
        kbps = 128
    return len(audio) * 8 / (kbps * 1000)


def status_of(node: dict) -> str:
    return node["asset"]["condition_state"] if node["asset"] else "no_record"


def visit_style(node: dict) -> tuple[tuple[int, int, int], str]:
    """Vivid binary health colors; unknown equipment stays neutral."""
    asset = node.get("asset") or {}
    state = str(asset.get("condition_state", "")).strip().lower()
    name = asset.get("tag") or node["id"]
    if state == "healthy":
        return (0, 255, 40), f"{name} · HEALTHY"
    if state and state not in ("no_record", "unknown", "unassigned"):
        return (255, 0, 20), f"{name} · {state.replace('_', ' ').upper()}"
    return (170, 180, 190), f"{name} · STATUS UNKNOWN"


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


LABEL_MODES = ("Needs attention", "Equipment", "All objects", "Nearby only", "Off")


def label_visible(mode: str, is_equipment: bool, state: str = "") -> bool:
    """Persistent label visibility: Needs attention = Excel assets that are not healthy (the default, keeps
    the scan readable); Equipment = mapped or non-pipe objects; All = everything."""
    if mode == "Needs attention":
        return state not in ("", "healthy", "no_record")
    return mode == "All objects" or (mode == "Equipment" and is_equipment)


def label_text(n: dict, full: bool = False) -> str:
    """'P-002 · pump · attention required' (short, default) or the full
    'OBJ-0018 · P-002 pump · attention_required · health 45' / 'OBJ-0142 · ladder/stairs 0.90'."""
    if n["asset"]:
        a = n["asset"]
        if not full:
            return f"{a['tag']} · {n['label']} · {a['condition_state'].replace('_', ' ')}"
        return f"{n['id']} · {a['tag']} {n['label']} · {a['condition_state']} · health {a['health_score']}"
    conf = n.get("label_confidence")
    return f"{n['id']} · {n['label']}" + (f" {conf:.2f}" if conf is not None and full else "")


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


class HtmlText:
    """A markdown-like status line rendered as styled HTML (viewer/ui.py); keeps the `.content` string API of
    the markdown handle it replaces, so existing code (and the MCP command replies) read the same text."""

    def __init__(self, handle: Any, render: Any) -> None:
        self.h, self.render, self._text = handle, render, ""

    @property
    def content(self) -> str:
        return self._text

    @content.setter
    def content(self, text: str) -> None:
        self._text = text
        self.h.content = self.render(text)


# ---------- viewer ----------

def start_facing(cfg: Cfg, door: np.ndarray) -> np.ndarray:
    """Initial worker direction at the door (x, 0, z): straight into the room, away from the door wall,
    unless viewer.start_facing gives [dx, dz]."""
    v = cfg["viewer"].get("start_facing", "auto")
    if isinstance(v, (list, tuple)):
        d = np.array([float(v[0]), 0.0, float(v[1])])
    else:
        wall_z = float(cfg["scene"].get("back_wall_z", door[2] - 1.0))
        d = np.array([0.0, 0.0, 1.0 if door[2] >= wall_z else -1.0])
    return d / (np.linalg.norm(d) + 1e-9)


class Sim:
    def __init__(self, cfg: Cfg) -> None:
        self.cfg, self.vc = cfg, cfg["viewer"]
        self.sg, path, self.tour = load_scene(cfg)
        self.nodes = [n for n in self.sg["objects"] if n["kind"] == "object"]
        self.pts = np.array(path["points"], np.float64)
        seg = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        self.s = np.concatenate([[0], np.cumsum(seg)])
        self.door_p = self.pts[0].copy()                # the worker starts (and resets) here
        self.start_dir = start_facing(cfg, self.door_p)
        # distance along the walk at which each tour stop is reached (pauses happen there)
        self.stops = self.tour["stops"] if self.tour else []
        self.stop_s = [float(self.s[min(st["sample_index"], len(self.s) - 1)]) for st in self.stops]
        self.next_stop = 0
        self.pause_until = self.pause_left = 0.0
        self.face_target: np.ndarray | None = None   # avatar faces this point while stopped
        self._saved_cams: dict[int, tuple] = {}      # each browser's own camera, restored when POV is turned off
        self._pov_look: np.ndarray | None = None     # smoothed POV look-at point
        self.stop_active: dict | None = None         # tour stop currently highlighted
        self.visit_active: dict | None = None        # visual lifetime is independent of stop/briefing timers
        self.visit_arrival_gap = 0.0
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
        self.revision = scene_revision(cfg, self.pick_lab)     # every derived artifact must match this
        self.selected: dict | None = None
        self.node_by_id = {n["id"]: n for n in self.nodes}
        self.obj_index = {o["id"]: k for k, o in enumerate(self.sg["objects"])}
        self.movable: dict[str, dict] = {}           # objects in removable_rows.npz: own splat, can be edited
        self.edit_note = ""                          # why editing is unavailable (shown in More)
        self.movable_members: set[str] = set()       # editable objects + parts that move with them
        self.inst: dict[str, dict] = {}              # shown instances (originals + copies) by name
        self.sel_inst: str | None = None
        self._picked_inst: str | None = None
        self.show_movable = False
        self.gizmo: Any = None
        self.edits: dict | None = None               # shared layout edits (viewer/scene_edits.py)
        self._gizmo_dirty = False
        self.tour0, self.path0 = self.tour, path     # unedited scene navigation (out/ files)
        self._bounds0 = {n["id"]: {k: n[k] for k in ("bbox_min", "bbox_max", "center", "extent")} for n in self.nodes}
        self.nav_inputs: tuple | None = None         # layout navigation (pipeline.layout_nav), None = out/ files
        self._nav_layout = ""                        # layout hash the current navigation was built for
        self.display = "Scan"                        # Scan | BIM | Overlay
        self.bim: Any = None                         # viewer.bim_layer.BimLayer once loaded and validated
        self.bim_note = "_BIM model loading…_"
        self.h_bim: dict[str, Any] = {}              # static IFC meshes by object id
        self._bim_tinted: str | None = None
        self._group_vis = {g: cfg["viewer"]["display"].get(f"show_{g}", True) for g in ("room", "walls", "ceiling")}
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
        self._restore_layout_nav()
        threading.Thread(target=self._load_bim, daemon=True).start()
        self.set_start_camera()                         # every new browser / refresh opens at the door ...
        if self.cb_pov.value:                           # ... in the worker's view
            self.set_pov(True)
        self._reset_pending = False
        if self.vc.get("reset_on_connect", True):       # refresh = start state (applied in the run loop)
            self.server.on_client_connect(lambda _c: setattr(self, "_reset_pending", True))
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
        self.sp_movable = np.zeros(len(x), bool)
        if self.vc["render_splats"]:
            idx = self.obj_index
            rows, parts = self.movable_rows(d["row_ids"])
            for oid, r in rows.items():
                self.sp_movable[r] = True
                n = self.node_by_id[oid]
                c = np.array(n["center"])
                members = [oid, *parts.get(oid, [])]                   # attached parts move with it
                pts = self.pick_xyz[np.isin(self.pick_lab, [idx[m] for m in members])]
                lo, hi = (np.array(n["bbox_min"]), np.array(n["bbox_max"])) if len(members) == 1 else \
                    (pts.min(0), pts.max(0))
                self.movable[oid] = {"c": c, "xyz": x[r] - c, "rgb": d["rgb"][r], "op": self.sp["opacity"][r],
                                     "cov": self.sp["cov"][r], "pick": pts - c, "members": members,
                                     "lo": lo - c, "hi": hi - c, "box": box_edges(lo - c, hi - c),
                                     "bim": None}
        self.movable_members = {m for v in self.movable.values() for m in v["members"]}
        self._load_edit_state()

    def movable_rows(self, row_ids: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
        """Editable objects' splat rows, only when removable.json was built from this scene version."""
        p, j = out_path(self.cfg, "removable_rows.npz"), out_path(self.cfg, "removable.json")
        if not p.exists() or not j.exists():
            self.edit_note = "No editable-object data for this scene (run `python -m viewer.removable`)."
            return {}, {}
        meta = json.loads(j.read_text(encoding="utf-8"))
        errs = compare(meta.get("revision"), self.revision, "removable.json")
        z = np.load(p)
        missing = [oid for oid in meta.get("removable", []) if oid not in z.files or oid not in self.node_by_id]
        if errs or missing or set(z.files) != set(meta.get("removable", [])):
            why = "; ".join(errs) or f"objects {missing} missing / removable_rows.npz does not match removable.json"
            self.edit_note = f"Edit scene disabled: {why}. Regenerate with `python -m viewer.removable`."
            print(f"[viewer] REJECTED editable-object data: {why}")
            return {}, {}
        return ({oid: np.searchsorted(row_ids, z[oid]) for oid in z.files},
                {k: [m for m in v if m in self.node_by_id] for k, v in meta.get("parts", {}).items()})

    def _load_edit_state(self) -> None:
        """Shared layout edits for this scene version (migrating older files); a mismatch disables editing."""
        if not self.movable:
            return
        pivots = {oid: m["c"] for oid, m in self.movable.items()}
        try:
            self.edits, notes = se.load(self.cfg, self.revision, pivots)
        except ArtifactMismatch as e:
            self.edit_note = f"Edit scene disabled: {e}. Move {se.FILE} aside to start a new edit history."
            print(f"[viewer] REJECTED {se.FILE}: {e}")
            self.sp_movable[:] = False
            self.movable = {}
            self.movable_members = set()
            return
        for msg in notes:
            print(f"[viewer] {msg}")
        self.base_layout = se.layout_hash(se.originals(pivots), pivots)
        self._nav_layout = self.base_layout

    def rebuild_splats(self, dark_threshold: float, crop: bool) -> dict[str, int]:
        """(Re)send the three splat groups with the brightness / crop filter; keep their visibility."""
        sc = self.server.scene
        keep = self.sp_lum >= dark_threshold
        if crop:
            keep &= self.sp_inside
        counts = {}
        self._splat_group_rows = {}
        for g in ("room", "walls", "ceiling"):
            m = keep & (self.sp_group == g) & ~self.sp_movable
            self._splat_group_rows[g] = np.flatnonzero(m)
            vis = self._group_vis[g] and self.display != "BIM"
            if self.vc["render_splats"]:
                h = sc.add_gaussian_splats(f"/splat/{g}", centers=self.sp["xyz"][m], rgbs=self.sp["rgb"][m],
                                           opacities=self.sp["opacity"][m], covariances=self.sp["cov"][m])
            else:
                h = sc.add_point_cloud(f"/splat/{g}", self.sp["xyz"][m],
                                       (self.sp["rgb"][m] * 255).astype(np.uint8), point_size=self.vc["point_size"])
            h.visible = vis
            self.h_splat[g] = h
            counts[g] = int(m.sum())
        self._refresh_visit_tint()
        return counts

    def _set_visit_tint(self, node: dict | None) -> None:
        """Tint the equipment itself; no transparent mesh can occlude the splats."""
        previous = getattr(self, "_visit_tint_id", None)
        if previous in getattr(self, "inst", {}):
            it = self.inst[previous]
            it.get("splat", it["h"]).rgbs = self.movable[it["oid"]]["rgb"]
        self._visit_tint_id = node["id"] if node else None
        self._refresh_visit_tint()

    @staticmethod
    def pale_of(n: dict) -> tuple[int, int, int]:
        color, _ = visit_style(n)
        return (185, 255, 200) if color[1] > color[0] else ((255, 190, 190) if color[0] > color[1] else (215, 220, 225))

    def lookup(self, oid: str | None) -> dict | None:
        """Scene object by id, or the identity-free node of a copy instance."""
        if oid is None:
            return None
        by_id = getattr(self, "node_by_id", None) or {n["id"]: n for n in self.nodes}
        if oid in by_id:
            return by_id[oid]
        if oid in getattr(self, "inst", {}):
            return self.view_node(by_id[self.inst[oid]["oid"]], oid)
        return None

    def _refresh_visit_tint(self) -> None:
        if not hasattr(self, "sp"):
            return
        oid = getattr(self, "_visit_tint_id", None)
        n = self.lookup(oid)
        amount = float(np.clip(self.vc.get("visit_tint_strength", .1), 0, 1))
        target = None
        label = -2
        if n is not None:
            target = np.array(self.pale_of(n), dtype=np.float32) / 255.0
            if not hasattr(self, "_splat_object_labels"):
                distance, nearest = cKDTree(self.pick_xyz).query(self.sp["xyz"])
                cutoff = float(self.cfg["label"]["member_dist"])
                self._splat_object_labels = np.where(distance <= cutoff, self.pick_lab[nearest], -1)
            index = getattr(self, "obj_index", None) or {o["id"]: k for k, o in enumerate(self.sg["objects"])}
            label = index.get(n.get("copy_of", n["id"]), -2)
        self._tint_bim(n, amount)
        for group, rows in getattr(self, "_splat_group_rows", {}).items():
            rgb = self.sp["rgb"][rows].copy()
            if target is not None:
                selected = self._splat_object_labels[rows] == label
                rgb[selected] = (1 - amount) * rgb[selected] + amount * target
            handle = self.h_splat[group]
            if self.vc["render_splats"]:
                handle.rgbs = rgb
            else:
                handle.colors = (rgb * 255).astype(np.uint8)
        if target is not None and oid in getattr(self, "inst", {}):
            it = self.inst[oid]
            rgb = self.movable[it["oid"]]["rgb"]
            it.get("splat", it["h"]).rgbs = (1 - amount) * rgb + amount * target

    def _bim_handle(self, oid: str | None) -> Any:
        if oid is None:
            return None
        if oid in getattr(self, "inst", {}):
            return self.inst[oid].get("bim")
        return getattr(self, "h_bim", {}).get(oid)

    def _bim_color(self, oid: str) -> tuple[int, int, int]:
        src = self.inst[oid]["oid"] if oid in getattr(self, "inst", {}) else oid
        bim = getattr(self, "bim", None)
        return bim.mesh[src]["color"] if bim is not None and src in bim.mesh else (150, 150, 150)

    def _tint_bim(self, n: dict | None, amount: float) -> None:
        """BIM class colours stay underneath the health feedback: the highlighted object's mesh gets the same
        light red/green tint as its splat (the strong outline is drawn on top in every display mode)."""
        prev = getattr(self, "_bim_tinted", None)
        h = self._bim_handle(prev)
        if h is not None:
            h.color = self._bim_color(prev)
        self._bim_tinted = None
        if n is None:
            return
        h = self._bim_handle(n["id"])
        if h is not None:
            base = np.array(self._bim_color(n["id"]), float)
            h.color = tuple(int(v) for v in np.round((1 - amount) * base + amount * np.array(self.pale_of(n))))
            self._bim_tinted = n["id"]

    # --- scene ---
    def _build_scene(self) -> None:
        sc = self.server.scene
        self._load_splat_groups()
        self.rebuild_splats(self.vc["display"]["dark_threshold"], self.vc["display"]["crop_outside_walls"])
        segs, cols = edges_for([n for n in self.nodes if n["id"] not in self.movable_members], self.vc["colors"])
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
        self.h_stop_box = sc.add_line_segments(
            "/stop/box", *edges_for([], self.vc["colors"]),
            thickness=self.vc.get("stop_box_thickness", 6.0), thickness_units="screen")
        self.h_stop_fill: Any = None
        self.h_stop_label: Any = None
        self.h_selected = sc.add_line_segments("/selected", *edges_for([], self.vc["colors"]),
                                               thickness=self.vc["selected_thickness"], thickness_units="screen")
        # one persistent label per object; which ones are visible depends on the Labels menu
        self.is_equipment = [bool(n["asset"]) or n["label"] not in self.vc["near_label_skip"]
                             for n in self.nodes]
        mode = self.vc["label_mode"]
        for oid in self.movable:
            self.add_instance(oid, oid, removed=False)
        self.h_all_labels = [sc.add_label(f"/movable/{n['id']}/label" if n["id"] in self.movable else
                                          f"/all_labels/{n['id']}", label_text(n),
                                          position=((0.0, n["bbox_min"][1] - n["center"][1], 0.0) if n["id"] in self.movable
                                                    else (n["center"][0], n["bbox_min"][1], n["center"][2])),
                                          anchor="bottom-center",
                                          font_screen_scale=(self.vc["equipment_label_scale"] if eq
                                                             else self.vc["all_label_scale"]),
                                          visible=label_visible(mode, eq, status_of(n)))
                             for n, eq in zip(self.nodes, self.is_equipment)]
        self.sync_instances()
        self._build_avatar()
        sc.add_icosphere("/avatar/ring", radius=1.0, color=tuple(self.vc["avatar"]["ring_color"]),
                         opacity=0.15, visible=self.vc["avatar"].get("show_ring", False),   # proximity radius disc
                         scale=(self.vc["proximity_radius"], 0.01, self.vc["proximity_radius"]))

    def _build_avatar(self) -> None:
        """Articulated worker. Local frame: forward +z, left/right x, up = -y (native).
        Limbs hang from pivot frames (hips, shoulders) that swing about local x."""
        sc, av = self.server.scene, self.av
        self.h_avatar = sc.add_frame("/avatar", show_axes=False)
        self.h_body = sc.add_frame("/avatar/body", show_axes=False)       # bobs while walking
        self.h_limbs: dict[str, Any] = {}
        self.rig = self._load_rigged_avatar()
        if self.rig is not None:                                          # Mixamo worker replaces the boxes
            return
        col = {k: tuple(v) for k, v in av["colors"].items()}
        sc.add_box("/avatar/body/torso", col["vest"], tuple(av["torso"]),
                   position=(0, -av["torso_h"], 0))
        sc.add_box("/avatar/body/belt", col["trousers"], (av["torso"][0] * 1.02, 0.06, av["torso"][2] * 1.05),
                   position=(0, -(av["hip_h"] + 0.03), 0))
        sc.add_icosphere("/avatar/body/head", radius=av["head_r"], color=col["skin"],
                         position=(0, -av["head_h"], 0))
        sc.add_icosphere("/avatar/body/helmet", radius=av["head_r"] * 1.15, color=col["helmet"],
                         position=(0, -(av["head_h"] + av["head_r"] * 0.35), 0), scale=(1, 0.6, 1))
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

    def _load_rigged_avatar(self) -> Any:
        """Mixamo character from data/avatar/*.glb (see README); None -> the box figure is used."""
        rc = self.av.get("rigged", {})
        if not rc.get("enabled", True) or not repo_path(rc.get("walk_glb", "")).exists():
            if rc.get("enabled", True):
                print(f"[viewer] no {rc.get('walk_glb')} (see README 'Worker avatar'); using the box figure")
            return None
        try:
            from viewer.avatar import RiggedAvatar
            rc = {**rc, "walk_glb": repo_path(rc["walk_glb"]),
                  "idle_glb": repo_path(rc["idle_glb"]) if rc.get("idle_glb") else None}
            up_sign = -1.0 if self.cfg["scene"]["up_direction"].startswith("-") else 1.0
            rig = RiggedAvatar(self.server, "/avatar/body", rc, self.cfg["scene"]["units_to_m"], up_sign)
            print(f"[viewer] rigged worker avatar: {len(rig.meshes)} parts, {rc['height_m']} m")
            return rig
        except Exception as e:  # noqa: BLE001 - a broken model file must not stop the viewer
            print(f"[viewer] rigged avatar failed ({type(e).__name__}: {e}); using the box figure")
            return None

    def pose_avatar(self, p: np.ndarray, moving: bool) -> None:
        av = self.av
        if self.face_target is not None and not moving:                 # face equipment only while standing
            tangent = self.face_target - p
        elif not moving and not self.arrived and np.linalg.norm(p - self.door_p) < 1e-3:
            tangent = self.start_dir                                    # at the door: face into the room
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

        if self.rig is not None:                # the rig's own walk/idle animation (run loop) moves the limbs
            return
        self.h_body.position = (0.0, bob, 0.0)
        self.h_limbs["hip_l"].wxyz, self.h_limbs["hip_r"].wxyz = rot_x(swing), rot_x(-swing)
        self.h_limbs["shoulder_l"].wxyz, self.h_limbs["shoulder_r"].wxyz = rot_x(arm), rot_x(-arm)

    def tangent(self) -> np.ndarray:
        i = int(np.clip(np.searchsorted(self.s, self.dist, side="right") - 1, 0, len(self.pts) - 2))
        t = self.pts[i + 1] - self.pts[i]
        return t / (np.linalg.norm(t) + 1e-9)

    # --- gui ---
    def _build_gui(self) -> None:
        """Tabs follow the demo video: Scene (extraction, health colours, occupancy grid; Advanced settings),
        Walk (go to equipment, inspection tour), Info (asset card + copilot diagnosis), Edit (move/remove
        objects, the route re-plans), Ask (MCP chat). Look: viewer/ui.py."""
        g = self.server.gui
        g.configure_theme(control_width="large", dark_mode=True, brand_color=ui.ACCENT,
                          show_logo=False, show_share_button=False)
        g.add_html(ui.THEME_CSS)
        g.add_html(ui.EXTRA_CSS)
        g.add_html(ui.brand_bar())
        # Viser has no API to change the active tab; updating this (invisible) snippet makes every open
        # browser click the named tab. Outside the tab group, so it is always mounted.
        self.h_tabjs = g.add_html("")
        self._tab_seq = 0
        self._equipment_options()                          # shared by the Walk and Info pickers
        tabs = g.add_tab_group()
        with tabs.add_tab("Scene", icon="box"):
            self._build_scene_tab(g)
        with tabs.add_tab("Walk", icon="walk"):
            self._build_walk_tab(g)
        with tabs.add_tab("Info", icon="info-circle"):
            self._build_equipment_tab(g)
        if self.movable:
            with tabs.add_tab("Edit", icon="arrows-move"):
                self._build_edit_tab(g)
        with tabs.add_tab("Ask", icon="messages"):
            self._build_chat_gui(g)
        self._bind_settings()

    # Info tab (equipment information): the main flow
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
        self.dd_equip = g.add_dropdown("Equipment", list(self.equip_node),
                                       initial_value=self.equip_opt[self.target["asset"]["tag"]],
                                       hint="Sorted by condition: most urgent first. Picking one shows its "
                                            "information and walks the worker there.")
        with g.add_folder("Clicked object", visible=False) as self.fo_clicked:   # shown after a click in the scan
            self.img_sel = g.add_image(np.full((8, 24, 3), 235, np.uint8), visible=False)   # scan views of it
            self.md_sel = g.add_markdown("")                                              # its data underneath
            self.b_identify = g.add_button("Ask the AI to know this", color="gray", icon="sparkles",
                                           hint="Uses 1 paid request from the team's AI budget")
            self.md_identify = g.add_markdown("")
            b_close = g.add_button("Close", color="gray", icon="x")
        self.h_hero = g.add_html("")                          # status card of the selected equipment/object
        self.h_audio = g.add_html("", visible=False)          # spoken briefing player
        self.b_ask = g.add_button("Ask the copilot", color="gray", icon="message-chatbot",
                                  hint="Ask the copilot anything about this equipment (1 live AI request).")
        self.md_loading = g.add_markdown("", visible=False)
        self.pb = g.add_progress_bar(0.0, animated=True, visible=False, color="teal")
        self.md_copilot = g.add_markdown("")
        self.md_details: dict[str, Any] = {}
        self.detail_folders: list[Any] = []
        with g.add_folder("Full Excel record", expand_by_default=False) as fo_all:
            self.md_card = g.add_markdown("")                     # readings, events, maintenance (compact card)
            for name in DETAIL_SECTIONS:                          # full data, one collapsible table each
                with g.add_folder(name, expand_by_default=False) as fo:
                    self.md_details[name] = g.add_markdown("")
                self.detail_folders.append(fo)
        self.detail_folders.append(fo_all)
        self._load_token = 0
        self.load_equipment()

        @self.dd_equip.on_update
        def _(e: Any) -> None:
            self.load_equipment()
            if e.client is not None:                    # picked by a person (tour/walk/click selections have no
                self.fo_clicked.visible = False
                self.go_to(self.selected_equipment())   # client): the worker walks there as well

        @self.b_ask.on_click
        def _(e: Any) -> None:
            self.open_question_dialog(e.client)

        @b_close.on_click
        def _(_e: Any) -> None:
            self.fo_clicked.visible = False

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

    def _build_walk_tab(self, g: Any) -> None:
        """Three ways to move the worker: pick equipment from the list, pick an object in the 3D scan
        (button, then click: the worker walks there), or run the inspection tour."""
        self.cb_walk_pov = g.add_checkbox("Worker view (POV)", bool(self.vc["pov"].get("on_start", False)),
                                          hint="See the walk through the worker's eyes. Switch off to get your own "
                                               "camera back.")
        self.cb_pov = self.cb_walk_pov                                # one POV switch (older code reads cb_pov)
        self.md_walk = HtmlText(g.add_html(""), ui.walk_status)     # walking to / arrived at (all modes)
        with g.add_folder("Go to equipment"):
            self.dd_walk = g.add_dropdown(
                "Equipment", list(self.equip_node), initial_value=self.equip_opt[self.target["asset"]["tag"]],
                hint="Every equipment from the Excel pack, most urgent first.")
            b_walk = g.add_button("Walk there", icon="walk",
                                  hint="Collision-free path from where the worker stands; its data opens in Info.")
        with g.add_folder("Go to an object in the scan"):
            b_pick = g.add_button("Pick an object to walk to", icon="hand-click",
                                  hint="Press, then click anything in the 3D scan (pump, valve, ladder, cabinet, ...): "
                                       "the worker walks straight to it.")
            self.h_clicked = g.add_html(ui.selected_chip(None))
        self.pick_to_walk = False
        with g.add_folder("Inspection tour"):
            self._build_tour_tab(g)

        @self.dd_walk.on_update
        def _(_e: Any) -> None:
            self.mark_selected(self.equip_node[self.dd_walk.value])

        @b_walk.on_click
        def _(_e: Any) -> None:
            self.go_to(self.equip_node[self.dd_walk.value])
            if self.playing:                                    # walk started: its information meanwhile
                self.show_tab("Info")

        @b_pick.on_click
        def _(_e: Any) -> None:
            self.pick_to_walk = not self.pick_to_walk                   # press again to cancel
            self.h_clicked.content = (ui.pick_prompt() if self.pick_to_walk else
                                      ui.selected_chip(None))

    def walk_to_picked(self, n: dict) -> None:
        """Pick mode: the clicked object becomes the destination right away."""
        self.pick_to_walk = False
        self.go_to(n)
        if self.playing:
            if not n.get("asset"):
                self.show_object(n)                                     # no Excel record: its 3D information
            self.show_tab("Info")

    def choose_equipment(self, tag: str) -> None:
        """Select equipment in the Info tab (from a click, a walk or the tour) and load its data."""
        opt = self.equip_opt.get(tag)
        if opt and getattr(self, "dd_walk", None) is not None and self.dd_walk.value != opt:
            self.dd_walk.value = opt                               # keep the Walk tab's picker in sync
        if opt and self.dd_equip.value != opt:
            self.dd_equip.value = opt
            self.load_equipment()

    @staticmethod
    def summary_opener(tag: str) -> str:
        return f"Copilot summary for {tag}."

    def load_equipment(self, speak: bool = True, on_spoken: Any = None) -> None:
        """Show the selected equipment's status + copilot diagnosis automatically, without spending credits:
        the saved AI answer when one exists for the same evidence, else the template from the Excel evidence.
        Then the copilot reads its summary aloud (ElevenLabs; cached per text, so repeats are free)."""
        n = self.selected_equipment()
        tag = n["asset"]["tag"]
        self.h_hero.content = ui.asset_hero(n["asset"], self.excel_sources(tag))
        if self.arrived and n is self.target:                 # the tour's spoken briefing owns the panel
            self.mark_selected(n)
            self.fill_details(n)
            self.show_diagnosis()
            return
        self._load_token += 1
        token = self._load_token
        self.md_copilot.content = f"_Loading {tag}…_"
        self.set_audio(None)                                   # stop the previous equipment's summary
        self.set_object_view(False)
        self.mark_selected(n)
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
            audio = None
            if speak and d is not None:
                _, audio = briefing(d, tag, self.cfg, opener=self.summary_opener(tag))
                if current():
                    self.set_audio(audio, f"Copilot summary · {tag}")
            if on_spoken is not None and current():
                on_spoken(audio)
        threading.Thread(target=work, daemon=True).start()

    def show_tab(self, label: str) -> None:
        """Switch every open browser to the tab with this label (see h_tabjs)."""
        self._tab_seq += 1
        # Mantine tabs ignore a bare .click(); they need the full pointer/mouse sequence a real click produces.
        js = ("for(const t of document.querySelectorAll('[role=tab]')){if(t.innerText.trim()==='%s'){"
              "for(const y of ['pointerdown','mousedown','pointerup','mouseup','click']){"
              "t.dispatchEvent(new (y.startsWith('pointer')?PointerEvent:MouseEvent)(y,{bubbles:true,"
              "cancelable:true,button:0,pointerType:'mouse',isPrimary:true}));}break;}}" % label)
        self.h_tabjs.content = (f'<img alt="" style="display:none" data-seq="{self._tab_seq}" '
                                f'src="data:image/gif;base64,invalid{self._tab_seq}" onerror="{js}">')

    def set_object_view(self, on: bool) -> None:
        """Info tab shows a plain scene object (no Excel record): hide the equipment-only sections."""
        for fo in self.detail_folders:
            fo.visible = not on
        self.b_ask.visible = not on                            # the copilot answers about Excel equipment only

    def show_object(self, n: dict) -> None:
        """All we know about a scene object without an Excel record: label, size, scan points, crop image."""
        self._load_token += 1                                  # cancels a pending equipment load
        self.set_audio(None)
        self.set_object_view(True)
        self.h_hero.content = ""                               # the Clicked object card above shows it
        self.md_copilot.content = "_Not in the Excel pack. Pick an equipment in the dropdown above to see Excel data._"

    def excel_sources(self, tag: str) -> list[str]:
        """Provenance of the card's values: Excel sheet + row (row 1 = header) or row counts."""
        try:
            xl = load_pack(str(repo_path(self.cfg["paths"]["xlsx"])))
        except Exception:  # noqa: BLE001 - provenance is optional
            return []
        out = []
        for sheet in ("Asset_Master", "Current_State"):
            df = xl.get(sheet)
            if df is not None and "tag" in df.columns:
                hit = np.flatnonzero((df["tag"] == tag).to_numpy())
                if len(hit):
                    out.append(f"{sheet} · row {int(hit[0]) + 2}")
        for sheet, what in (("Telemetry_Sample_3d", "readings"), ("Alarms_Events", "events"),
                            ("Maintenance_History", "work orders")):
            df = xl.get(sheet)
            if df is not None and "tag" in df.columns:
                k = int((df["tag"] == tag).sum())
                if k:
                    out.append(f"{sheet} · {k} {what}")
        return out

    def fill_details(self, n: dict) -> None:
        """Full Excel + 3D record of the equipment in the collapsible tables (fast: Excel is cached)."""
        try:
            xl = load_pack(str(repo_path(self.cfg["paths"]["xlsx"])))
            det = asset_details(build_evidence(n["asset"]["tag"], self.cfg), xl, n, self.cfg["scene"]["units_to_m"])
        except Exception as e:  # noqa: BLE001 - keep the panel usable
            det = {k: f"⚠️ _Could not load: {type(e).__name__}: {e}_" for k in DETAIL_SECTIONS}
        for k, md in self.md_details.items():
            md.content = det.get(k, "")
        try:
            self.md_card.content = asset_card(build_evidence(n["asset"]["tag"], self.cfg), compact=True, hero=False)
        except Exception as e:  # noqa: BLE001
            self.md_card.content = f"⚠️ _Could not load: {type(e).__name__}: {e}_"

    def set_audio(self, audio: bytes | None, caption: str = "") -> None:
        """Show (and autoplay) an mp3 in the Info tab, or remove the player (stops playback)."""
        if not audio:
            self.h_audio.visible = False
            self.h_audio.content = ""
            return
        b64 = base64.b64encode(audio).decode()
        self.h_audio.content = ui.audio_card(b64, caption)
        self.h_audio.visible = True

    def equipment_view(self, tag: str, d: dict) -> str:
        """Copilot diagnosis (saved AI answer, else built from the Excel evidence): findings with numbered
        citations, next safe steps, sources and the citation check. The status card above shows the asset."""
        body = to_markdown(d).split("\n", 1)[1] if to_markdown(d).startswith("### ") else to_markdown(d)
        return self.answer_header(d) + body

    @staticmethod
    def answer_header(d: dict) -> str:
        src = d.get("meta", {}).get("source", "")
        if src.startswith("template_fallback"):
            return "#### Copilot diagnosis\n_From the Excel evidence, no AI call. **Ask the copilot** for a live answer._\n\n"
        return f"#### Copilot diagnosis · {src}\n\n"

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
            b_go = client.gui.add_button("Ask", icon="send")
            b_cancel = client.gui.add_button("Cancel", color="gray")

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
    def refresh_steps(self, current: int | None = None) -> None:
        """Tour route chips (door › stops); hidden outside tour mode."""
        h = getattr(self, "h_steps", None)
        if h is None:
            return
        on = bool(self.tour) and self.walk_mode == "tour"
        h.content = ui.tour_stepper(self.stops, current, self.next_stop) if on else ""

    def _build_tour_tab(self, g: Any) -> None:
        self.h_steps = g.add_html("")
        b_play = g.add_button(
            "Play / Pause", icon="player-play",
            hint="Start the inspection round: from the door past every equipment with a warning (pausing at each), "
                 "ending at the faulty pump with a spoken copilot briefing. Press again to pause.")
        b_reset = g.add_button("Reset", color="gray", icon="refresh", hint="Back to the door, tour from the start.")
        self.md_stop = HtmlText(g.add_html(""), ui.stop_line)
        self.md_stop.content = ("_Press Play to start the tour._" if self.tour
                                else "_No tour: run python -m pipeline.tour._")
        self.refresh_steps()

        @b_play.on_click
        def _(_e: Any) -> None:
            why = self.nav_stale() or (getattr(self, "tour_note", "") if self.tour_route is None else "")
            if why:                                   # never walk a route built for another layout
                self.playing = False
                self.md_stop.content = f"⚠️ {why}"
                return
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
                self.prefetch_stop_briefings()
            else:
                self.pause_left = max(0.0, self.pause_until - now)
                self.blink_stop(0.0)            # freeze the stop highlight in its visible state

        @b_reset.on_click
        def _(_e: Any) -> None:
            if self.nav_stale() or self.tour_route is None:
                self.md_stop.content = f"⚠️ {self.nav_stale() or self.tour_note}"
                return
            if self.walk_mode != "tour":
                self.set_walk_mode("tour")
            self.playing, self.dist = False, 0.0
            self.leave_target()
            self.next_stop, self.pause_until = 0, 0.0
            self.clear_stop_highlight()
            if self.tour:
                self.md_stop.content = "_Press Play to start the tour._"
            self.refresh_steps()
            self.update()

        @self.sl_pos.on_update
        def _(_e: Any) -> None:
            if not self.playing and not self.nav_stale():
                self.dist = float(self.sl_pos.value)
                self.next_stop = sum(s <= self.dist - 1e-6 for s in self.stop_s)   # scrubbing skips pauses
                self.pause_until = self.pause_left = 0.0
                self.end_stop()
                self.refresh_steps()
                self.update()

    # Scene tab: what the scan became (extraction, Excel health colours, occupancy grid) + Advanced settings
    def _build_scene_tab(self, g: Any) -> None:
        c = self.sg["counts"]
        skip = getattr(self, "movable_members", None) or self.movable
        self.box_nodes = [n for n in self.nodes if n["id"] not in skip]               # order of h_boxes segments
        self.n_attention = sum(1 for n in self.nodes if status_of(n) in ("attention_required", "degraded"))
        self.h_stats = g.add_html(ui.scene_stats(c["objects"], c["mapped"], self.n_attention))
        b_reveal = g.add_button("Replay extraction", icon="player-play",
                                hint="Start from the raw scan; the extracted object boxes appear in waves while "
                                     "the counters climb.")
        self.cb_boxes = g.add_checkbox("Object boxes", False, hint="A frame around every extracted object.")
        self.h_boxes.visible = False
        self.cb_health = g.add_checkbox("Health colours", False,
                                        hint="Colour every box mapped to the Excel pack by its Current_State: "
                                             "green healthy, amber watch/degraded, red attention required.")
        self.cb_grid = g.add_checkbox("Occupancy grid", False,
                                      hint="The navigation grid on the floor: red = obstacles, orange = too close for "
                                           "the worker, teal = walkable. Routes only use teal cells.")
        self.dd_labels = g.add_dropdown("Labels", list(LABEL_MODES), initial_value=self.vc["label_mode"])
        self.dd_display = g.add_dropdown(
            "Display", ("Scan", "BIM", "Overlay"), initial_value="Scan",
            hint="Scan: photoreal splats. BIM: the IFC model generated from this scan (scene.ifc). "
                 "Overlay: splats plus BIM wireframes.")
        self.h_grid: Any = None
        self._reveal_seq = 0

        @b_reveal.on_click
        def _(_e: Any) -> None:
            self._reveal_seq += 1
            threading.Thread(target=self.reveal_extraction, args=(self._reveal_seq,), daemon=True).start()

        @self.cb_health.on_update
        def _(_e: Any) -> None:
            if self.cb_health.value and not self.cb_boxes.value:
                self.cb_boxes.value = True                       # recolouring needs the boxes on screen
            self.color_boxes()

        @self.cb_grid.on_update
        def _(_e: Any) -> None:
            self.show_grid(self.cb_grid.value)

        with g.add_folder("Advanced", expand_by_default=False):
            self._build_advanced(g)

    # --- scene shots: extraction replay, Excel health colours, occupancy grid ---
    def box_colors(self, nodes: list[dict]) -> np.ndarray:
        """Neutral frames, or (Health colours on) Excel Current_State colours for mapped assets."""
        neutral = np.array(self.vc.get("box_neutral_color", [205, 212, 222]), np.uint8)
        dim = np.array(self.vc.get("box_unmapped_color", [95, 104, 118]), np.uint8)
        health = bool(getattr(getattr(self, "cb_health", None), "value", False))
        cols = []
        for n in nodes:
            if not health:
                c = neutral
            elif n["asset"]:
                c = np.array(self.vc["highlight_colors"][status_of(n)], np.uint8)
            else:
                c = dim
            cols.append(np.tile(c, (12, 2, 1)))
        return np.concatenate(cols) if cols else np.zeros((0, 2, 3), np.uint8)

    def color_boxes(self, k: int | None = None) -> None:
        """Show the first k extracted boxes (all when None) in the current colour mode."""
        nodes = self.box_nodes if k is None else self.box_order[:k]
        segs = (np.concatenate([box_edges(n["bbox_min"], n["bbox_max"]) for n in nodes]) if nodes
                else np.zeros((0, 2, 3), np.float32))
        # re-added rather than updated: the browser ignores colour updates on an existing line set
        self.h_boxes = self.server.scene.add_line_segments(
            "/boxes", segs.astype(np.float32), self.box_colors(nodes), thickness=self.vc["box_thickness"],
            thickness_units="screen", visible=bool(getattr(getattr(self, "cb_boxes", None), "value", True)))
        if getattr(self, "_corr_items", None):
            self.show_accepted(self._corr_items)                 # same colour mode as the other boxes
        old = getattr(self, "h_health", None)
        if old is not None:
            old.remove()
        self.h_health = None
        mapped = [n for n in nodes if n["asset"]]
        if getattr(getattr(self, "cb_health", None), "value", False) and mapped:
            segs_m = np.concatenate([box_edges(n["bbox_min"], n["bbox_max"]) for n in mapped]).astype(np.float32)
            self.h_health = self.server.scene.add_line_segments(            # thick status frames on top
                "/health_boxes", segs_m, self.box_colors(mapped), thickness=self.vc.get("health_thickness", 3.5),
                thickness_units="screen")

    def reveal_extraction(self, seq: int) -> None:
        """Raw scan -> boxes appear in waves (sweeping away from the door) while the counters climb."""
        start = np.asarray(self.tour_route[0][0] if self.tour_route else self.pts[0], float)
        self.box_order = sorted(self.box_nodes, key=lambda n: float(np.hypot(n["center"][0] - start[0],
                                                                             n["center"][2] - start[2])))
        for h in self.h_all_labels:
            h.visible = False
        if not self.cb_boxes.value:
            self.cb_boxes.value = True
        waves = int(self.vc.get("reveal_waves", 14))
        total = len(self.box_order)
        self.color_boxes(0)
        self.h_stats.content = ui.scene_stats(0, 0, 0, scanning=True)
        time.sleep(0.8)
        for w in range(1, waves + 1):
            if seq != self._reveal_seq:
                return
            k = round(total * w / waves)
            shown = self.box_order[:k]
            self.color_boxes(k)
            self.h_stats.content = ui.scene_stats(
                k + (len(self.nodes) - total) * w // waves, sum(1 for n in shown if n["asset"]),
                sum(1 for n in shown if status_of(n) in ("attention_required", "degraded")), scanning=w < waves)
            time.sleep(float(self.vc.get("reveal_wave_s", 0.45)))
        self.color_boxes()
        c = self.sg["counts"]
        self.h_stats.content = ui.scene_stats(c["objects"], c["mapped"], self.n_attention)
        self._sync_visit_labels()

    def show_grid(self, on: bool) -> None:
        """Occupancy grid on the floor (red obstacle, orange too close, teal walkable): the planner's own layers."""
        if self.h_grid is not None:
            self.h_grid.remove()
            self.h_grid = None
        if not on:
            return
        pl = self.get_planner()
        L, grid = pl.layers, pl.grid
        obst, infl, walk = L["obstacles"], L["inflated"], pl.walk
        kinds = [(obst, (235, 60, 70)), (infl & ~obst & L["floor"], (245, 150, 40)), (walk, (20, 184, 166))]
        pts, cols = [], []
        y = self.cfg["scene"]["floor_y"] - self.vc.get("grid_lift", 0.01)            # up = -y
        for mask, col in kinds:
            ix, iz = np.nonzero(mask)
            x, z = grid.center(ix, iz)
            pts.append(np.stack([x, np.full(len(x), y), z], 1))
            cols.append(np.tile(np.array(col, np.uint8), (len(x), 1)))
        self.h_grid = self.server.scene.add_point_cloud(
            "/occupancy_grid", np.concatenate(pts).astype(np.float32), np.concatenate(cols),
            point_size=grid.cell * 0.85, point_shape="square")

    def _bind_settings(self) -> None:
        @self.cb_walk_pov.on_update
        def _(_e: Any) -> None:
            self.set_pov(self.cb_walk_pov.value)

    def _build_advanced(self, g: Any) -> None:
        """Everything the demo doesn't need on screen: worker, splat display, inspector, boundary review."""
        self.md_bim = g.add_markdown(self.bim_note)
        if self.edit_note:
            g.add_markdown(f"<small>{self.edit_note}</small>")
        with g.add_folder("Worker", expand_by_default=False):
            self.sl_pos = g.add_slider("Tour position", 0.0, float(self.s[-1]), 0.01, 0.0,
                                       hint="Drag to move the worker along the route (pauses are skipped).")
            self.sl_size = g.add_slider("Worker size", 0.5, 1.2, 0.05,
                                        self.rig.size if self.rig is not None else 1.0,
                                        visible=self.rig is not None,
                                        hint="Scales the worker to fit the room (the scan's metre scale is an "
                                             "estimate). Set the default in viewer.avatar.rigged.size.")
            self.sl_speed = g.add_slider("Walk speed (u/s)", 0.1, 2.0, 0.1, self.vc["walk_speed"])
            self.sl_radius = g.add_slider("Proximity radius", 0.5, 3.0, 0.1, self.vc["proximity_radius"])
        dc = self.vc["display"]
        with g.add_folder("Scene display", expand_by_default=False):
            cb_ceiling = g.add_checkbox("Show ceiling", dc["show_ceiling"])
            cb_walls = g.add_checkbox("Show walls", dc["show_walls"])
            cb_crop = g.add_checkbox("Crop outside walls", dc["crop_outside_walls"])
            sl_dark = g.add_slider("Hide dark splats below", 0.0, 0.2, 0.01, dc["dark_threshold"])
            b_apply = g.add_button("Apply filter", color="gray")
            md_disp = g.add_markdown("")
        with g.add_folder("Boundary repair: before / after", expand_by_default=False):
            g.add_markdown(f"After (white frames): **{len(self.nodes)}** objects · **{self.n_ctx}** context repairs "
                           f"(magenta) · **{self.n_geo}** geometry pipe joins (cyan)  \nBefore: **{self.n_baseline}** "
                           f"baseline objects (orange). Fewer boxes is not the goal; check whole components.")
            self.cb_base = g.add_checkbox("Show baseline boxes (before)", False)
            self.cb_rep = g.add_checkbox("Highlight repaired objects", False)
            self.h_rep.visible = False
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

        @self.sl_size.on_update
        def _(_e: Any) -> None:
            if self.rig is not None:
                self.rig.set_size(float(self.sl_size.value))

        @self.dd_labels.on_update
        def _(_e: Any) -> None:
            mode = self.dd_labels.value
            for n, h, eq in zip(self.nodes, self.h_all_labels, self.is_equipment):
                h.visible = label_visible(mode, eq, status_of(n))
                h.text = label_text(n, full=mode == "All objects")
            for h in self.h_corr_acc:
                if "_label" in h.name:
                    h.visible = self.review_labels_on()
            self._sync_visit_labels()
            self.update()

        @self.cb_boxes.on_update
        def _(_e: Any) -> None:
            if self.cb_boxes.value:
                self.color_boxes()
            self.h_boxes.visible = self.cb_boxes.value
            if getattr(self, "h_health", None) is not None:
                self.h_health.visible = self.cb_boxes.value
            for h in self.h_corr_acc:
                if "_label" not in h.name:
                    h.visible = self.corr_frames_on()
            for it in self.inst.values():
                it["box"].visible = self.cb_boxes.value

        @self.sl_radius.on_update
        def _(_e: Any) -> None:
            r = float(self.sl_radius.value)
            self.server.scene.add_icosphere("/avatar/ring", radius=1.0,
                                            color=tuple(self.vc["avatar"]["ring_color"]),
                                            opacity=0.15, scale=(r, 0.01, r),
                                            visible=self.vc["avatar"].get("show_ring", False))
            self.update()

        @cb_ceiling.on_update
        def _(_e: Any) -> None:
            self._group_vis["ceiling"] = cb_ceiling.value
            self.h_splat["ceiling"].visible = cb_ceiling.value and self.display != "BIM"

        @cb_walls.on_update
        def _(_e: Any) -> None:
            self._group_vis["walls"] = cb_walls.value
            self.h_splat["walls"].visible = cb_walls.value and self.display != "BIM"

        @self.dd_display.on_update
        def _(_e: Any) -> None:
            self.set_display(self.dd_display.value)

        @b_apply.on_click
        def _(_e: Any) -> None:
            md_disp.content = "_Updating splats…_"
            c = self.rebuild_splats(float(sl_dark.value), bool(cb_crop.value))
            md_disp.content = (f"room {c['room']:,} · walls {c['walls']:,} · ceiling {c['ceiling']:,} splats "
                               f"(of {len(self.sp_lum):,})")


        @self.cb_corr.on_update
        def _(_e: Any) -> None:
            for h in self.h_corr:
                h.visible = self.cb_corr.value
            for h in self.h_corr_acc:
                h.visible = (self.review_labels_on() and self.cb_corr.value) if "_label" in h.name \
                    else self.corr_frames_on()

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

    def corr_frames_on(self) -> bool:
        """Manual-correction frames are ordinary object boxes: shown with 'Object boxes' (and the preview switch)."""
        boxes = getattr(self, "cb_boxes", None)
        corr = getattr(self, "cb_corr", None)
        return bool(boxes is not None and boxes.value and (corr is None or corr.value))

    def show_accepted(self, items: list[dict]) -> None:
        """Accepted/applied manual corrections (api/annotate.py): drawn like every other object box (same
        neutral colour and thickness; dim grey with Health colours, as they have no workbook identity)."""
        self._corr_items = items
        for h in self.h_corr_acc:
            h.remove()
        self.h_corr_acc = []
        sc = self.server.scene
        health = bool(getattr(getattr(self, "cb_health", None), "value", False))
        col = tuple(self.vc.get("box_unmapped_color", [95, 104, 118]) if health
                    else self.vc.get("box_neutral_color", [205, 212, 222]))
        for it in items:
            bb = it["bbox"]
            self.h_corr_acc.append(sc.add_line_segments(f"/corrections/{it['cid']}", box_edges(*bb), col,
                                                        thickness=self.vc["box_thickness"], thickness_units="screen",
                                                        visible=self.corr_frames_on()))
            self.h_corr_acc.append(sc.add_label(f"/corrections/{it['cid']}_label",
                                                f"{it['cid']} {it['cls']} · {it['status']} · {it['evidence']}",
                                                position=((bb[0][0] + bb[1][0]) / 2, bb[0][1] - 0.05,
                                                          (bb[0][2] + bb[1][2]) / 2), anchor="bottom-center",
                                                font_screen_scale=0.7, visible=self.review_labels_on()))

    def review_labels_on(self) -> bool:
        """Boundary-review labels (MC-xxxx · applied · …) only with Labels = All objects: keeps the scan readable."""
        return getattr(getattr(self, "dd_labels", None), "value", self.vc["label_mode"]) == "All objects"

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
        self.update_visit_highlight(p)
        if self.walk_mode == "tour":            # "Go to equipment" never auto-diagnoses (paid budget)
            at_target = (self.dist >= self.s[-1] - 1e-6 and self.box_gap(self.target, p) <= self.arrive_gap
                         and not self.warning_stop_due())
            if at_target and not self.arrived:
                self.arrived = True
                self.on_arrival()
            elif not at_target and self.arrived:
                self.leave_target()
        self.pose_avatar(p, moving=self.playing and self.dist < self.s[-1] and time.time() >= self.pause_until)
        r = float(self.sl_radius.value)
        idx = self.tree.query_ball_point(p[[0, 2]], r + self.max_half)
        near_all = [self.nodes[i] for i in idx if self.box_gap(self.nodes[i], p) <= r
                    and not self.nodes[i].get("_layout_removed")]
        mapped = [n for n in near_all if n["asset"]]
        others = [n for n in near_all if not n["asset"]]
        self.light_nearby(mapped)
        for h in self._labels:
            h.remove()
        show = []                                        # persistent labels cover the other modes
        if self.dd_labels.value == "Nearby only":
            show = mapped + [n for n in others if n["label"] not in self.vc["near_label_skip"]]
        if self.visit_active is not None:
            show = [n for n in show if n["id"] != self.visit_active["object_id"]]
        self._labels = [self.server.scene.add_label(
            f"/near/{n['id']}", label_text(n),
            position=(n["center"][0], n["bbox_min"][1], n["center"][2]), anchor="bottom-center",
            font_screen_scale=1.0 if n["asset"] else self.vc["other_label_scale"])
            for n in show]

    def light_nearby(self, mapped: list[dict]) -> None:
        """While walking: nearby Excel assets light up in their health colour."""
        moving = self.playing and self.dist < self.s[-1]
        active = self.visit_active["object_id"] if self.visit_active else None
        lit = [n for n in mapped if n["id"] != active] if moving else []
        ids = tuple(n["id"] for n in lit)
        if ids != getattr(self, "_lit_ids", ()):
            self._lit_ids = ids
            if lit:
                segs, _ = edges_for(lit, self.vc["colors"])
                cols = np.concatenate([np.tile(np.array(self.vc["highlight_colors"][status_of(n)], np.uint8),
                                               (12, 2, 1)) for n in lit])
                self.h_highlight = self.server.scene.add_line_segments(
                    "/highlight", segs, cols, thickness=self.vc["highlight_thickness"], thickness_units="screen")
            self.h_highlight.visible = bool(lit)

    def update_visit_highlight(self, p: np.ndarray) -> None:
        """Release on departure, with hysteresis to avoid flicker at the proximity boundary."""
        if self.visit_active is None:
            return
        n = self.lookup(self.visit_active["object_id"])
        radius = max(float(self.sl_radius.value), self.visit_arrival_gap)
        if n is None or self.box_gap(n, p) > radius + float(self.vc.get("visit_release_margin", .25)):
            self.clear_visit_highlight()
        else:
            self.blink_stop(0.0)

    def _sync_visit_labels(self) -> None:
        """Keep the health outline/fill legible without discarding Oren's selection controls."""
        active = self.visit_active["object_id"] if self.visit_active else None
        mode = getattr(getattr(self, "dd_labels", None), "value", self.vc.get("label_mode", "Equipment"))
        for n, h, eq in zip(self.nodes, self.h_all_labels, self.is_equipment):
            h.visible = n["id"] != active and label_visible(mode, eq, status_of(n))
        selected = getattr(self, "selection_highlight_id", None)
        if selected is None and self.selected is not None:
            selected = self.selected["id"]
        # A separately moved copy retains its own selection feedback.
        same = active is not None and selected == active and getattr(self, "sel_inst", None) in (None, active)
        self.h_selected.visible = not same
        fill = getattr(self, "h_selected_fill", None)
        if fill is not None:
            fill.visible = not same

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
        """Object hit first by the click ray: its scan points (Scan), its IFC element (BIM) or either (Overlay).
        Scan and BIM geometry of one object resolve to the same scene object."""
        origin = np.asarray(origin, np.float64)
        d = direction / (np.linalg.norm(direction) + 1e-12)
        best: tuple[float, Any, Any] = (np.inf, None, None)
        if self.display in ("Scan", "Overlay") or self.bim is None:
            best = self._pick_scan(origin, d)
        if self.display in ("BIM", "Overlay") and self.bim is not None:
            b = self._pick_bim(origin, d)
            if b[0] < best[0]:
                best = b
        self._picked_inst = best[2]
        return best[1]

    def _pick_scan(self, origin: np.ndarray, d: np.ndarray) -> tuple[float, Any, Any]:
        v = self.pick_xyz - origin
        t = v @ d
        perp = np.linalg.norm(v - np.outer(t, d), axis=1)
        tol = self.vc["pick_radius"] + self.vc["pick_radius_per_unit"] * np.maximum(t, 0)
        objs = self.sg["objects"]
        hit = (t > 0) & (perp < tol) & (self.pick_lab >= 0)
        hit &= np.array([objs[i]["kind"] == "object" for i in range(len(objs))])[np.maximum(self.pick_lab, 0)]
        hit &= ~np.isin(self.pick_lab, [self.obj_index[m] for m in self.movable_members])
        best = (float(t[hit].min()), objs[int(self.pick_lab[np.where(hit)[0][np.argmin(t[hit])]])], None) \
            if hit.any() else (np.inf, None, None)
        for name, it in self.inst.items():                    # movable objects where they are now
            if it["removed"]:
                continue
            o, dl = self._to_local(name, origin, d)
            v = self.movable[it["oid"]]["pick"] - o
            t = v @ dl
            perp = np.linalg.norm(v - np.outer(t, dl), axis=1)
            ok = (t > 0) & (perp < self.vc["pick_radius"] + self.vc["pick_radius_per_unit"] * np.maximum(t, 0))
            if ok.any() and float(t[ok].min()) < best[0]:
                best = (float(t[ok].min()), self.node_by_id[it["oid"]], name)
        return best

    def _pick_bim(self, origin: np.ndarray, d: np.ndarray) -> tuple[float, Any, Any]:
        from viewer.bim_layer import ray_hits
        t, oid = self.bim.pick(origin, d, [o for o in self.bim.mesh if o not in self.movable_members])
        best = (t, self.node_by_id.get(oid), None)
        for name, it in self.inst.items():
            m = self.movable[it["oid"]]
            if it["removed"] or m["bim"] is None:
                continue
            o, dl = self._to_local(name, origin, d)
            t = ray_hits(m["bim"][0].astype(np.float64), m["bim"][1], o, dl)
            if t < best[0]:
                best = (t, self.node_by_id[it["oid"]], name)
        return best

    def _to_local(self, name: str, origin: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ray in an instance's local frame (relative to its source object's pivot)."""
        h = self.inst[name]["h"]
        R = SO3(np.asarray(h.wxyz)).as_matrix()
        return R.T @ (origin - np.asarray(h.position)), R.T @ d

    def on_scene_click(self, e: Any) -> None:
        n = self.pick(np.array(e.ray_origin), np.array(e.ray_direction))
        if n is not None:
            self.select(n, inst=self._picked_inst)
            if getattr(self, "pick_to_walk", False):
                self.walk_to_picked(n)
            else:
                if not n.get("asset"):
                    self.show_object(n)                                 # Info below the card: no Excel record
                self.show_tab("Info")

    def mark_selected(self, n: dict, segs: np.ndarray | None = None) -> None:
        """Highlight selection with thin edges; never cover the equipment with filled box faces.
        segs: the object's current box edges (a moved instance); default = its box in the scan."""
        if segs is None:
            segs, _ = edges_for([n], self.vc["colors"])
        col = np.array(self.vc["selected_color"], np.uint8)
        self.h_selected.points = segs
        self.h_selected.colors = np.tile(col, (len(segs), 2, 1))
        old_fill = getattr(self, "h_selected_fill", None)
        if old_fill is not None:
            old_fill.remove()
        self.h_selected_fill = None
        self.selection_highlight_id = n["id"]
        self._sync_visit_labels()

    def view_node(self, n: dict, inst: str | None) -> dict:
        """What a selection shows: the scene object, or for a copy an identity-free object at the copy's place
        (copies never inherit the source's asset record, health or verified identity)."""
        if inst is None or not getattr(self, "inst", {}).get(inst, {}).get("copy"):
            return n
        lo, hi = self.inst_bounds(inst)
        return {**n, "id": inst, "asset": None, "copy_of": n["id"], "bbox_min": lo.tolist(), "bbox_max": hi.tolist(),
                "center": ((lo + hi) / 2).tolist(), "extent": (hi - lo).tolist(), "label_source": "copy",
                "label_reason": f"copy of {n['id']}; no asset identity", "repair": None, "fragments": None}

    def bim_text(self, n: dict) -> str:
        src = n.get("copy_of", n["id"])
        if self.bim is None or src not in self.bim.mesh:
            return ""
        m = self.bim.mesh[src]
        approx = {"scan mesh": "mesh of the scan points", "fitted cylinder": "fitted cylinder (DN-snapped)"}
        how = approx.get(m["model_source"], m["model_source"] or "?")
        return (f"\n\n**BIM:** `{m['cls']}` · {how}" + (" · approximate catalog shape" if "catalog" in how else "")
                + (f" · GlobalId `{m['guid']}`" if "copy_of" not in n else " · copy (not in scene.ifc)"))

    def select(self, n: dict, inst: str | None = None) -> None:
        self.sel_inst = inst if inst is not None or n["id"] not in self.movable else n["id"]
        n = self.view_node(n, self.sel_inst)
        self.selected = n
        self.hide_gizmo()
        if self.movable:
            self.md_edit.content = ((f"**{self.sel_inst}** selected" + (f" (copy of {n['copy_of']}, no asset identity)."
                                                                       if "copy_of" in n else "."))
                                    if self.sel_inst else
                                    "_This object can't be edited; pick one of the movable objects._")
        self.mark_selected(n, self.inst_box(self.sel_inst) if self.sel_inst else None)
        self.md_sel.content = selection_text(n, self.cfg["scene"]["units_to_m"]) + self.bim_text(n)
        crop = out_path(self.cfg, "crops") / f"{n['id']}.png"
        if crop.exists():
            from PIL import Image
            self.img_sel.image = np.asarray(Image.open(crop).convert("RGB"))
            self.img_sel.visible = True
        else:
            self.img_sel.visible = False
        self.md_identify.content = ""
        self.fo_clicked.visible = True
        what = (f"<b>{ui.esc(n['asset']['tag'])}</b> · {ui.esc(n['label'])}" if n["asset"] else
                f"<b>{ui.esc(n['id'])}</b> · {ui.esc(n['label'])} <em>(no Excel record)</em>")
        self.h_clicked.content = ui.selected_chip(what)
        if n["asset"]:                                               # clicked equipment -> Info tab
            self.choose_equipment(n["asset"]["tag"])

    # --- edit scene: move / remove / duplicate the objects that cut out cleanly (viewer.removable) ---
    # One rigid transform per instance (frame /movable/<name>) carries its splat piece, BIM mesh, box, label and
    # pick points; the layout lives in out/scene_edits.json (viewer/scene_edits.py), never in scene.ifc.
    def _build_edit_tab(self, g: Any) -> None:
        g.add_markdown(f"**{len(self.movable)} objects** cut out cleanly and can be moved, removed or duplicated "
                       "(scan piece and BIM element together). Click one in the 3D view or pick it below.")
        dd = g.add_dropdown("Movable object", ["-", *self.movable], initial_value="-")
        cb_show = g.add_checkbox("Show movable objects", self.show_movable)
        self.md_edit = g.add_markdown("_Nothing selected._")
        b_move = g.add_button("Move", color="gray", icon="arrows-move")
        add_nudge_gui(
            gui=g,
            vc=self.vc,
            up=up_vector(self.cfg["scene"]["up_direction"]),
            units_to_m=self.cfg["scene"]["units_to_m"],
            on_nudge=self.nudge,
        )
        b_dup = g.add_button("Duplicate", color="gray", icon="copy", hint="The copy has no asset identity (no Excel record).")
        b_remove = g.add_button("Remove", icon="trash", color="red")
        b_restore = g.add_button("Restore object", color="gray", hint="Put it back where it was and delete its copies")
        b_undo = g.add_button("Undo", color="gray", icon="arrow-back-up")
        b_restore_all = g.add_button("Restore all", color="gray")
        with g.add_folder("Layout and navigation"):
            self.md_layout = g.add_markdown("")
            b_apply = g.add_button("Apply layout", icon="route",
                                   hint="Rebuild occupancy, walk routes and the tour for the edited layout.")
            g.add_markdown("<small>Edits are saved to `scene_edits.json` (visual layout); `scene.ifc` is not "
                           "changed. An edited IFC is only written by the button below, as a separate file.</small>")
            b_ifc = g.add_button("Export edited IFC (separate file)", color="gray", icon="file-export")
            self.md_ifc = g.add_markdown("")
        self.update_layout_status()

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
                self.duplicate(self.sel_inst)

        @b_remove.on_click
        def _(_e: Any) -> None:
            if self.sel_inst:
                self.remove_instance(self.sel_inst)

        @b_restore.on_click
        def _(_e: Any) -> None:
            oid = (self.selected or {}).get("copy_of") or (self.selected or {}).get("id")
            if oid in self.movable:
                self.edit_action(f"restore {oid}", lambda: self.restore(oid))
                self.select(self.node_by_id[oid])

        @b_undo.on_click
        def _(_e: Any) -> None:
            self.undo_edit()

        @b_restore_all.on_click
        def _(_e: Any) -> None:
            self.edit_action("restore all", lambda: [self.restore(oid) for oid in list(self.movable)])
            self.hide_gizmo()

        @b_apply.on_click
        def _(_e: Any) -> None:
            threading.Thread(target=self.apply_layout, daemon=True).start()

        @b_ifc.on_click
        def _(_e: Any) -> None:
            threading.Thread(target=self.export_edited_ifc, daemon=True).start()

    def add_instance(self, name: str, oid: str, pos: Any = None, wxyz: Any = (1.0, 0.0, 0.0, 0.0),
                     removed: bool = False) -> None:
        """Frame at the object's pose holding its splat, BIM mesh, box and label, so they move together."""
        m, sc = self.movable[oid], self.server.scene
        h = sc.add_frame(f"/movable/{name}", show_axes=False, position=tuple(m["c"] if pos is None else pos),
                         wxyz=tuple(wxyz), visible=not removed)
        splat = sc.add_gaussian_splats(f"/movable/{name}/splat", centers=m["xyz"], rgbs=m["rgb"], opacities=m["op"],
                                       covariances=m["cov"], visible=self.display != "BIM")
        box = sc.add_line_segments(f"/movable/{name}/box", m["box"],
                                   tuple(self.vc["colors"][status_of(self.node_by_id[oid])] if name == oid
                                         else self.vc["colors"]["no_record"]),
                                   thickness=self.vc["box_thickness"], thickness_units="screen",
                                   visible=self.cb_boxes.value if hasattr(self, "cb_boxes") else False)
        show = sc.add_line_segments(f"/movable/{name}/show", m["box"], tuple(self.vc["movable_color"]),
                                    thickness=self.vc["selected_thickness"], thickness_units="screen",
                                    visible=self.show_movable)
        self.inst[name] = {"oid": oid, "h": h, "splat": splat, "box": box, "show": show, "bim": None,
                           "removed": removed, "copy": name != oid}
        self._add_instance_bim(name)

    def _add_instance_bim(self, name: str) -> None:
        it = self.inst[name]
        m = self.movable[it["oid"]]
        if self.bim is None or m["bim"] is None or (self.display == "Scan" and not self.h_bim):
            return
        it["bim"] = self.server.scene.add_mesh_simple(
            f"/movable/{name}/bim", m["bim"][0], m["bim"][1].astype(np.uint32), color=self._bim_color(name),
            wireframe=self.display == "Overlay", visible=self.display != "Scan")

    def inst_box(self, name: str) -> np.ndarray:
        h = self.inst[name]["h"]
        R = SO3(np.asarray(h.wxyz)).as_matrix()
        return (self.movable[self.inst[name]["oid"]]["box"] @ R.T + np.asarray(h.position)).astype(np.float32)

    def inst_bounds(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounds of an instance as it is shown now (its points under the shared transform)."""
        h, m = self.inst[name]["h"], self.movable[self.inst[name]["oid"]]
        R = SO3(np.asarray(h.wxyz)).as_matrix()
        P = m["pick"] @ R.T + np.asarray(h.position)
        return P.min(0), P.max(0)

    def live_bounds(self, n: dict) -> tuple[np.ndarray, np.ndarray]:
        """Where an object is shown now: moved editable objects follow their instance; others their scan box."""
        if n["id"] in self.inst and self._is_moved(n["id"]):
            return self.inst_bounds(n["id"])
        return np.array(n["bbox_min"]), np.array(n["bbox_max"])

    def _is_moved(self, name: str) -> bool:
        it = self.inst[name]
        return it["copy"] or not (np.allclose(it["h"].position, self.movable[it["oid"]]["c"], atol=1e-4)
                                  and np.allclose(it["h"].wxyz, (1, 0, 0, 0), atol=1e-5))

    def show_gizmo(self, name: str) -> None:
        h = self.inst[name]["h"]
        self.hide_gizmo()
        up = up_vector(self.cfg["scene"]["up_direction"])
        extent = float(np.ptp(self.movable[self.inst[name]["oid"]]["box"].reshape(-1, 3), 0).max())
        self._gizmo_dirty = False
        self.gizmo = self.server.scene.add_transform_controls(
            "/gizmo", scale=gizmo_scale(extent=extent, vc=self.vc), line_width=self.vc.get("gizmo_line_width", 5.0),
            position=h.position, wxyz=h.wxyz, depth_test=False,
            active_axes=tuple(bool(x) for x in up == 0),                 # drag on the floor; Lift buttons go up / down
            rotation_limits=((0, 0), (-1000, 1000), (0, 0)))               # yaw only (up = -y)

        @self.gizmo.on_update
        def _(_e: Any) -> None:
            self.move_instance(name, self.gizmo.position, self.gizmo.wxyz)

    def move_instance(self, name: str, position: Any, wxyz: Any) -> None:
        """Gizmo drag (one undo step per Move): splat, BIM mesh, box, label and picking share this transform."""
        if not self._gizmo_dirty:
            se.push_history(self.edits, f"move {name}")
            self._gizmo_dirty = True
        h = self.inst[name]["h"]
        h.position, h.wxyz = tuple(float(x) for x in position), tuple(float(x) for x in wxyz)
        if self.sel_inst == name:
            self.h_selected.points = self.inst_box(name)
        if self.visit_active is not None and self.visit_active["object_id"] == name:
            self._draw_stop_box(self.lookup(name))
        self.save_edits()
        self.layout_changed()

    def nudge(self, dp: np.ndarray, yaw: float) -> None:
        """Step the selected object (Move / Lift / Turn buttons)."""
        name = self.sel_inst
        if not name or self.inst[name]["removed"]:
            return
        h = self.inst[name]["h"]
        up = up_vector(self.cfg["scene"]["up_direction"])
        position = np.asarray(h.position) + dp
        wxyz = (SO3.exp(up * yaw) @ SO3(np.asarray(h.wxyz))).wxyz
        self._gizmo_dirty = False
        self.move_instance(name, position, wxyz)
        if self.gizmo is not None:
            self.gizmo.position, self.gizmo.wxyz = h.position, h.wxyz

    def hide_gizmo(self) -> None:
        if self.gizmo is not None:
            self.gizmo.remove()
            self.gizmo = None
        self._gizmo_dirty = False

    def duplicate(self, name: str) -> str:
        src = self.inst[name]
        oid = src["oid"]
        k = 1 + max([int(x.rsplit("_copy", 1)[1]) for x in self.inst if x.startswith(oid + "_copy")] or [0])
        new = f"{oid}_copy{k}"
        self.edit_action(f"duplicate {name}", lambda: self.add_instance(
            new, oid, np.asarray(src["h"].position) + np.array(self.vc["duplicate_offset"]), src["h"].wxyz))
        self.select(self.node_by_id[oid], inst=new)
        self.show_gizmo(new)
        return new

    def remove_instance(self, name: str) -> None:
        def do() -> None:
            if not self.inst[name]["copy"]:
                self.inst[name]["removed"] = True
                self.inst[name]["h"].visible = False
            else:                                                       # a copy is deleted for good
                self.inst.pop(name)["h"].remove()
        if getattr(self, "_visit_tint_id", None) == name:
            self.clear_visit_highlight()
        self.edit_action(f"remove {name}", do)
        self.md_edit.content = f"**{name}** removed."
        self.sel_inst = None
        self.hide_gizmo()
        self.h_selected.points, self.h_selected.colors = edges_for([], self.vc["colors"])

    def restore(self, oid: str) -> None:
        for name in [x for x in self.inst if x.startswith(oid + "_copy")]:
            self.inst.pop(name)["h"].remove()
        it = self.inst[oid]
        it["h"].position, it["h"].wxyz, it["h"].visible = tuple(self.movable[oid]["c"]), (1.0, 0.0, 0.0, 0.0), True
        it["removed"] = False

    def edit_action(self, action: str, fn: Any) -> None:
        """Record the layout for undo, change it, save and invalidate navigation."""
        self.hide_gizmo()
        se.push_history(self.edits, action)
        fn()
        self.save_edits()
        self.layout_changed()

    def undo_edit(self) -> None:
        self.hide_gizmo()
        what = se.undo(self.edits)
        if what is None:
            self.md_edit.content = "_Nothing to undo._"
            return
        self.sync_instances()
        self.save_edits()
        self.layout_changed()
        self.md_edit.content = f"Undid: {what}."

    def sync_instances(self) -> None:
        """Shown instances <- self.edits (startup, undo)."""
        if self.edits is None:
            return
        want = self.edits["instances"]
        for name in [x for x in self.inst if x not in want]:
            self.inst.pop(name)["h"].remove()
        for name, e in want.items():
            if name not in self.inst:
                self.add_instance(name, e["source"])
            it = self.inst[name]
            it["h"].position, it["h"].wxyz = tuple(e["position"]), tuple(e["wxyz"])
            it["removed"] = e["removed"]
            it["h"].visible = not e["removed"]

    def save_edits(self) -> None:
        if self.edits is None:
            return
        self.edits["instances"] = {name: se.instance(it["oid"], self.movable[it["oid"]]["c"], it["copy"],
                                                     it["h"].position, it["h"].wxyz, it["removed"])
                                   for name, it in self.inst.items()}
        se.save(self.cfg, self.edits)

    # --- layout -> navigation (bounds, occupancy, routes, tour) ---
    def layout_hash(self) -> str:
        return se.layout_hash(self.edits["instances"], self.edits["pivots"]) if self.edits else ""

    def nav_stale(self) -> str | None:
        """Why walking must wait for Apply layout (None = navigation matches the shown layout)."""
        if self.edits is None or self.layout_hash() == self._nav_layout:
            return None
        return ("The object layout changed: routes, occupancy and the tour are out of date. "
                "Press **Edit scene → Apply layout** to rebuild navigation (or undo / restore the edits).")

    def layout_changed(self) -> None:
        """After any edit: stop walking on stale routes and hide them until Apply layout (or the edit is undone)."""
        if self.edits is None:
            return
        cur = self.layout_hash()
        applied = self.edits.get("applied_layout") or {}
        if cur != self._nav_layout and cur == self.base_layout:
            self._use_nav(None)                                         # back to the unedited scene
        elif cur != self._nav_layout and cur == applied.get("hash"):    # back to the applied layout
            res = self._try_build_nav(cur)
            if res is not None:
                self._use_nav(res)
        why = self.nav_stale()
        if why:
            self.playing = False
            self.h_path.visible = False
            for h in self.h_tour_markers:
                h.visible = False
            self.md_walk.content = f"⚠️ {why}"
            if self.tour:
                self.md_stop.content = f"⚠️ {why}"
        self.update_layout_status()

    def update_layout_status(self) -> None:
        if not hasattr(self, "md_layout") or self.edits is None:
            return
        n_moved = sum(self._is_moved(k) and not it["copy"] for k, it in self.inst.items())
        n_copy = sum(it["copy"] for it in self.inst.values())
        n_rem = sum(it["removed"] and not it["copy"] for it in self.inst.values())
        why = self.nav_stale()
        state = ("⚠️ **Navigation out of date**: walking is disabled until you press Apply layout." if why else
                 "✅ Navigation matches this layout." + (" (rebuilt for the edited layout)" if self.nav_inputs else ""))
        self.md_layout.content = (f"{n_moved} moved · {n_rem} removed · {n_copy} copies · "
                                  f"{len(self.edits['history'])} undo steps  \n{state}"
                                  + (f"  \n⚠️ {self.tour_note}" if getattr(self, "tour_note", "") else "")
                                  + (f"  \n⚠️ {self.nav_error}" if why and getattr(self, "nav_error", "") else ""))

    def _try_build_nav(self, layout: str) -> dict | None:
        """_build_nav for the automatic rebuilds (edit callbacks, startup). On failure (missing / stale out/ files)
        navigation stays marked out of date, so walking remains blocked and the reason is shown; the unedited
        routes are not used, because they would be stale for this layout."""
        try:
            res = self._build_nav(layout)
        except Exception as e:  # noqa: BLE001 - a broken rebuild must not stop the viewer or an edit
            self.nav_error = f"Navigation rebuild failed ({type(e).__name__}: {e}); press Apply layout to retry."
            print(f"[viewer] {self.nav_error}")
            return None
        self.nav_error = ""
        return res

    def _build_nav(self, layout: str) -> dict:
        from pipeline.layout_nav import build
        members = {oid: [self.obj_index[x] for x in m["members"]] for oid, m in self.movable.items()}
        return build(self.cfg, self.edits, members, layout)

    def apply_layout(self) -> None:
        if self.edits is None:
            return
        cur = self.layout_hash()
        if self.nav_stale() is None:
            self.update_layout_status()
            self.md_layout.content += "  \n_Already applied._"
            return
        self.md_layout.content = "_Rebuilding occupancy, routes and tour for the edited layout…_"
        try:
            res = None if cur == self.base_layout else self._build_nav(cur)
        except Exception as e:  # noqa: BLE001
            self.nav_error = f"Apply layout failed: {type(e).__name__}: {e}"
            self.md_layout.content = f"⚠️ {self.nav_error}"
            return
        self.nav_error = ""
        if self.layout_hash() != cur:                                   # edited again meanwhile
            self.update_layout_status()
            return
        self.edits["applied_layout"] = None if res is None else {
            "hash": cur, "at": time.strftime("%Y-%m-%d %H:%M:%S"), "report": res["report"]}
        self.save_edits()
        self._use_nav(res)
        if res is not None:
            r = res["report"]
            self.md_layout.content += (f"  \nObstacle cells {r['obstacle_cells_before']} → {r['obstacle_cells']}, "
                                       f"walkable {r['walkable_cells_before']} → {r['walkable_cells']}; "
                                       f"target {r['target']} {r['target_status']}.")

    def _use_nav(self, res: dict | None) -> None:
        """Switch bounds, planner, path and tour to a layout's navigation (None = the unedited out/ files)."""
        from pipeline.tour import plan_tour
        self.playing = False
        if self.visit_active is not None:
            self.clear_stop_highlight()
        for n in self.nodes:                                            # bounds back to the scan, then the layout's
            n.update(self._bounds0[n["id"]])
            n.pop("_layout_removed", None)
        self.nav_inputs = None if res is None else res["inputs"]
        for oid, b in ({} if res is None else res["bounds"]).items():
            if b["removed"]:
                self.node_by_id[oid]["_layout_removed"] = True
            else:
                self.node_by_id[oid].update({k: b[k] for k in ("bbox_min", "bbox_max", "center", "extent")})
        self.tree = cKDTree(np.array([n["center"] for n in self.nodes])[:, [0, 2]])
        self.max_half = max(float(np.hypot(n["extent"][0], n["extent"][2])) / 2 for n in self.nodes)
        self.planner = None
        self.tour_note = ""
        if res is None:
            self.tour = self.tour0
            route = (np.array((self.tour0 or self.path0)["points"], np.float64),
                     list(self.tour0["stops"]) if self.tour0 else [])
        else:
            try:
                t = plan_tour(self.cfg, inputs=res["inputs"])
                self.tour, route = t, (np.asarray(t["points"], np.float64), t["stops"])
            except Exception as e:  # noqa: BLE001 - target removed / unreachable: no tour in this layout
                self.tour, route = None, None
                self.tour_note = f"No inspection tour in this layout: {e}"
        self._nav_layout = self.layout_hash() if self.edits else ""
        self.tour_route = route
        if route is not None:
            self.stops = route[1]
            self.walk_mode = "goto"                                     # force set_walk_mode to rebuild the route
            self.set_walk_mode("tour")
            self.draw_tour_markers()
        self.h_path.visible = route is not None
        self.md_walk.content = f"⚠️ {self.tour_note}" if self.tour_note else ""
        self.update_layout_status()

    def _restore_layout_nav(self) -> None:
        """At startup: saved edits whose layout was applied get that navigation back (layout_changed rebuilds it,
        guarded); unapplied edits, or a rebuild that fails, block walking until Apply layout."""
        self.tour_note = self.nav_error = ""
        if self.edits is None:
            return
        self._nav_layout = self.base_layout
        self.layout_changed()

    def export_edited_ifc(self) -> None:
        from pipeline.ifc_edits import export_edited
        if self.edits is None:
            return
        self.md_ifc.content = "_Writing scene_edited.ifc…_"
        try:
            r = export_edited(self.cfg, self.edits, {oid: m["members"] for oid, m in self.movable.items()})
            self.md_ifc.content = (f"Wrote **{r['file']}** ({r['moved']} moved, {r['removed']} removed, "
                                   f"{r['copies']} copies without asset data). `scene.ifc` is unchanged.")
        except Exception as e:  # noqa: BLE001
            self.md_ifc.content = f"⚠️ Edited IFC export failed: {type(e).__name__}: {e}"

    # --- display modes: Scan / BIM / Overlay ---
    def _load_bim(self) -> None:
        """Validate and tessellate out/scene.ifc in the background; mismatched versions are rejected."""
        try:
            from viewer.bim_layer import BimLayer
            layer = BimLayer(self.cfg, self.nodes, self.revision)
        except Exception as e:  # noqa: BLE001 - the scan viewer works without BIM
            self.bim_note = f"⚠️ **BIM unavailable:** {e}"
            print(f"[viewer] BIM REJECTED: {e}")
            if hasattr(self, "md_bim"):
                self.md_bim.content = self.bim_note
            return
        for oid, m in self.movable.items():
            v, f = layer.local(m["members"], m["c"])
            m["bim"] = (v, f) if len(f) else None
        self.bim = layer
        self.bim_note = (f"BIM: **{len(layer.mesh)}** IFC elements from `scene.ifc` (segmentation "
                         f"`{self.revision['segmentation']}`). Pipes are fitted cylinders, pumps/motors/valves "
                         f"catalog shapes, the rest meshes of the scan points: approximate geometry.")
        if hasattr(self, "md_bim"):
            self.md_bim.content = self.bim_note
        if self.display != "Scan":
            self.set_display(self.display)

    def _ensure_bim_handles(self) -> None:
        if self.h_bim or self.bim is None:
            return
        sc = self.server.scene
        for oid, m in self.bim.mesh.items():
            if oid in self.movable_members:
                continue
            self.h_bim[oid] = sc.add_mesh_simple(f"/bim/{oid}", m["verts"].astype(np.float32),
                                                 m["faces"].astype(np.uint32), color=m["color"], visible=False)

    def set_display(self, mode: str) -> None:
        if mode != "Scan" and self.bim is None:
            if hasattr(self, "md_bim"):
                self.md_bim.content = self.bim_note + ("  \n_Not loaded yet; try again in a moment._"
                                                       if "loading" in self.bim_note else "")
            if self.dd_display.value != "Scan":
                self.dd_display.value = "Scan"
            mode = "Scan"
        self.display = mode
        scan, bim = mode in ("Scan", "Overlay"), mode in ("BIM", "Overlay")
        if bim:
            self._ensure_bim_handles()
        for g, h in self.h_splat.items():
            h.visible = scan and self._group_vis[g]
        for name, it in self.inst.items():
            it["splat"].visible = scan
            if it["bim"] is None and bim:
                self._add_instance_bim(name)
            if it["bim"] is not None:
                it["bim"].visible, it["bim"].wireframe = bim, mode == "Overlay"
        for h in self.h_bim.values():
            h.visible, h.wireframe = bim, mode == "Overlay"
        self._refresh_visit_tint()

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
            with c.atomic():
                if saved:
                    c.camera.position, c.camera.look_at, c.camera.fov = saved
                else:                                   # joined while in POV: overview behind the worker
                    self.overview_camera(c)
        self._saved_cams = {}

    def overview_pose(self, p: np.ndarray | None = None, f: np.ndarray | None = None
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(position, look_at, up) above and behind the worker (or point p), looking where it faces (f)."""
        pc = self.vc["pov"]
        up = np.array([0.0, -1.0, 0.0])                                 # scene up = -y
        p = self.position() if p is None else p
        f = getattr(self, "_facing", self.start_dir) if f is None else f
        pos = p + up * pc.get("overview_height", 1.8) - f * pc.get("overview_back", 0.4)
        look = p + f * pc.get("overview_look_ahead", 4.0) + up * pc.get("overview_look_h", 0.8)
        return pos, look, up

    def overview_camera(self, c: Any) -> None:
        pos, look, up = self.overview_pose()
        c.camera.up_direction = tuple(up)
        c.camera.fov = float(np.radians(self.vc["pov"]["fov_deg"]))
        c.camera.position, c.camera.look_at = tuple(pos), tuple(look)

    def door_pov_pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(eye, look_at, up) of the worker standing at the door, looking straight into the room."""
        pc = self.vc["pov"]
        up = np.array([0.0, -1.0, 0.0])                                 # scene up = -y
        f = self.start_dir
        eye_h = 0.94 * self.rig.height_units if self.rig is not None else self.av["head_h"]
        eye = self.door_p + up * eye_h + f * pc["eye_forward"]
        return eye, eye + f * pc["look_ahead"] - up * pc["look_down"], up

    def set_start_camera(self) -> None:
        """Fixed camera for every new browser / refresh and viser's 'Reset View': the door, looking straight
        into the room (worker's eyes with POV on start, else an overview behind the worker)."""
        if self.vc["pov"].get("on_start", True):
            pos, look, up = self.door_pov_pose()
        else:
            pos, look, up = self.overview_pose(self.door_p, self.start_dir)
        ic = self.server.initial_camera
        ic.up, ic.fov = tuple(up), float(np.radians(self.vc["pov"]["fov_deg"]))
        ic.position, ic.look_at = tuple(pos), tuple(look)

    def reset_to_start(self) -> None:
        """Start state (refresh / new browser): tour route, worker at the door facing straight into the room,
        nothing highlighted, target equipment selected, POV as configured, camera at the start view."""
        self.playing = False
        if self.tour_route is not None:                 # a layout edit can leave no tour: keep its note
            self.set_walk_mode("tour")                  # tour route from the door, dist 0, leaves the target
            self.md_walk.content = ""
        self.clear_stop_highlight()
        tag = self.target["asset"]["tag"]
        opt = self.equip_opt.get(tag)
        if opt and self.dd_walk.value != opt:
            self.dd_walk.value = opt
        if opt and self.dd_equip.value != opt:          # Info tab back to the target, without speaking
            self.dd_equip.value = opt
            self.load_equipment(speak=False)
        pov = bool(self.vc["pov"].get("on_start", True))
        self.cb_walk_pov.value = self.cb_pov.value = pov
        self.h_body.visible = not pov
        self._saved_cams = {}
        self._pov_look = None                           # no smoothing from the old view: snap to the door
        self.update()
        pos, look, up = self.door_pov_pose() if pov else self.overview_pose(self.door_p, self.start_dir)
        fov = float(np.radians(self.vc["pov"]["fov_deg"]))
        for c in self.server.get_clients().values():
            with c.atomic():
                c.camera.up_direction, c.camera.fov = tuple(up), fov
                c.camera.position, c.camera.look_at = tuple(pos), tuple(look)

    def update_pov(self) -> None:
        """Camera at the worker's eyes, looking where the worker walks; at a stop or the target it looks
        at the equipment. Smoothed so turns at corners and stops are a pan, not a jump."""
        pc = self.vc["pov"]
        p = self.position()
        up = np.array([0.0, -1.0, 0.0])                                 # scene up = -y
        f = getattr(self, "_facing", self.tangent())
        eye_h = 0.94 * self.rig.height_units if self.rig is not None else self.av["head_h"]
        eye = p + up * (eye_h + float(self.h_body.position[1])) + f * pc["eye_forward"]
        walking = self.playing and self.dist < self.s[-1] and time.time() >= self.pause_until
        focus = self.face_target if self.face_target is not None and not walking else (
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
        return (f"#### Arrived at {a['tag']}\n"
                f"_3D mapping: {a['mapping_method']} (confidence {a['mapping_confidence']})_\n\n")

    def show_diagnosis(self) -> None:
        if not self.arrived or self._diag is None:
            return
        self.set_object_view(False)
        tag = self.target["asset"]["tag"]
        self.md_copilot.content = f"#### Arrived at {tag}\n\n" + self.equipment_view(tag, self._diag)
        if self._audio:
            self.set_audio(self._audio, f"Copilot briefing · {tag}")

    def on_arrival(self) -> None:
        if self.walk_mode == "tour":
            self.show_tab("Info")
        self.choose_equipment(self.target["asset"]["tag"])
        if self._diag is not None:
            self.show_diagnosis()
            return
        self.md_copilot.content = self.arrival_header() + "_The copilot is preparing your briefing…_"
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
        print(f"[viewer] http://localhost:{self.port}  ({len(self.nodes)} objects, "
              f"path {self.s[-1]:.2f} u)")
        dt = 1.0 / self.vc["fps"]
        self._init_remote()
        last_dist = self.dist
        while True:
            now = time.time()
            self.poll_remote(now)
            if self.playing and now < self.pause_until and self.stop_active is not None:
                self.blink_stop(now)                                   # pausing at a stop
            elif self.playing and self.stop_active is not None and self.stop_active["kind"] != "target":
                self.end_stop()                                       # visual highlight lasts until departure
            if self.playing and now >= self.pause_until:
                nxt = self.stop_s[self.next_stop] if self.next_stop < len(self.stops) else float(self.s[-1])
                self.dist = min(self.dist + float(self.sl_speed.value) * dt, nxt, float(self.s[-1]))
                self.sl_pos.value = round(self.dist, 2)
                self.update()
                if self.next_stop < len(self.stops) and self.dist >= self.stop_s[self.next_stop] - 1e-6:
                    self.reach_stop(self.next_stop)
                if self.dist >= self.s[-1] and self.next_stop >= len(self.stops):
                    self.playing = False                               # end reached and every stop visited
            if self.rig is not None:                                   # walk while moving, else breathe
                if self.dist != last_dist:
                    self.rig.walk(self.dist)
                else:
                    self.rig.idle(now)
                last_dist = self.dist
            if self._reset_pending:                                    # a browser (re)connected
                self._reset_pending = False
                try:
                    self.reset_to_start()
                except Exception as e:  # noqa: BLE001 - a refresh must never stop the viewer
                    print(f"[viewer] reset on connect failed: {type(e).__name__}: {e}")
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
                "  \nIts data is in the **Info** tab." if st["has_asset"]
                else "  \n_No asset record: this object is not in the Excel pack._"))
            return
        self.md_stop.content = stop_text(st, k + 1, len(self.stops))
        self.refresh_steps(k)
        if st["kind"] != "target":
            tc = self.cfg["tour"]
            if tc.get("brief_at_stops", True) and st.get("has_asset", True) and st["tag"] in self.equip_opt:
                self.md_stop.content += "  \n_Copilot summary playing…_"
                self.pause_until = time.time() + float(tc.get("brief_wait_s", 12))   # until the audio is ready
                self.brief_stop(st["tag"])
            else:
                self.pause_until = time.time() + float(self.tour["pause_s"])

    def brief_stop(self, tag: str) -> None:
        """Select the stop's equipment (Info tab follows) and play its spoken copilot summary;
        the pause then lasts as long as the audio (the target P-002 has its own arrival briefing)."""
        opt = self.equip_opt[tag]
        if self.dd_walk.value != opt:
            self.dd_walk.value = opt
        if self.dd_equip.value != opt:
            self.dd_equip.value = opt
        self.load_equipment(speak=True, on_spoken=lambda audio: self._stop_brief_ready(tag, audio))
        self.show_tab("Info")

    def _stop_brief_ready(self, tag: str, audio: bytes | None) -> None:
        if self.walk_mode != "tour" or not self.stop_active or self.stop_active.get("tag") != tag:
            return                                                  # the worker already moved on / reset
        tc = self.cfg["tour"]
        hold = audio_seconds(audio, self.cfg) + float(tc.get("brief_gap_s", 1.0)) if audio else float(tc["pause_s"])
        hold = max(hold, float(tc["pause_s"]))
        if self.playing:
            self.pause_until = time.time() + hold
        else:                                                       # tour paused meanwhile: resume will use it
            self.pause_left = hold

    def prefetch_stop_briefings(self) -> None:
        """Warm the spoken summaries of all tour stops (ElevenLabs once per text; cached after)."""
        tags = [st["tag"] for st in self.stops if st.get("kind") != "target" and st["tag"] in self.equip_opt]

        def work() -> None:
            for tag in tags:
                try:
                    briefing(diagnose(tag, None, self.cfg, allow_paid=False), tag, self.cfg,
                             opener=self.summary_opener(tag))
                except Exception as e:  # noqa: BLE001 - the stop falls back to a plain pause
                    print(f"[viewer] stop briefing {tag} failed: {type(e).__name__}: {e}")
        if tags and self.cfg["tour"].get("brief_at_stops", True):
            threading.Thread(target=work, daemon=True).start()

    # --- routes: inspection tour vs. go to one equipment ---
    def _build_chat_gui(self, g: Any) -> None:
        """Chat: quick commands (instant, free) -> Claude Code headless with our MCP tools."""
        g.add_html(ui.lead("<strong>Ask about the room or give a command.</strong> Questions go to Claude with "
                           "the copilot's tools; commands like <em>take me to M-006</em> or <em>start tour</em> run "
                           "instantly. Type <em>help</em> for the list."))
        self.tx_chat = g.add_text("Question", "Which assets need attention?")
        b_chat = g.add_button("Ask", icon="send")
        self.md_chat_status = g.add_markdown("", visible=False)
        self.md_claude = g.add_markdown("_Checking Claude Code…_")
        self.md_chat = g.add_markdown("")
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
                self.md_chat_status.content = (f"**Claude is working** (copilot tools via MCP) · "
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
        badge = "Quick command" if route == "quick" else f"Claude · {time.time() - t0:.0f} s"
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
        if self.nav_stale():
            return None, self.nav_stale()
        self.get_planner()
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
            if self.nav_stale():
                return {"ok": False, "message": self.nav_stale().replace("**", "")}
            t = plan_tour(self.cfg, list(c.get("stops") or []), inputs=self.nav_inputs)
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

    def get_planner(self) -> Any:
        """Walk planner on the navigation of the shown layout (out/ files, or the applied edited layout)."""
        if self.planner is None:
            from pipeline.tour import WalkPlanner
            self.planner = WalkPlanner(self.cfg, inputs=self.nav_inputs)
        return self.planner

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
        self.pause_left = 0.0
        self.end_stop()
        self.update_visit_highlight(self.position())
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
        self.refresh_steps()
        self.update()

    def go_to(self, node: dict) -> None:
        """Plan from the worker's current position to one object and start walking (Go to mode)."""
        why = self.nav_stale()
        if why or node.get("_layout_removed") or (node["id"] in self.inst and self.inst[node["id"]]["removed"]):
            self.playing = False
            self.md_walk.content = f"⚠️ {why or 'This object is removed in the current layout.'}"
            return
        if self.walk_mode != "goto":
            self.set_walk_mode("goto")
        p = self.position()
        plan = self.get_planner().to_object(float(p[0]), float(p[2]), node)
        if not plan["ok"]:
            self.md_walk.content = f"⚠️ {plan['message']}"
            return
        self.set_route(plan["points"], [goto_record(node, len(plan["points"]) - 1)])
        dest = node["asset"]["tag"] if node.get("asset") else f"{node['id']} ({node['label']})"
        note = "" if node.get("asset") else "  \n_No asset record: this object is not in the Excel pack._"
        if node.get("asset"):
            self.choose_equipment(node["asset"]["tag"])
        self.md_walk.content = (f"🚶 Walking to **{dest}**: "
                                f"{plan['length'] * self.cfg['scene']['units_to_m']:.1f} m, "
                                f"min clearance {plan['min_clearance']:.2f}{note}")
        self.playing = True

    def highlight_stop(self, st: dict) -> None:
        """Solid health outline and a pale tint applied directly to the actual equipment."""
        n = self.lookup(st["object_id"])
        self.clear_visit_highlight()
        _, title = visit_style(n)
        lo, hi = self._draw_stop_box(n)
        self.h_stop_label = self.server.scene.add_label(
            "/stop/label", title,
            position=(float((lo[0] + hi[0]) / 2), float(lo[1]) - 0.15, float((lo[2] + hi[2]) / 2)),
            anchor="bottom-center", font_screen_scale=self.vc["stop_label_scale"])
        self.stop_active = st
        self.visit_active = st
        self.visit_arrival_gap = self.box_gap(n, self.position())
        self._set_visit_tint(n)
        self.h_highlight.visible = False
        self._sync_visit_labels()
        self.face_target = np.array(n["center"], np.float64)
        self.pose_avatar(self.position(), moving=False)

    def _draw_stop_box(self, n: dict) -> tuple[np.ndarray, np.ndarray]:
        """Strong red/green outline around the object where it is shown now (follows a moved object)."""
        col, title = visit_style(n)
        lo, hi = self.live_bounds(n) if n["id"] in getattr(self, "inst", {}) else             (np.array(n["bbox_min"]), np.array(n["bbox_max"]))
        pad = self.vc.get("stop_outline_pad", .01)
        segs = box_edges(lo - pad, hi + pad)
        self.h_stop_box.points = segs
        self.h_stop_box.colors = np.tile(np.array(col, np.uint8), (len(segs), 2, 1))
        self.h_stop_box.visible = True
        if self.h_stop_label is not None:
            self.h_stop_label.position = (float((lo[0] + hi[0]) / 2), float(lo[1]) - 0.15, float((lo[2] + hi[2]) / 2))
        return lo, hi

    def blink_stop(self, now: float) -> None:
        # Existing playback callbacks use this hook; the health highlight is now steady.
        self.h_stop_box.visible = self.visit_active is not None
        if self.h_stop_fill is not None:
            self.h_stop_fill.visible = self.visit_active is not None

    def clear_visit_highlight(self) -> None:
        self.h_stop_box.visible = False
        for h in (self.h_stop_fill, self.h_stop_label):
            if h is not None:
                h.remove()
        self.h_stop_fill = self.h_stop_label = None
        self.visit_active = None
        self.visit_arrival_gap = 0.0
        self._set_visit_tint(None)
        self._sync_visit_labels()

    def end_stop(self) -> None:
        """Release briefing/facing state without removing the equipment's health indication."""
        self.stop_active = None
        self.face_target = None

    def clear_stop_highlight(self) -> None:
        """Explicit reset/replay clears both playback and visual state."""
        self.end_stop()
        self.clear_visit_highlight()


if __name__ == "__main__":
    Sim(load_config()).run()
