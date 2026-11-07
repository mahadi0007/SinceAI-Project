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
from pipeline.common import load_config, out_path, repo_path
from pipeline.load_splat import covariances, load_splat

Cfg = dict[str, Any]
IDLE_TEXT = "_Press Play: the copilot briefs the worker once they stand in front of the target._"
EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


# ---------- data ----------

def load_scene(cfg: Cfg) -> tuple[dict, dict]:
    with open(out_path(cfg, "scene_graph.json"), encoding="utf-8") as f:
        sg = json.load(f)
    with open(out_path(cfg, "path.json"), encoding="utf-8") as f:
        path = json.load(f)
    return sg, path


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
        md += ["", f"_Gemini's low-confidence guess: {v.get('label')} ({v.get('confidence')})_"]
    a = n["asset"]
    if a:
        md += ["", "| Tag | Class | Health | Status | Urgency |", "|---|---|---|---|---|",
               f"| **{a['tag']}** | {a['asset_class']} | **{a['health_score']}** | {a['condition_state']} | "
               f"{a['urgency']} |", "",
               f"Issue: **{a['issue_code']}**: {a['issue_summary']}  ",
               f"Next check: {a['recommended_next_check']}  ",
               f"_3D mapping: {a['mapping_method']} (confidence {a['mapping_confidence']})_"]
    else:
        md += ["", "_No asset record in the Excel pack (not mapped)._"]
    return "\n".join(md)


# ---------- viewer ----------

class Sim:
    def __init__(self, cfg: Cfg) -> None:
        self.cfg, self.vc = cfg, cfg["viewer"]
        self.sg, path = load_scene(cfg)
        self.nodes = [n for n in self.sg["objects"] if n["kind"] == "object"]
        self.pts = np.array(path["points"], np.float64)
        seg = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        self.s = np.concatenate([[0], np.cumsum(seg)])
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
        self.server = viser.ViserServer(port=self.vc["port"])
        self.server.scene.set_up_direction(self.cfg["scene"]["up_direction"])
        self._labels: list[Any] = []
        self.av = scaled_avatar(self.vc["avatar"])
        self._build_scene()
        self._build_gui()
        self.server.scene.on_click()(self.on_scene_click)

    # --- scene ---
    def _build_scene(self) -> None:
        sc = self.server.scene
        d = load_splat(str(repo_path(self.cfg["paths"]["ply"])))
        if self.vc["render_splats"]:
            sc.add_gaussian_splats("/splat", centers=d["xyz"], rgbs=d["rgb"],
                                   opacities=d["opacity"][:, None],
                                   covariances=covariances(d["scales"], d["quats_wxyz"]).astype(np.float32))
        else:
            sc.add_point_cloud("/splat", d["xyz"], (d["rgb"] * 255).astype(np.uint8),
                               point_size=self.vc["point_size"])
        segs, cols = edges_for(self.nodes, self.vc["colors"])
        self.h_boxes = sc.add_line_segments("/boxes", segs, cols, thickness=self.vc["box_thickness"],
                                            thickness_units="screen")
        lift = np.array([0, -self.vc["path_lift"], 0])                # up = -y
        p = self.pts + lift
        sc.add_line_segments("/path", np.stack([p[:-1], p[1:]], 1).astype(np.float32),
                             tuple(self.vc["path_color"]), thickness=4.0, thickness_units="screen")
        t = self.target
        sc.add_label("/target_label", f"TARGET {t['asset']['tag']} · {t['asset']['issue_code']}",
                     position=(t["center"][0], t["bbox_min"][1] - 0.1, t["center"][2]),
                     anchor="bottom-center")
        self.h_highlight = sc.add_line_segments("/highlight", *edges_for([], self.vc["colors"]),
                                                thickness=self.vc["highlight_thickness"],
                                                thickness_units="screen")
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
        if self.arrived:                                                # face the target
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
        self.cb_boxes = g.add_checkbox("Show all frames", True)
        self.dd_labels = g.add_dropdown("Labels", list(LABEL_MODES), initial_value=self.vc["label_mode"])
        with g.add_folder("Selected object (click in the scene)"):
            self.md_sel = g.add_markdown("_Click any object in the 3D view to inspect it._")
            self.img_sel = g.add_image(np.full((8, 24, 3), 235, np.uint8), visible=False)
            b_fly = g.add_button("Fly to object", visible=True)
            self.b_identify = g.add_button("Ask Gemini what this is")
            self.md_identify = g.add_markdown("")
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
            self.md_identify.content = f"_Asking Gemini about {n['id']}…_"

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
                self.dist = 0.0
            self.playing = not self.playing
            if self.playing:
                self.prefetch_diagnosis()

        @self.b_ask.on_click
        def _(_e: Any) -> None:
            tag, q = self.dd_asset.value, self.tx_question.value
            self.run_job(tag, q, lambda d: setattr(self.md_copilot, "content", to_markdown(d)))

        @b_reset.on_click
        def _(_e: Any) -> None:
            self.playing, self.dist = False, 0.0
            self.leave_target()
            self.update()

        @self.sl_pos.on_update
        def _(_e: Any) -> None:
            if not self.playing:
                self.dist = float(self.sl_pos.value)
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

    # --- per-frame ---
    def position(self) -> np.ndarray:
        return np.array([np.interp(self.dist, self.s, self.pts[:, k]) for k in range(3)])

    def update(self) -> None:
        p = self.position()
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

    def run_job(self, tag: str, question: str | None, on_done: Any) -> None:
        """Run diagnose() in a thread; show progress bar + elapsed seconds until it returns."""
        if self._busy:
            return
        self._busy = True
        t0, expected = time.time(), self.cfg["copilot"]["expected_latency_s"]
        q = f" · _{question}_" if question else ""

        def tick() -> None:
            while self._busy:
                el = time.time() - t0
                self.set_loading(True, f"⏳ **Gemini is analysing {tag}**{q}\n\n"
                                       f"Evidence pack → model → citation check · **{el:.0f} s** "
                                       f"(usually ~{expected} s)", min(95.0, 100 * el / expected))
                time.sleep(0.5)

        def work() -> None:
            try:
                d = diagnose(tag, question, self.cfg)
            finally:
                self._busy = False
                self.set_loading(False)
            on_done(d)

        threading.Thread(target=tick, daemon=True).start()
        threading.Thread(target=work, daemon=True).start()

    def prefetch_diagnosis(self) -> None:
        """Silently prepare the target diagnosis + spoken briefing while the avatar walks
        (Featherless ~60 s, TTS a few s). Nothing is shown until the worker stands at the target;
        if it is still running on arrival, the panel says so and the briefing plays when ready."""
        if self._diag is not None or self._prefetching:
            return
        self._prefetching = True
        tag = self.target["asset"]["tag"]

        def work() -> None:
            try:
                d = diagnose(tag, None, self.cfg)
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
        print(f"[viewer] http://localhost:{self.vc['port']}  ({len(self.nodes)} objects, "
              f"path {self.s[-1]:.2f} u)")
        dt = 1.0 / self.vc["fps"]
        while True:
            if self.playing:
                self.dist = min(self.dist + float(self.sl_speed.value) * dt, float(self.s[-1]))
                self.sl_pos.value = round(self.dist, 2)
                self.update()
                if self.dist >= self.s[-1]:
                    self.playing = False
            time.sleep(dt)


if __name__ == "__main__":
    Sim(load_config()).run()
