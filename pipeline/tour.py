"""Stage 5: inspection tour door -> warning assets -> faulty target -> out/tour.json, out/tour.png

Stops come automatically from the Excel Current_State (via scene_graph.json): mapped assets whose
condition is in tour.conditions, most urgent first (condition rank, then lowest health), at most
tour.max_stops. Unreachable stops are skipped with a reason. The visiting order minimises the
walked distance (exact search over all orders; the tour is small); the tour always ends at the
navigation target (the faulty asset). Every leg is A* on the same inflated grid as navgrid.py.

Inputs : out/navgrid.npz, out/path.json, out/scene_graph.json
Outputs: out/tour.json, out/tour.png
Run    : python -m pipeline.tour
"""
from __future__ import annotations

import itertools
import json
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from pipeline.common import load_config, out_path
from pipeline.navgrid import (Grid, approach_cells, astar, box_distance, dijkstra, nearest_walkable,
                              resample, shortcut)

Cfg = dict[str, Any]
Cell = tuple[int, int]


# ---------- inputs ----------

def grid_from_npz(g: Any) -> Grid:
    """Rebuild the navgrid Grid geometry (origin, cell, size) from out/navgrid.npz."""
    grid = Grid.__new__(Grid)
    grid.cell = float(g["cell"])
    grid.x0, grid.z0 = (float(v) for v in g["origin"])
    grid.nx, grid.nz = g["walkable"].shape
    return grid


def load_inputs(cfg: Cfg) -> tuple[Grid, dict, dict, dict]:
    g = np.load(out_path(cfg, "navgrid.npz"))
    with open(out_path(cfg, "path.json"), encoding="utf-8") as f:
        path = json.load(f)
    with open(out_path(cfg, "scene_graph.json"), encoding="utf-8") as f:
        sg = json.load(f)
    layers = {k: g[k] for k in ("walkable", "inflated", "clearance", "obstacles", "floor")}
    return grid_from_npz(g), layers, path, sg


def cell_of(grid: Grid, x: float, z: float) -> Cell:
    ix, iz = grid.index(np.array([x]), np.array([z]))
    return int(ix[0]), int(iz[0])


# ---------- stop selection ----------

def candidate_stops(sg: dict, target_id: str, tc: Cfg) -> list[dict]:
    """Mapped warning assets from Excel status, most urgent first (excluding the end target)."""
    rank = {c: i for i, c in enumerate(tc["conditions"])}
    nodes = [n for n in sg["objects"] if n["asset"] and n["id"] != target_id
             and n["asset"]["condition_state"] in rank]
    return sorted(nodes, key=lambda n: (rank[n["asset"]["condition_state"]], n["asset"]["health_score"]))


def stop_goal(n: dict, grid: Grid, dist: np.ndarray, reach: float) -> Cell | None:
    """Reachable cell closest to the asset's box face, or None if nothing is within reach."""
    cells = approach_cells(n, grid, dist, reach)
    return min(cells, key=lambda c: box_distance(n, *grid.center(*c))) if cells else None


# ---------- ordering ----------

def best_order(start: Cell, stops: list[Cell], end: Cell, walk: np.ndarray) -> tuple[list[int], float]:
    """Exact shortest visiting order (start -> all stops -> end) on geodesic distances."""
    pts = [start] + stops + [end]
    d = np.zeros((len(pts), len(pts)))
    for i, p in enumerate(pts):
        field = dijkstra(walk, p)
        d[i] = [field[q] for q in pts]
    best, best_len = list(range(len(stops))), np.inf
    for perm in itertools.permutations(range(len(stops))):
        seq = [0] + [k + 1 for k in perm] + [len(pts) - 1]
        length = sum(d[a, b] for a, b in zip(seq, seq[1:]))
        if length < best_len:
            best, best_len = list(perm), float(length)
    return best, best_len


# ---------- path ----------

def leg_points(walk: np.ndarray, grid: Grid, a: Cell, b: Cell) -> np.ndarray:
    cells = shortcut(walk, astar(walk, a, b))
    return np.array([grid.center(*c) for c in cells])


def build_path(walk: np.ndarray, grid: Grid, seq: list[Cell], step: float) -> tuple[np.ndarray, list[int]]:
    """Concatenate legs; return x-z samples and the sample index where each waypoint is reached."""
    xz = [np.array([grid.center(*seq[0])])]
    marks = [0]
    for a, b in zip(seq, seq[1:]):
        leg = resample(leg_points(walk, grid, a, b), step) if a != b else np.array([grid.center(*b)])
        xz.append(leg[1:] if len(leg) > 1 else leg)
        marks.append(sum(len(p) for p in xz) - 1)
    return np.concatenate(xz), marks


def validate(xz: np.ndarray, grid: Grid, layers: dict) -> float:
    ix, iz = grid.index(xz[:, 0], xz[:, 1])
    assert layers["walkable"][ix, iz].all(), "tour sample in non-walkable cell"
    assert not layers["inflated"][ix, iz].any(), "tour sample inside inflated obstacle"
    return float(layers["clearance"][ix, iz].min())


# ---------- main ----------

def stop_record(n: dict, sample_index: int, kind: str) -> dict:
    a = n["asset"]
    return {"kind": kind, "tag": a["tag"], "object_id": n["id"], "label": n["label"],
            "asset_class": a["asset_class"], "condition_state": a["condition_state"],
            "health_score": a["health_score"], "issue_code": a["issue_code"],
            "center": n["center"], "sample_index": sample_index}


def plan_tour(cfg: Cfg, stops: list[str] | None = None) -> dict:
    """Plan the tour. stops=None -> automatic from Excel status; otherwise these tags in this order."""
    tc, floor_y = cfg["tour"], cfg["scene"]["floor_y"]
    grid, layers, path, sg = load_inputs(cfg)
    walk = layers["walkable"]
    start = nearest_walkable(walk, cell_of(grid, path["start"][0], path["start"][2]))
    end = nearest_walkable(walk, cell_of(grid, path["goal"][0], path["goal"][2]))
    dist = dijkstra(walk, start)
    by_id = {n["id"]: n for n in sg["objects"]}
    target = by_id[path["target_object"]]

    mode = "auto_excel" if stops is None else "manual"
    if stops is None:
        cands = candidate_stops(sg, target["id"], tc)
    else:
        by_tag = {n["asset"]["tag"]: n for n in sg["objects"] if n["asset"]}
        cands = [by_tag[t] for t in stops if t in by_tag and by_tag[t]["id"] != target["id"]]
    chosen, goals, skipped = [], [], []
    for n in cands:
        if len(chosen) >= tc["max_stops"]:
            skipped.append({"tag": n["asset"]["tag"], "reason": f"max_stops={tc['max_stops']} reached"})
            continue
        g = stop_goal(n, grid, dist, tc["reach"])
        if g is None:
            skipped.append({"tag": n["asset"]["tag"], "reason": f"no walkable floor within {tc['reach']} u"})
            continue
        chosen.append(n)
        goals.append(g)

    order = best_order(start, goals, end, walk)[0] if mode == "auto_excel" else list(range(len(goals)))
    seq = [start] + [goals[k] for k in order] + [end]
    xz, marks = build_path(walk, grid, seq, cfg["navgrid"]["resample"])
    min_clear = validate(xz, grid, layers)
    length = float(np.linalg.norm(np.diff(xz, axis=0), axis=1).sum())

    stops_out = [stop_record(chosen[k], marks[i + 1], "stop") for i, k in enumerate(order)]
    stops_out.append(stop_record(target, marks[-1], "target"))
    pts3 = np.stack([xz[:, 0], np.full(len(xz), floor_y), xz[:, 1]], 1)
    return {"mode": mode, "stops": stops_out, "skipped": skipped,
            "points": pts3.round(4).tolist(), "length": round(length, 3),
            "min_clearance": round(min_clear, 4), "pause_s": tc["pause_s"]}


def plot_tour(cfg: Cfg, tour: dict) -> None:
    grid, layers, _, sg = load_inputs(cfg)
    img = np.ones((grid.nz, grid.nx, 3))
    img[layers["floor"].T] = (0.80, 0.92, 0.80)
    img[(layers["inflated"] & layers["floor"]).T] = (0.98, 0.85, 0.65)
    img[layers["obstacles"].T] = (0.25, 0.25, 0.30)
    fig, ax = plt.subplots(figsize=(13, 10))
    ext = (grid.x0, grid.x0 + grid.nx * grid.cell, grid.z0, grid.z0 + grid.nz * grid.cell)
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
    p = np.array(tour["points"])
    ax.plot(p[:, 0], p[:, 2], "-", color="crimson", lw=2)
    ax.plot(p[0, 0], p[0, 2], "o", color="green", ms=11, label="start (door)")
    by_id = {n["id"]: n for n in sg["objects"]}
    for i, s in enumerate(tour["stops"], 1):
        n = by_id[s["object_id"]]
        lo, hi = n["bbox_min"], n["bbox_max"]
        col = "red" if s["kind"] == "target" else "orange"
        ax.add_patch(Rectangle((lo[0], lo[2]), hi[0] - lo[0], hi[2] - lo[2], fill=False, ec=col, lw=2))
        q = p[s["sample_index"]]
        ax.plot(q[0], q[2], "*" if s["kind"] == "target" else "o", color=col, ms=16 if s["kind"] == "target" else 9)
        ax.annotate(f"{i}. {s['tag']} ({s['condition_state']}, {s['health_score']})", (q[0], q[2]),
                    xytext=(6, 6), textcoords="offset points", fontsize=9, color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
    ax.set_title(f"Inspection tour ({tour['mode']}): {len(tour['stops']) - 1} stops + target, "
                 f"{tour['length']:.2f} u (~{tour['length'] * cfg['scene']['units_to_m']:.1f} m), "
                 f"min clearance {tour['min_clearance']:.2f}")
    ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_aspect("equal"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(out_path(cfg, "tour.png"), dpi=130); plt.close(fig)


def run(cfg: Cfg) -> dict:
    tour = plan_tour(cfg)
    with open(out_path(cfg, "tour.json"), "w", encoding="utf-8") as f:
        json.dump(tour, f, indent=1)
    plot_tour(cfg, tour)
    u2m = cfg["scene"]["units_to_m"]
    print(f"[tour] mode={tour['mode']}  length {tour['length']:.2f} u (~{tour['length'] * u2m:.1f} m)  "
          f"min clearance {tour['min_clearance']:.3f}  samples {len(tour['points'])}")
    for i, s in enumerate(tour["stops"], 1):
        print(f"  {i}. {s['tag']:7s} {s['asset_class']:16s} {s['condition_state']:18s} health {s['health_score']:3d} "
              f"{s['issue_code']:20s} {s['object_id']}  ({s['kind']})")
    for s in tour["skipped"]:
        print(f"  skipped {s['tag']}: {s['reason']}")
    print(f"[out] {out_path(cfg, 'tour.json')}, {out_path(cfg, 'tour.png')}")
    return tour


if __name__ == "__main__":
    run(load_config())
