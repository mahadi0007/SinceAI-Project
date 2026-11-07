"""Stage 3: x-z occupancy grid + A* + LOS smoothing -> out/navgrid.npz, out/path.json, out/navgrid.png

Inputs: out/segment_points.npz, out/assets.json (from segment.py).
Walkable = floor cells (dilated) minus obstacles inflated by the avatar radius.
The target object is taken from config (navgrid.target_object) or auto-picked:
a floor-standing, pump-sized object reachable at the far end of the walk.
mapping.py later forces the faulty asset (P-002) onto this object.
"""
from __future__ import annotations

import heapq
import json
import math
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import ndimage

from pipeline.common import load_config, out_path

Cfg = dict[str, Any]
Cell = tuple[int, int]   # (ix, iz)
NEIGH = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


# ---------- inputs ----------

def load_inputs(cfg: Cfg) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    d = np.load(out_path(cfg, "segment_points.npz"))
    p = out_path(cfg, "assets_labeled.json")
    with open(p if p.exists() else out_path(cfg, "assets.json"), encoding="utf-8") as f:
        assets = json.load(f)
    return d["xyz"].astype(np.float64), d["labels"], assets


class Grid:
    """x-z grid; cell (ix, iz) covers [x0 + ix*c, x0 + (ix+1)*c)."""

    def __init__(self, xyz: np.ndarray, cell: float, pad: int) -> None:
        self.cell = cell
        self.x0 = xyz[:, 0].min() - pad * cell
        self.z0 = xyz[:, 2].min() - pad * cell
        self.nx = int(math.ceil((xyz[:, 0].max() - self.x0) / cell)) + pad + 1
        self.nz = int(math.ceil((xyz[:, 2].max() - self.z0) / cell)) + pad + 1

    def index(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return ((x - self.x0) / self.cell).astype(int), ((z - self.z0) / self.cell).astype(int)

    def center(self, ix: float, iz: float) -> tuple[float, float]:
        return self.x0 + (ix + 0.5) * self.cell, self.z0 + (iz + 0.5) * self.cell

    def count(self, pts: np.ndarray) -> np.ndarray:
        g = np.zeros((self.nx, self.nz), np.int32)
        ix, iz = self.index(pts[:, 0], pts[:, 2])
        np.add.at(g, (ix, iz), 1)
        return g


# ---------- grid layers ----------

def structure_ids(assets: list[dict], names: list[str]) -> list[int]:
    return [i for i, a in enumerate(assets) if a.get("name") in names]


def floor_mask(grid: Grid, xyz: np.ndarray, labels: np.ndarray, assets: list[dict],
               obstacles: np.ndarray, nc: Cfg) -> np.ndarray:
    floor_ids = structure_ids(assets, ["floor"])
    observed = grid.count(xyz[np.isin(labels, floor_ids)]) > 0
    floor = ndimage.binary_dilation(observed, iterations=nc["floor_dilate"])
    if nc["floor_fill"] == "footprint":
        # assume floor continues under the scanned area: fill the scene footprint
        footprint = ndimage.binary_closing(grid.count(xyz) > 0, iterations=nc["footprint_close"])
        footprint = ndimage.binary_fill_holes(footprint)
        floor |= footprint & ~obstacles
    return floor


def obstacle_mask(grid: Grid, xyz: np.ndarray, labels: np.ndarray, assets: list[dict],
                  floor_y: float, nc: Cfg) -> np.ndarray:
    h = floor_y - xyz[:, 1]                                   # up = -y
    skip = structure_ids(assets, ["floor", "ceiling"])
    m = (h > nc["obstacle_min_h"]) & (h < nc["obstacle_max_h"]) & ~np.isin(labels, skip)
    return grid.count(xyz[m]) >= nc["obstacle_min_points"]


def inflate(obstacles: np.ndarray, radius: float, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (inflated mask, clearance in units = distance to nearest obstacle cell)."""
    clearance = ndimage.distance_transform_edt(~obstacles) * cell
    return clearance < radius, clearance


def largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


# ---------- search ----------

def _step(d: tuple[int, int]) -> float:
    return math.sqrt(2.0) if d[0] and d[1] else 1.0


def dijkstra(walk: np.ndarray, start: Cell) -> np.ndarray:
    dist = np.full(walk.shape, np.inf)
    dist[start] = 0.0
    pq = [(0.0, start)]
    while pq:
        d, (i, j) = heapq.heappop(pq)
        if d > dist[i, j]:
            continue
        for di, dj in NEIGH:
            a, b = i + di, j + dj
            if 0 <= a < walk.shape[0] and 0 <= b < walk.shape[1] and walk[a, b]:
                nd = d + _step((di, dj))
                if nd < dist[a, b]:
                    dist[a, b] = nd
                    heapq.heappush(pq, (nd, (a, b)))
    return dist


def astar(walk: np.ndarray, start: Cell, goal: Cell) -> list[Cell]:
    def h(c: Cell) -> float:
        dx, dz = abs(c[0] - goal[0]), abs(c[1] - goal[1])
        return max(dx, dz) + (math.sqrt(2) - 1) * min(dx, dz)

    g = {start: 0.0}
    came: dict[Cell, Cell] = {}
    pq = [(h(start), start)]
    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        for di, dj in NEIGH:
            nb = (cur[0] + di, cur[1] + dj)
            if 0 <= nb[0] < walk.shape[0] and 0 <= nb[1] < walk.shape[1] and walk[nb]:
                ng = g[cur] + _step((di, dj))
                if ng < g.get(nb, math.inf):
                    g[nb], came[nb] = ng, cur
                    heapq.heappush(pq, (ng + h(nb), nb))
    raise RuntimeError(f"A*: no path from {start} to {goal}")


def bresenham(a: Cell, b: Cell) -> list[Cell]:
    (x0, y0), (x1, y1) = a, b
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err, out = dx + dy, []
    while True:
        out.append((x0, y0))
        if (x0, y0) == (x1, y1):
            return out
        e2 = 2 * err
        if e2 >= dy:
            err += dy; x0 += sx
        if e2 <= dx:
            err += dx; y0 += sy


def line_free(walk: np.ndarray, a: Cell, b: Cell) -> bool:
    return all(walk[c] for c in bresenham(a, b))


def shortcut(walk: np.ndarray, path: list[Cell]) -> list[Cell]:
    """Greedy line-of-sight smoothing: jump to the farthest visible waypoint."""
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_free(walk, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def resample(pts: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.arange(0, s[-1], step).tolist() + [s[-1]]
    return np.stack([np.interp(t, s, pts[:, k]) for k in range(pts.shape[1])], 1)


# ---------- start / target ----------

def pick_start(walk: np.ndarray, grid: Grid, nc: Cfg) -> Cell:
    if nc["start"] is not None:
        ix, iz = grid.index(np.array([nc["start"][0]]), np.array([nc["start"][1]]))
        return nearest_walkable(walk, (int(ix[0]), int(iz[0])))
    cells = np.argwhere(walk)
    key = cells[:, 1] if nc["auto_start"] == "min_z" else cells[:, 0] * (1 if nc["auto_start"] == "max_x" else -1)
    c = cells[np.argmin(key)] if nc["auto_start"] == "min_z" else cells[np.argmax(key)]
    return int(c[0]), int(c[1])


def nearest_walkable(walk: np.ndarray, c: Cell) -> Cell:
    cells = np.argwhere(walk)
    k = np.argmin(((cells - np.array(c)) ** 2).sum(1))
    return int(cells[k][0]), int(cells[k][1])


def approach_cells(obj: dict, grid: Grid, dist: np.ndarray, reach: float) -> list[Cell]:
    """Reachable cells within `reach` of the object's x-z box."""
    lo, hi = np.array(obj["bbox_min"]), np.array(obj["bbox_max"])
    ix0, iz0 = grid.index(np.array([lo[0] - reach]), np.array([lo[2] - reach]))
    ix1, iz1 = grid.index(np.array([hi[0] + reach]), np.array([hi[2] + reach]))
    out = []
    for i in range(max(int(ix0[0]), 0), min(int(ix1[0]) + 1, grid.nx)):
        for j in range(max(int(iz0[0]), 0), min(int(iz1[0]) + 1, grid.nz)):
            if np.isfinite(dist[i, j]):
                out.append((i, j))
    return out


def reachable_any(obj: dict, grid: Grid, dist: np.ndarray, reach: float) -> bool:
    return bool(approach_cells(obj, grid, dist, reach))


def box_distance(obj: dict, x: float, z: float) -> float:
    lo, hi = obj["bbox_min"], obj["bbox_max"]
    dx = max(lo[0] - x, 0, x - hi[0])
    dz = max(lo[2] - z, 0, z - hi[2])
    return math.hypot(dx, dz)


def pick_target(assets: list[dict], grid: Grid, dist: np.ndarray, nc: Cfg) -> tuple[dict, Cell]:
    """Configured object, or the farthest reachable floor-standing pump-sized object."""
    tc = nc["target"]
    reach, by_label = tc["reach"], False
    if tc["object_id"]:
        cands = [a for a in assets if a["id"] == tc["object_id"]]
    else:
        # prefer objects Gemini vision confidently labelled as the target's class; else geometry rule
        cands = [a for a in assets if a.get("label") in tc["prefer_labels"]
                 and a.get("label_source") == "gemini_vision" and "overhead" not in a["tags"]
                 and a.get("label_confidence", 0) >= tc["prefer_min_conf"]
                 and reachable_any(a, grid, dist, tc["label_reach"])]
        if cands:
            reach, by_label = tc["label_reach"], True
            print(f"[target] {len(cands)} reachable objects labelled {tc['prefer_labels']}: "
                  + ", ".join(f"{a['id']} ({a['label_confidence']})" for a in cands))
        else:
            cands = [a for a in assets if a["kind"] == "object" and "overhead" not in a["tags"]
                     and a["hint"] != "pipe segment" and a["n_points"] >= tc["min_points"]
                     and a["features"]["height_bottom"] <= tc["max_height_bottom"]
                     and max(a["extent"]) <= tc["max_extent"]]
    best: tuple[tuple[float, float], dict, Cell] | None = None
    for a in cands:
        cells = approach_cells(a, grid, dist, reach)
        if not cells:
            continue
        # goal = approach cell closest to the box face
        goal = min(cells, key=lambda c: box_distance(a, *grid.center(*c)))
        # labelled: most confident label first, then farthest walk; else farthest walk
        score = (a.get("label_confidence", 0) if by_label else 0.0, float(dist[goal]))
        if best is None or score > best[0]:
            best = (score, a, goal)
    if best is None:
        raise RuntimeError("no reachable target object")
    print(f"[target] {best[1]['id']} hint={best[1]['hint']} n={best[1]['n_points']} "
          f"center={np.round(best[1]['center'], 2).tolist()} from {len(cands)} candidates")
    return best[1], best[2]


# ---------- plot ----------

def plot_grid(grid: Grid, floor: np.ndarray, obstacles: np.ndarray, inflated: np.ndarray,
              path: np.ndarray, assets: list[dict], target: dict, title: str, out) -> None:
    img = np.ones((grid.nz, grid.nx, 3))
    img[floor.T] = (0.80, 0.92, 0.80)
    img[(inflated & floor).T] = (0.98, 0.85, 0.65)
    img[obstacles.T] = (0.25, 0.25, 0.30)
    fig, ax = plt.subplots(figsize=(16, 7))
    ext = (grid.x0, grid.x0 + grid.nx * grid.cell, grid.z0, grid.z0 + grid.nz * grid.cell)
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
    for a in assets:
        if a["kind"] == "object":
            lo, hi = a["bbox_min"], a["bbox_max"]
            is_t = a["id"] == target["id"]
            ax.add_patch(Rectangle((lo[0], lo[2]), hi[0] - lo[0], hi[2] - lo[2], fill=False,
                                   ec="red" if is_t else (0.4, 0.4, 0.9), lw=2 if is_t else 0.3,
                                   alpha=1 if is_t else 0.5))
    ax.plot(path[:, 0], path[:, 2], "-", color="crimson", lw=2)
    ax.plot(*path[0, [0, 2]], "o", color="green", ms=10, label="start")
    ax.plot(*path[-1, [0, 2]], "*", color="red", ms=16, label=f"goal ({target['id']})")
    ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_aspect("equal")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[plot] {out}")


# ---------- main ----------

def run(cfg: Cfg) -> dict:
    nc, floor_y = cfg["navgrid"], cfg["scene"]["floor_y"]
    xyz, labels, assets = load_inputs(cfg)
    grid = Grid(xyz, nc["cell"], nc["pad_cells"])
    obstacles = obstacle_mask(grid, xyz, labels, assets, floor_y, nc)
    inflated, clearance = inflate(obstacles, nc["avatar_radius"], grid.cell)
    floor = floor_mask(grid, xyz, labels, assets, obstacles, nc)
    walk = largest_component(floor & ~inflated)
    print(f"[grid] {grid.nx}x{grid.nz} cells @ {grid.cell}  floor {floor.sum()}  "
          f"obstacle {obstacles.sum()}  walkable {walk.sum()} (fill={nc['floor_fill']})")

    start = pick_start(walk, grid, nc)
    dist = dijkstra(walk, start)
    target, goal = pick_target(assets, grid, dist, nc)
    raw = astar(walk, start, goal)
    smooth = shortcut(walk, raw)
    xz = np.array([grid.center(*c) for c in smooth])
    samples = resample(xz, nc["resample"])

    six, siz = grid.index(samples[:, 0], samples[:, 1])
    assert walk[six, siz].all(), "path sample in non-walkable cell"
    assert not inflated[six, siz].any(), "path sample inside inflated obstacle"
    min_clear = float(clearance[six, siz].min())
    length = float(np.linalg.norm(np.diff(samples, axis=0), axis=1).sum())
    print(f"[path] A* {len(raw)} cells -> {len(smooth)} waypoints -> {len(samples)} samples, "
          f"length {length:.2f} units (~{length * cfg['scene']['units_to_m']:.1f} m), "
          f"min clearance {min_clear:.3f} (avatar radius {nc['avatar_radius']})")

    pts3 = np.stack([samples[:, 0], np.full(len(samples), floor_y), samples[:, 1]], 1)
    result = {"start": pts3[0].round(4).tolist(), "goal": pts3[-1].round(4).tolist(),
              "target_object": target["id"], "target_center": target["center"],
              "waypoints": [[round(x, 4), floor_y, round(z, 4)] for x, z in xz],
              "points": pts3.round(4).tolist(), "length": round(length, 3),
              "min_clearance": round(min_clear, 4), "floor_fill": nc["floor_fill"]}
    with open(out_path(cfg, "path.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(out_path(cfg, "navgrid.npz"), walkable=walk, obstacles=obstacles,
                        inflated=inflated, floor=floor, clearance=clearance.astype(np.float32),
                        origin=np.array([grid.x0, grid.z0]), cell=grid.cell)
    plot_grid(grid, floor, obstacles, inflated, pts3, assets, target,
              f"Nav grid (floor_fill={nc['floor_fill']}) — path {length:.2f} u, "
              f"min clearance {min_clear:.2f}", out_path(cfg, "navgrid.png"))
    return result


if __name__ == "__main__":
    run(load_config())
