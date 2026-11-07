"""Navigation for an edited object layout (viewer.app "Apply layout").

The scan's segmented points are moved with each edited object (removed objects dropped, copies added), then the
occupancy grid is rebuilt exactly like pipeline.navgrid (same grid geometry as out/navgrid.npz), and the walk goal
next to the target is recomputed from the target's moved bounds. The original navgrid.npz / path.json / tour.json
are never changed; the result goes to out/layout_nav.npz + out/layout_nav.json, tagged with the layout hash and
the scene revision, and is handed to pipeline.tour (plan_tour / WalkPlanner `inputs=`).
"""
from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np

from pipeline.common import out_path
from pipeline.navgrid import (dijkstra, floor_mask, inflate, largest_component, load_inputs as nav_inputs,
                              nearest_walkable, obstacle_mask)
from pipeline.tour import cell_of, load_inputs, stop_goal

Cfg = dict[str, Any]


def _apply(points: np.ndarray, it: dict, pivot: Any) -> np.ndarray:
    from viewer.scene_edits import apply
    return apply(points, it["position"], it["wxyz"], pivot)


def edited_points(xyz: np.ndarray, labels: np.ndarray, members: dict[str, list[int]], instances: dict[str, dict],
                  pivots: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Segmented points after the layout: every editable object's points (with its parts) follow its instances."""
    all_members = [k for ks in members.values() for k in ks]
    moving = np.isin(labels, all_members)
    xs, ls = [xyz[~moving]], [labels[~moving]]
    for it in instances.values():
        if it["removed"]:
            continue
        m = np.isin(labels, members[it["source"]])
        xs.append(_apply(xyz[m], it, pivots[it["source"]]))
        ls.append(labels[m])
    return np.concatenate(xs), np.concatenate(ls)


def moved_bounds(xyz: np.ndarray, labels: np.ndarray, objects: list[dict], instances: dict[str, dict],
                 pivots: dict[str, Any]) -> dict[str, dict]:
    """Original editable objects -> their bounds after the layout ({} entry 'removed': True if removed)."""
    idx = {o["id"]: k for k, o in enumerate(objects)}
    out = {}
    for name, it in instances.items():
        if it["copy"]:
            continue
        oid = it["source"]
        if it["removed"]:
            out[oid] = {"removed": True}
            continue
        P = _apply(xyz[labels == idx[oid]], it, pivots[oid])
        lo, hi = P.min(0), P.max(0)
        out[oid] = {"removed": False, "bbox_min": lo.tolist(), "bbox_max": hi.tolist(),
                    "center": ((lo + hi) / 2).tolist(), "extent": (hi - lo).tolist()}
    return out


def build(cfg: Cfg, state: dict[str, Any], members: dict[str, list[int]], layout: str) -> dict[str, Any]:
    """Grid layers, path goal and scene graph for this layout. Returns {inputs, bounds, report}."""
    from viewer.scene_edits import layout_hash
    if layout_hash(state["instances"], state["pivots"]) != layout:
        raise ValueError("layout changed while navigation was being rebuilt")
    nc, floor_y = cfg["navgrid"], cfg["scene"]["floor_y"]
    grid, base_layers, path, sg = load_inputs(cfg)
    xyz, labels, assets = nav_inputs(cfg)
    ex, el = edited_points(xyz, labels, members, state["instances"], state["pivots"])
    ix, iz = grid.index(ex[:, 0], ex[:, 2])
    inside = (ix >= 0) & (ix < grid.nx) & (iz >= 0) & (iz < grid.nz)
    ex, el = ex[inside], el[inside]
    obstacles = obstacle_mask(grid, ex, el, assets, floor_y, nc)
    inflated, clearance = inflate(obstacles, nc["avatar_radius"], grid.cell)
    floor = floor_mask(grid, ex, el, assets, obstacles, nc)
    walk = largest_component(floor & ~inflated)
    layers = {"walkable": walk, "inflated": inflated, "clearance": clearance.astype(np.float32),
              "obstacles": obstacles, "floor": floor}

    bounds = moved_bounds(xyz, labels, sg["objects"], state["instances"], state["pivots"])
    sg = copy.deepcopy(sg)
    for o in sg["objects"]:
        b = bounds.get(o["id"])
        if b and not b["removed"]:
            o.update({k: b[k] for k in ("bbox_min", "bbox_max", "center", "extent")})
    by_id = {o["id"]: o for o in sg["objects"]}
    target = by_id[path["target_object"]]
    path = dict(path)
    report = {"layout": layout, "obstacle_cells": int(obstacles.sum()), "walkable_cells": int(walk.sum()),
              "obstacle_cells_before": int(base_layers["obstacles"].sum()),
              "walkable_cells_before": int(base_layers["walkable"].sum()), "target": target["id"]}
    if bounds.get(target["id"], {}).get("removed"):
        path["goal"] = None
        report["target_status"] = "removed"
    else:
        start = nearest_walkable(walk, cell_of(grid, path["start"][0], path["start"][2]))
        goal = stop_goal(target, grid, dijkstra(walk, start), cfg["tour"]["reach"])
        if goal is None:
            path["goal"] = None
            report["target_status"] = "unreachable"
        else:
            gx, gz = grid.center(*goal)
            path["goal"] = [gx, floor_y, gz]
            report["target_status"] = "reachable"
    np.savez_compressed(out_path(cfg, "layout_nav.npz"), origin=np.array([grid.x0, grid.z0]), cell=grid.cell,
                        **layers)
    out_path(cfg, "layout_nav.json").write_text(json.dumps(
        {"revision": state["revision"], "report": report, "bounds": bounds,
         "note": "derived from scene_edits.json; navgrid.npz / path.json / tour.json are the unedited scene"},
        indent=1), encoding="utf-8")
    return {"inputs": (grid, layers, path, sg), "bounds": bounds, "report": report}
