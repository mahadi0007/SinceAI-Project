"""Stage 1: splat -> structural planes + object clusters -> out/assets.json.

Steps: load -> voxel + SOR -> normal-checked structural planes -> RANSAC planes ->
DBSCAN -> direction-class split of oversized clusters (pipe runs along x/y/z vs
fittings) -> noise reassignment -> overhead filter -> containment merge -> features.
Outputs: out/assets.json, out/segment_points.npz, out/segment_front.png, out/segment_top.png
"""
from __future__ import annotations

import json
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree

from pipeline.common import load_config, out_path, repo_path
from pipeline.load_splat import load_splat

Cfg = dict[str, Any]
NOISE = -1
OTHER = 3  # direction class for non-linear points (x=0, y=1, z=2)


# ---------- loading / cleanup ----------

def load_points(cfg: Cfg) -> dict[str, np.ndarray]:
    lc = cfg["load"]
    d = load_splat(str(repo_path(cfg["paths"]["ply"])), lc["min_opacity"],
                   lc["max_scale"], lc["crop_floaters"])
    print(f"[load] file rows {d['n_file']:,}  valid prefix {d['n_valid_prefix']:,}  "
          f"kept {len(d['xyz']):,}")
    return d


def downsample_and_denoise(d: dict[str, np.ndarray], sc: Cfg) -> tuple[np.ndarray, ...]:
    """Voxel + SOR. Returns xyz, rgb, opacity (from nearest originals), normals."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(d["xyz"].astype(np.float64)))
    pc.colors = o3d.utility.Vector3dVector(d["rgb"].astype(np.float64))
    pc = pc.voxel_down_sample(sc["voxel"])
    pc, _ = pc.remove_statistical_outlier(sc["outlier_nb"], sc["outlier_std"])
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(sc["normal_knn"]))
    xyz = np.asarray(pc.points)
    _, nn = cKDTree(d["xyz"]).query(xyz, k=sc["opacity_knn"])
    opacity = d["opacity"][nn].mean(1)
    print(f"[clean] voxel {sc['voxel']} + SOR -> {len(xyz):,} points")
    return xyz, np.asarray(pc.colors), opacity, np.asarray(pc.normals)


# ---------- planes ----------

def remove_known_planes(xyz: np.ndarray, normals: np.ndarray, sc: Cfg) -> tuple[list[dict], np.ndarray]:
    """Structural plane = inside the band AND normal aligned with the plane axis."""
    remaining = np.ones(len(xyz), bool)
    planes = []
    for p in sc["known_planes"]:
        band = np.abs(xyz[:, p["axis"]] - p["value"]) < p["tol"]
        m = remaining & band & (np.abs(normals[:, p["axis"]]) > sc["plane_normal_dot"])
        planes.append({"name": p["name"], "label": p["label"], "idx": np.where(m)[0]})
        remaining &= ~m
        print(f"[plane] {p['name']}: {m.sum():,} points ({(band & ~m).sum():,} in band kept, normal not aligned)")
    return planes, remaining


def _dbscan(pts: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    return np.asarray(pc.cluster_dbscan(eps=eps, min_points=min_pts))


def _groups(lab: np.ndarray, min_size: int) -> list[np.ndarray]:
    ids, cnt = np.unique(lab[lab >= 0], return_counts=True)
    return [np.where(lab == i)[0] for i, c in zip(ids, cnt) if c >= min_size]


def ransac_extra_planes(xyz: np.ndarray, remaining: np.ndarray, sc: Cfg) -> list[dict]:
    """Peel off further large connected planes (> min_frac of remaining). Mutates `remaining`."""
    rc = sc["ransac"]
    planes = []
    for k in range(rc["max_planes"]):
        idx = np.where(remaining)[0]
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[idx]))
        model, inl = pc.segment_plane(rc["dist"], rc["n"], rc["iters"])
        inl = np.asarray(inl)
        comps = _groups(_dbscan(xyz[idx[inl]], rc["component_eps"], 4),
                        int(rc["component_min_frac"] * len(idx)))
        sel = idx[inl[np.concatenate(comps)]] if comps else np.array([], int)
        if len(sel) < rc["min_frac"] * len(idx):
            break
        normal = np.round(np.asarray(model[:3]), 3).tolist()
        planes.append({"name": f"plane_{k}", "label": "wall", "idx": sel})
        remaining[sel] = False
        print(f"[ransac] plane_{k} normal {normal}: {len(sel):,} points")
    return planes


# ---------- clustering ----------

def dbscan_clusters(xyz: np.ndarray, sc: Cfg) -> list[np.ndarray]:
    groups = _groups(_dbscan(xyz, sc["dbscan"]["eps"], sc["dbscan"]["min_points"]),
                     sc["min_cluster_points"])
    print(f"[dbscan] {len(groups)} clusters >= {sc['min_cluster_points']} pts")
    return groups


def direction_classes(xyz: np.ndarray, normals: np.ndarray, sp: Cfg) -> np.ndarray:
    """Per-point class 0/1/2 = pipe-like along x/y/z, 3 = other.
    Pipe-like if point-PCA linearity or normal-scatter cylindricity is high and
    the implied axis is near a coordinate axis. Smoothed by kNN majority vote."""
    _, nn = cKDTree(xyz).query(xyz, k=sp["pca_knn"])
    nb = xyz[nn] - xyz[nn].mean(1, keepdims=True)
    w, v = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb))
    lin = (w[:, 2] - w[:, 1]) / np.maximum(w[:, 2], 1e-12)
    nrm = normals[nn]
    wn, vn = np.linalg.eigh(np.einsum("nki,nkj->nij", nrm, nrm) / sp["pca_knn"])
    cyl = (wn[:, 1] - wn[:, 0]) / np.maximum(wn[:, 2], 1e-12)
    use_cyl = cyl > sp["cyl_thresh"]
    axis = np.where(use_cyl[:, None], vn[:, :, 0], v[:, :, 2])
    aligned = np.abs(axis).max(1) > sp["axis_align"]
    pipe = (use_cyl | (lin > sp["lin_thresh"])) & aligned
    cls = np.where(pipe, np.abs(axis).argmax(1), OTHER)
    _, nn = cKDTree(xyz).query(xyz, k=sp["smooth_knn"])
    votes = np.zeros((len(xyz), 4), np.int32)
    np.add.at(votes, (np.repeat(np.arange(len(xyz)), nn.shape[1]), cls[nn].ravel()), 1)
    return votes.argmax(1)


def is_oversized(pts: np.ndarray, sp: Cfg) -> bool:
    diag = float(np.linalg.norm(np.ptp(pts, 0)))
    return diag > sp["max_diag"] or len(pts) > sp["max_points"]


Part = tuple[np.ndarray, str | None, int]   # (indices, hint, axis class)


def split_by_direction(g: np.ndarray, xyz: np.ndarray, cls: np.ndarray, sp: Cfg) -> list[Part]:
    """Split one cluster into straight pipe runs per axis + non-linear parts."""
    parts: list[Part] = []
    for c in range(4):
        sub = g[cls[g] == c]
        if len(sub) < sp["min_part_points"]:
            continue
        for p in _groups(_dbscan(xyz[sub], sp["class_eps"], sp["class_min_points"]),
                         sp["min_part_points"]):
            idx = sub[p]
            diag = float(np.linalg.norm(np.ptp(xyz[idx], 0)))
            hint = "pipe segment" if c != OTHER else (
                "fitting/valve" if diag <= sp["fitting_max_diag"] else None)
            parts.append((idx, hint, c))
    return parts


def split_oversized(groups: list[np.ndarray], xyz: np.ndarray, normals: np.ndarray,
                    sc: Cfg) -> list[Part]:
    sp = sc["split"]
    cls = direction_classes(xyz, normals, sp)
    print(f"[split] direction classes x/y/z/other: {np.bincount(cls, minlength=4).tolist()}")
    out: list[Part] = []
    n_big = 0
    for g in groups:
        if not is_oversized(xyz[g], sp):
            out.append((g, None, OTHER))
            continue
        n_big += 1
        parts = split_by_direction(g, xyz, cls, sp) or [(g, None, OTHER)]
        kept = sum(len(p) for p, _, _ in parts)
        print(f"[split] {len(g):,} pts diag {np.linalg.norm(np.ptp(xyz[g], 0)):.2f} -> "
              f"{len(parts)} parts ({kept / len(g):.0%} of points)")
        out += parts
    print(f"[split] {n_big} oversized clusters -> total {len(out)} clusters")
    return out


def reassign_noise(labels: np.ndarray, xyz: np.ndarray, max_dist: float) -> np.ndarray:
    """Give noise points the label of the nearest clustered point within max_dist."""
    lab = labels.copy()
    noise, clus = np.where(lab == NOISE)[0], np.where(lab >= 0)[0]
    d, j = cKDTree(xyz[clus]).query(xyz[noise], distance_upper_bound=max_dist)
    ok = np.isfinite(d)
    lab[noise[ok]] = lab[clus[j[ok]]]
    print(f"[noise] reassigned {ok.sum():,} of {len(noise):,} noise points (<= {max_dist})")
    return lab


# ---------- post-filters on object list ----------

def filter_overhead(objs: list[dict], xyz: np.ndarray, opacity: np.ndarray, cfg: Cfg) -> list[dict]:
    """Clusters above the ceiling are tagged 'overhead'. Those lying entirely more than
    `margin` above it are kept only if big and opaque enough (else floaters)."""
    oc, ceil = cfg["segment"]["overhead"], cfg["scene"]["ceiling_y"]
    out, dropped = [], 0
    for o in objs:
        y = xyz[o["idx"], 1]                         # up = -y, so smaller y = higher
        if y.mean() < ceil:
            clearly_above = ceil - y.max() > oc["margin"]
            weak = len(o["idx"]) < oc["min_points"] or opacity[o["idx"]].mean() <= oc["min_opacity"]
            if clearly_above and weak:
                dropped += 1
                continue
            o["tags"].append("overhead")
        out.append(o)
    print(f"[overhead] kept {sum('overhead' in o['tags'] for o in out)}, dropped {dropped}")
    return out


def merge_collinear_pipes(objs: list[dict], xyz: np.ndarray, mc: Cfg) -> list[dict]:
    """Union pipe segments on the same axis whose cross-sections overlap and whose
    gap along the axis is small (one physical pipe broken by junctions)."""
    pipes = [i for i, o in enumerate(objs) if o["hint"] == "pipe segment" and o["axis"] < OTHER]
    box = {i: (xyz[objs[i]["idx"]].min(0), xyz[objs[i]["idx"]].max(0)) for i in pipes}
    parent = {i: i for i in pipes}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a_i, i in enumerate(pipes):
        for j in pipes[a_i + 1:]:
            ax = objs[i]["axis"]
            if objs[j]["axis"] != ax:
                continue
            (lo1, hi1), (lo2, hi2) = box[i], box[j]
            cross = [k for k in range(3) if k != ax]
            inter = np.prod([max(0.0, min(hi1[k], hi2[k]) - max(lo1[k], lo2[k])) for k in cross])
            smaller = min(np.prod([hi1[k] - lo1[k] for k in cross]),
                          np.prod([hi2[k] - lo2[k] for k in cross]))
            gap = max(lo1[ax], lo2[ax]) - min(hi1[ax], hi2[ax])
            if smaller > 0 and inter / smaller > mc["min_cross_overlap"] and gap < mc["max_gap"]:
                parent[find(j)] = find(i)
    merged: dict[int, dict] = {}
    for i in pipes:
        r = find(i)
        if r in merged:
            merged[r]["idx"] = np.concatenate([merged[r]["idx"], objs[i]["idx"]])
            merged[r]["tags"] = sorted(set(merged[r]["tags"]) | set(objs[i]["tags"]))
        else:
            merged[r] = objs[i]
    out = [o for i, o in enumerate(objs) if i not in parent] + list(merged.values())
    print(f"[collinear] {len(pipes)} pipe segments -> {len(merged)} pipe runs")
    return out


def merge_contained(objs: list[dict], xyz: np.ndarray, tol: float) -> list[dict]:
    """Merge an object into a larger one of the same hint whose box fully contains it."""
    objs = sorted(objs, key=lambda o: -len(o["idx"]))
    boxes = [(xyz[o["idx"]].min(0), xyz[o["idx"]].max(0)) for o in objs]
    alive = [True] * len(objs)
    for i in range(len(objs)):
        for j in range(i):
            if alive[j] and objs[i]["hint"] == objs[j]["hint"] and \
               (boxes[i][0] >= boxes[j][0] - tol).all() and (boxes[i][1] <= boxes[j][1] + tol).all():
                objs[j]["idx"] = np.concatenate([objs[j]["idx"], objs[i]["idx"]])
                alive[i] = False
                break
    print(f"[contain] merged {alive.count(False)} contained boxes")
    return [o for o, a in zip(objs, alive) if a]


# ---------- features / output ----------

def shape_features(pts: np.ndarray) -> dict[str, float]:
    ev = np.maximum(np.sort(np.linalg.eigvalsh(np.cov((pts - pts.mean(0)).T)))[::-1], 1e-12)
    return {"linearity": float((ev[0] - ev[1]) / ev[0]),
            "planarity": float((ev[1] - ev[2]) / ev[0]),
            "sphericity": float(ev[2] / ev[0])}


def oriented_box(pts: np.ndarray) -> dict[str, list]:
    try:
        ob = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pts))
        return {"center": np.round(ob.center, 4).tolist(), "R": np.round(ob.R, 4).tolist(),
                "extent": np.round(ob.extent, 4).tolist()}
    except RuntimeError:  # degenerate (coplanar) point sets
        return {}


def cluster_record(i: int, idx: np.ndarray, xyz: np.ndarray, rgb: np.ndarray, opacity: np.ndarray,
                   cfg: Cfg, kind: str, hint: str | None, tags: list[str]) -> dict[str, Any]:
    pts = xyz[idx]
    lo, hi = pts.min(0), pts.max(0)
    floor_y = cfg["scene"]["floor_y"]
    f = shape_features(pts)
    f.update({"height_bottom": float(floor_y - hi[1]), "height_top": float(floor_y - lo[1]),
              "diag": float(np.linalg.norm(hi - lo)), "mean_opacity": float(opacity[idx].mean())})
    if hint is None and kind == "object" and f["linearity"] > cfg["segment"]["pipe_linearity"]:
        hint = "pipe segment"
    rec = {"id": f"OBJ-{i:04d}", "kind": kind, "hint": hint, "tags": tags,
           "center": ((lo + hi) / 2).round(4).tolist(), "extent": (hi - lo).round(4).tolist(),
           "bbox_min": lo.round(4).tolist(), "bbox_max": hi.round(4).tolist(),
           "n_points": int(len(idx)), "color": rgb[idx].mean(0).round(3).tolist(),
           "features": {k: round(v, 4) for k, v in f.items()}}
    if kind == "object":
        rec["obb"] = oriented_box(pts)
    return rec


def build_assets(planes: list[dict], objs: list[dict], xyz: np.ndarray, rgb: np.ndarray,
                 opacity: np.ndarray, cfg: Cfg) -> tuple[list[dict], np.ndarray]:
    assets, labels = [], np.full(len(xyz), NOISE, np.int32)
    for p in planes:
        if len(p["idx"]):
            labels[p["idx"]] = len(assets)
            assets.append(cluster_record(len(assets), p["idx"], xyz, rgb, opacity, cfg,
                                         "structure", p["label"], []))
            assets[-1]["name"] = p["name"]
    for o in sorted(objs, key=lambda o: -len(o["idx"])):
        labels[o["idx"]] = len(assets)
        assets.append(cluster_record(len(assets), o["idx"], xyz, rgb, opacity, cfg,
                                     "object", o["hint"], o["tags"]))
    return assets, labels


# ---------- plots ----------

def plot_view(xyz: np.ndarray, labels: np.ndarray, assets: list[dict], axes: tuple[int, int],
              flip_v: bool, title: str, path, pc: Cfg) -> None:
    rng = np.random.default_rng(0)
    cols = rng.uniform(0.15, 0.95, (len(assets), 3))
    for i, A in enumerate(assets):
        if A["kind"] == "structure":
            cols[i] = pc["structure_color"]
    sel = np.arange(len(xyz))
    if len(sel) > pc["max_points"]:
        sel = rng.choice(sel, pc["max_points"], replace=False)
    a, b = axes
    sgn = -1.0 if flip_v else 1.0
    lab = labels[sel]
    c = np.where(lab[:, None] >= 0, cols[np.maximum(lab, 0)], np.array(pc["noise_color"]))
    order = np.argsort(lab >= 0)  # draw noise first
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.scatter(xyz[sel[order], a], sgn * xyz[sel[order], b], c=c[order], s=pc["point_size"], linewidths=0)
    for i, A in enumerate(assets):
        if A["kind"] != "object":
            continue
        lo, hi = A["bbox_min"], A["bbox_max"]
        y0, y1 = sorted([sgn * lo[b], sgn * hi[b]])
        ax.add_patch(Rectangle((lo[a], y0), hi[a] - lo[a], y1 - y0, fill=False, ec=cols[i],
                               lw=0.6, ls="--" if A["hint"] == "fitting/valve" else "-"))
    n_obj = sum(A["kind"] == "object" for A in assets)
    noise = (labels == NOISE).mean()
    ax.set_title(f"{title} — {n_obj} objects + {len(assets) - n_obj} planes · "
                 f"noise {noise:.1%} (dark red) · planes grey · dashed = fitting/valve")
    ax.set_xlabel("x"); ax.set_ylabel(("-" if flip_v else "") + "xyz"[b])
    ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(path, dpi=pc["dpi"]); plt.close(fig)
    print(f"[plot] {path}")


# ---------- main ----------

def run(cfg: Cfg) -> list[dict]:
    t0 = time.time()
    sc = cfg["segment"]
    xyz, rgb, opacity, normals = downsample_and_denoise(load_points(cfg), sc)
    planes, remaining = remove_known_planes(xyz, normals, sc)
    planes += ransac_extra_planes(xyz, remaining, sc)
    oi = np.where(remaining)[0]
    groups = dbscan_clusters(xyz[oi], sc)
    big = max(groups, key=len)
    print(f"[dbscan] largest before split: {len(big):,} pts, diag {np.linalg.norm(np.ptp(xyz[oi][big], 0)):.2f}")
    parts = split_oversized(groups, xyz[oi], normals[oi], sc)

    lab = np.full(len(xyz), NOISE, np.int32)
    lab[~remaining] = -2                                  # structure: excluded from reassignment
    for k, (p, _, _) in enumerate(parts):
        lab[oi[p]] = k
    lab[oi] = reassign_noise(lab[oi], xyz[oi], sc["noise_reassign_dist"])
    objs = [{"idx": np.where(lab == k)[0], "hint": h, "axis": c, "tags": []}
            for k, (_, h, c) in enumerate(parts)]
    objs = filter_overhead(objs, xyz, opacity, cfg)
    objs = merge_collinear_pipes(objs, xyz, sc["collinear"])
    objs = merge_contained(objs, xyz, sc["contain_tol"])
    assets, labels = build_assets(planes, objs, xyz, rgb, opacity, cfg)

    ob = [A for A in assets if A["kind"] == "object"]
    hints: dict[str, int] = {}
    for A in ob:
        key = A["hint"] or "unlabelled"
        hints[key] = hints.get(key, 0) + 1
    top = max(ob, key=lambda A: A["n_points"])
    noise = float((labels == NOISE).mean())
    print(f"[result] largest object: {top['id']} {top['n_points']:,} pts, diag {top['features']['diag']:.2f}")
    print(f"[result] hints {hints}  overhead {sum('overhead' in A['tags'] for A in ob)}")
    with open(out_path(cfg, "assets.json"), "w", encoding="utf-8") as f:
        json.dump(assets, f, indent=1)
    np.savez_compressed(out_path(cfg, "segment_points.npz"), xyz=xyz.astype(np.float32),
                        rgb=rgb.astype(np.float32), opacity=opacity.astype(np.float32), labels=labels)
    plot_view(xyz, labels, assets, (0, 1), True, "Front (x vs -y)", out_path(cfg, "segment_front.png"), cfg["plots"])
    plot_view(xyz, labels, assets, (0, 2), False, "Top (x vs z)", out_path(cfg, "segment_top.png"), cfg["plots"])
    print(f"\nEXTRACTED OBJECTS: {len(ob)}  (+{len(assets) - len(ob)} structural)  "
          f"noise {noise:.1%}  [{time.time() - t0:.0f}s]")
    return assets


if __name__ == "__main__":
    run(load_config())
