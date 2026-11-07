"""Boundary repair on top of the baseline segmentation, reversible and evidence-logged.

Start: the baseline's retained FINE fragments (leaves, before collinear/containment merges; overhead
drops respected). Two repair families, each written as an edit log over leaves/voxels:

1. geometry (pipes only): join pipe leaves when fitted centrelines agree in direction, lateral offset
   and thickness, the axial gap is small, no valve/fitting/equipment points sit in the gap (barrier),
   and the WHOLE joined run still fits one straight line (no transitive drift). Box overlap or
   containment is never a reason to merge. Bends/tees/valves are not crossed.
2. context (equipment): accepted vision proposals (Qwen, per-view boxes/points) refined by SAM 2.1
   masks; support = VISIBLE Gaussians inside the mask (ID buffer of calibrated renders, no
   projection through occluders). Objects mostly inside the masks across views are merged;
   mixed non-pipe objects are split using only mask-supported voxel subsets. Elongated pipe runs are
   never merged into or split by an equipment candidate (mask overlap alone is not enough).

Every edit lists inputs/outputs (leaf ids, voxel counts) and evidence; undo = drop the entry and
re-apply the log (apply_log). Baseline labels are kept in segment_points.npz as labels_baseline.
Run: python -m pipeline.repair geo        -> out_geo/
     python -m pipeline.repair context[:<set>]  set = <renderer>-<proposal source>, e.g. ewa-ewa -> out/,
                                                 others -> out_ctx_<set>/
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from pipeline.common import ROOT, load_config
from pipeline.segment import NOISE, OTHER, cluster_record

Cfg = dict[str, Any]


# ---------- baseline leaves ----------

def load_base(seg_dir: str = "out_baseline") -> dict:
    d = dict(np.load(ROOT / seg_dir / "segment_points.npz"))
    d["assets"] = json.loads((ROOT / seg_dir / "assets.json").read_text(encoding="utf-8"))
    return d


def leaves_of(base: dict) -> dict[int, np.ndarray]:
    """leaf (fine fragment) id -> voxel indices, only fragments kept in some baseline object."""
    frag, lab = base["frag"], base["labels"]
    keep = set()
    for a in base["assets"]:
        keep.update(a.get("fragments", []))
    sel = np.where((frag >= 0) & (lab >= 0))[0]
    order = np.argsort(frag[sel], kind="stable")
    f_sorted = frag[sel][order]
    out: dict[int, np.ndarray] = {}
    starts = np.flatnonzero(np.r_[True, f_sorted[1:] != f_sorted[:-1]])
    ends = np.r_[starts[1:], len(f_sorted)]
    for s, e in zip(starts, ends):
        f = int(f_sorted[s])
        if f in keep:
            out[f] = sel[order[s:e]]
    return out


# ---------- 1. geometry: pipe continuity ----------

def line_fit(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """centroid, unit axis, t-range (t0, t1), cross-section half-width (robust)."""
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    ax = vt[0]
    t = (P - c) @ ax
    lat = np.linalg.norm((P - c) - np.outer(t, ax), axis=1)
    return c, ax, float(t.min()), float(t.max()), float(np.percentile(lat, 90))


def pipe_features(P: np.ndarray, axis_class: int | None = None) -> dict:
    """Short partial-surface pieces give noisy PCA axes; leaves classified as x/y/z-aligned pipe
    runs use that coordinate axis (they were split by it), others the PCA axis."""
    c, ax, _, _, _ = line_fit(P)
    if axis_class is not None and axis_class < OTHER:
        ax = np.eye(3)[axis_class]
    t = (P - c) @ ax
    lat = np.linalg.norm((P - c) - np.outer(t, ax), axis=1)
    return {"c": c, "ax": ax, "t0": float(t.min()), "t1": float(t.max()),
            "hw": max(float(np.percentile(lat, 90)), 0.01), "len": float(t.max() - t.min())}


def geo_group(base: dict, leaves: dict[int, np.ndarray], gc: Cfg) -> tuple[list[list[int]], list[dict]]:
    xyz = base["xyz"].astype(np.float64)
    hint, axis = base["frag_hint"], base["frag_axis"]
    pipes = [f for f in leaves if hint[f] == "pipe segment" and axis[f] < OTHER and len(leaves[f]) >= gc["min_points"]]
    feat = {f: pipe_features(xyz[leaves[f]], int(axis[f])) for f in pipes}
    pipes = [f for f in pipes if feat[f]["len"] >= 2 * feat[f]["hw"]]          # elongated only
    # barriers = compact non-pipe pieces (valve/fitting/equipment sized). A large mixed cluster that
    # merely hugs the pipe is not evidence of a valve in the gap.
    def compact(f: int) -> bool:
        return float(np.linalg.norm(np.ptp(xyz[leaves[f]], 0))) <= gc["barrier_max_diag"]
    barrier_idx = np.concatenate([leaves[f] for f in leaves if hint[f] != "pipe segment" and compact(f)]
                                 or [np.zeros(0, int)])
    btree = cKDTree(xyz[barrier_idx]) if len(barrier_idx) else None
    parent = {f: f for f in pipes}
    members = {f: [f] for f in pipes}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # candidate pairs: endpoints close
    ends = []
    for f in pipes:
        F = feat[f]
        ends += [(f, F["c"] + F["t0"] * F["ax"]), (f, F["c"] + F["t1"] * F["ax"])]
    etree = cKDTree(np.array([e[1] for e in ends]))
    pairs = set()
    for i, j in etree.query_pairs(gc["max_gap"] + 0.1):
        a, b = ends[i][0], ends[j][0]
        if a != b:
            pairs.add((min(a, b), max(a, b)))
    log, rejected = [], {"angle": 0, "offset": 0, "thickness": 0, "gap": 0, "barrier": 0, "run_fit": 0}
    for a, b in sorted(pairs, key=lambda p: (p[0], p[1])):
        A, B = feat[a], feat[b]
        cosang = abs(float(A["ax"] @ B["ax"]))
        if cosang < np.cos(np.radians(gc["max_angle_deg"])):
            rejected["angle"] += 1
            continue
        d = B["c"] - A["c"]
        off = float(np.linalg.norm(d - (d @ A["ax"]) * A["ax"]))
        off2 = float(np.linalg.norm(-d - (-d @ B["ax"]) * B["ax"]))
        hw = max(A["hw"], B["hw"])
        if max(off, off2) > gc["offset_frac"] * hw + gc["offset_abs"]:
            rejected["offset"] += 1
            continue
        ratio = max(A["hw"], B["hw"]) / min(A["hw"], B["hw"])
        if ratio > gc["max_thickness_ratio"]:
            rejected["thickness"] += 1
            continue
        tb = (xyz[leaves[b]] - A["c"]) @ A["ax"]
        gap = max(tb.min() - A["t1"], A["t0"] - tb.max())
        if gap > gc["max_gap"]:
            rejected["gap"] += 1
            continue
        if gap > 0 and btree is not None:                     # anything non-pipe in the gap?
            mid = A["c"] + A["ax"] * ((A["t1"] + tb.min()) / 2 if tb.min() > A["t1"] else (A["t0"] + tb.max()) / 2)
            nb = btree.query_ball_point(mid, gap / 2 + hw)
            if len(nb) >= gc["barrier_points"]:
                rejected["barrier"] += 1
                continue
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        P = xyz[np.concatenate([leaves[f] for f in members[ra] + members[rb]])]
        c, ax, _, _, _ = line_fit(P)
        lat = np.linalg.norm((P - c) - np.outer((P - c) @ ax, ax), axis=1)
        if np.percentile(lat, 90) > gc["run_fit_frac"] * hw + gc["offset_abs"]:   # no transitive drift
            rejected["run_fit"] += 1
            continue
        parent[rb] = ra
        members[ra] += members.pop(rb)
        log.append({"op": "merge", "family": "geometry", "leaves": [a, b], "evidence": {
            "angle_deg": round(float(np.degrees(np.arccos(min(1, cosang)))), 2), "offset": round(max(off, off2), 4),
            "thickness_ratio": round(ratio, 3), "gap": round(float(gap), 4)}})
    groups = [m for r, m in members.items() if find(r) == r]
    grouped = {f for g in groups for f in g}
    groups += [[f] for f in leaves if f not in grouped]
    print(f"[geo] {len(pipes)} pipe leaves, {len(pairs)} candidate pairs, {len(log)} joins; rejected {rejected}")
    return groups, log


# ---------- 2. context: proposal + SAM + visibility ----------

def claim_support(pos: np.ndarray, neg: np.ndarray, vox_group: np.ndarray, G: int, protected: np.ndarray,
                  xyz: np.ndarray, rc: Cfg, describe: Any = None, protected_subset: bool = False) -> dict:
    """Which voxels ONE candidate (automatic proposal or manual correction) may claim from per-voxel visible
    support pos/neg. Shared by context_edits and pipeline.correct.
      whole group  only if visible-in-mask share >= member_frac AND observed coverage >= min_coverage AND
                   spatial_frac of its voxels lie within spatial_dist of a supported voxel;
      subset       otherwise only observed voxels with pos > 0 and pos >= neg; unseen voxels stay where they
                   were and are counted as uncertain (a small visible patch never absorbs a whole cluster).
    protected groups (long pipe runs, structures) are never claimed whole; with protected_subset=True (manual,
    reviewer-drawn masks) their supported subset may still be cut out, otherwise they are skipped."""
    n = len(pos)
    seen = (pos + neg) > rc["seen_eps"]
    strength = np.where(seen, pos / np.maximum(pos + neg, 1e-12), 0.0)
    ok = vox_group >= 0
    vis_g = np.bincount(vox_group[ok], weights=(pos + neg)[ok], minlength=G)
    in_g = np.bincount(vox_group[ok], weights=pos[ok], minlength=G)
    take = np.zeros(n, bool)
    members, splits, skipped = [], [], []
    uncertain = 0
    inferred = np.zeros(n, bool)
    cand = np.where(in_g >= rc["min_in_px"])[0]
    order = np.argsort(vox_group, kind="stable")
    vs = vox_group[order]
    for gi in cand:
        lo, hi = np.searchsorted(vs, gi), np.searchsorted(vs, gi, side="right")
        vg = np.sort(order[lo:hi])
        frac = in_g[gi] / max(vis_g[gi], 1e-12)
        extra = describe(int(gi)) if describe else {}
        if protected[gi] and not protected_subset:
            skipped.append({"group": int(gi), **extra, "reason": "pipe run protected (no merge/split)",
                            "support": round(float(frac), 3)})
            continue
        sup = vg[seen[vg] & (pos[vg] > 0) & (pos[vg] >= neg[vg])]
        cov = float(seen[vg].mean())
        spatial = 0.0
        if len(sup):
            dd, _ = cKDTree(xyz[sup]).query(xyz[vg])
            spatial = float((dd <= rc["spatial_dist"]).mean())
        ev = {"group": int(gi), **extra, "support": round(float(frac), 3), "coverage": round(cov, 3),
              "spatial": round(spatial, 3), "voxels_total": int(len(vg))}
        if (not protected[gi] and frac >= rc["member_frac"] and cov >= rc["min_coverage"]
                and spatial >= rc["spatial_frac"]):
            take[vg] = True
            unseen = vg[~seen[vg]]
            strength[unseen] = frac                     # inferred voxels carry the group's support
            inferred[unseen] = True
            members.append({**ev, "voxels": int(len(vg)), "inferred_unseen": int(len(unseen))})
            continue
        if len(sup) < rc["min_split_voxels"]:
            skipped.append({**ev, "reason": "too few supported voxels", "supported": int(len(sup))})
            continue
        take[sup] = True
        n_unseen = int((~seen[vg]).sum())
        uncertain += n_unseen
        splits.append({**ev, "voxels_moved": int(len(sup)), "unseen_left_uncertain": n_unseen,
                       "why_not_whole": ("protected (pipe run / structure)" if protected[gi] else
                                         "low coverage" if cov < rc["min_coverage"] else
                                         "spatially inconsistent" if spatial < rc["spatial_frac"] else
                                         "low support")})
    return {"take": take, "strength": strength, "seen": seen, "inferred": inferred, "members": members,
            "splits": splits, "skipped": skipped, "uncertain": uncertain}


def context_edits(base: dict, groups: list[list[int]], leaves: dict[int, np.ndarray], cfg: Cfg,
                  prop_dir: Path, freshness: Any = None) -> tuple[np.ndarray, list[dict], dict[int, dict]]:
    """Returns voxel -> group-or-candidate label, the edit log and candidate metadata.
    Groups are numbered 0..G-1; accepted candidates get labels G, G+1, ...

    Phase 1 - every candidate claims voxels independently (no candidate sees another's result):
      whole group  only if visible-in-mask share >= member_frac AND observed coverage >= min_coverage AND
                   the group is spatially consistent with its supported voxels (spatial_frac of its voxels lie
                   within spatial_dist of one); otherwise
      subset       only observed voxels with pos > 0 and pos >= neg; unseen voxels stay where they were and
                   are counted as uncertain (never filled in).
    Phase 2 - competing claims are resolved per voxel by support strength pos/(pos+neg); a voxel goes to a
      candidate only if it beats the runner-up by claim_margin, otherwise it stays put (ambiguous).
      Candidates that lose or leave ambiguous > provisional_frac of their claim are marked provisional.
    File order never decides ownership. `freshness(info) -> reason|None` rejects stale associations."""
    rc = cfg["repair"]["context"]
    n = len(base["xyz"])
    xyz = base["xyz"].astype(np.float64)
    vox_group = np.full(n, -1, np.int64)
    for gi, g in enumerate(groups):
        for f in g:
            vox_group[leaves[f]] = gi

    def elongated(g: list[int]) -> bool:
        P = xyz[np.concatenate([leaves[f] for f in g])]
        return float(np.ptp(P, 0).max()) >= rc["pipe_protect_min_len"]
    is_pipe_group = np.array([all(base["frag_hint"][f] == "pipe segment" for f in g) and elongated(g) for g in groups])
    G = len(groups)
    log: list[dict] = []
    claims: list[dict] = []
    files = sorted(prop_dir.glob("*_assoc.npz")) if prop_dir.exists() else []
    for fpath in files:
        info = json.loads(fpath.with_name(fpath.name.replace("_assoc.npz", "_assoc.json")).read_text(encoding="utf-8"))
        if freshness is not None:
            why = freshness(info)
            if why:
                log.append({"op": "reject_stale", "family": "context", "region": info["region"], "reason": why})
                print(f"[context] {info['region']}: stale association rejected ({why})")
                continue
        A = np.load(fpath)
        for ci, c in enumerate(info["candidates"]):
            if not c.get("usable"):
                continue
            tag = f"{info['region']}/{c['cid']}"
            if tag in (rc.get("disabled") or []):
                log.append({"op": "disabled", "family": "context", "candidate": c["cid"], "region": info["region"]})
                continue
            pos, neg = A[f"pos_{ci}"].astype(np.float64), A[f"neg_{ci}"].astype(np.float64)
            cl = claim_support(pos, neg, vox_group, G, is_pipe_group, xyz, rc,
                               describe=lambda gi: {"leaves": groups[gi]})
            take, strength = cl["take"], cl["strength"]
            ev_members, ev_splits, ev_skipped, uncertain = cl["members"], cl["splits"], cl["skipped"], cl["uncertain"]
            idx = np.where(take)[0]
            if len(idx) < rc["min_candidate_voxels"]:
                continue
            idx = largest_components(xyz, idx, rc["cc_eps"], rc["cc_min_frac"])
            ext = np.ptp(xyz[idx], 0)
            if ext.max() > rc["max_extent"]:
                log.append({"op": "reject", "family": "context", "candidate": c["cid"], "region": info["region"],
                            "reason": f"implausible extent {ext.round(2).tolist()}"})
                continue
            claims.append({"tag": tag, "c": c, "info": info, "idx": idx, "strength": strength[idx],
                           "members": ev_members, "splits": ev_splits, "skipped": ev_skipped,
                           "uncertain_unseen": uncertain})
    # ---- phase 2a: corroboration - same-class claims on largely the same voxels (e.g. one valve proposed from two
    # neighbouring regions) are one component seen twice: union them instead of letting them compete ----
    claims.sort(key=lambda k: k["tag"])
    fam = {"valve": "valve", "control valve": "valve"}
    merged = True
    while merged:
        merged = False
        for a_i in range(len(claims)):
            for b_i in range(a_i + 1, len(claims)):
                A, B = claims[a_i], claims[b_i]
                if fam.get(A["c"]["cls"], A["c"]["cls"]) != fam.get(B["c"]["cls"], B["c"]["cls"]):
                    continue
                inter = np.intersect1d(A["idx"], B["idx"], assume_unique=True)
                if len(inter) < rc["corroborate_overlap"] * min(len(A["idx"]), len(B["idx"])):
                    continue
                idx = np.union1d(A["idx"], B["idx"])
                st = np.zeros(n)
                st[A["idx"]] = A["strength"]
                st[B["idx"]] = np.maximum(st[B["idx"]], B["strength"])
                A.update(idx=idx, strength=st[idx], members=A["members"] + B["members"], splits=A["splits"] + B["splits"],
                         skipped=A["skipped"] + B["skipped"], uncertain_unseen=A["uncertain_unseen"] + B["uncertain_unseen"],
                         corroborated_by=A.get("corroborated_by", []) + [B["tag"]] + B.get("corroborated_by", []))
                claims.pop(b_i)
                merged = True
                break
            if merged:
                break
    # ---- phase 2b: order-independent resolution of competing claims ----
    best = np.full(n, -1.0)
    second = np.full(n, -1.0)
    owner = np.full(n, -1, np.int64)
    for k, cl in enumerate(claims):
        s, i = cl["strength"], cl["idx"]
        better = s > best[i]
        second[i] = np.where(better, best[i], np.maximum(second[i], s))
        owner[i] = np.where(better, k, owner[i])
        best[i] = np.where(better, s, best[i])
    contested = second >= 0
    decided = (owner >= 0) & (~contested | (best - second >= rc["claim_margin"]))
    cand_meta: dict[int, dict] = {}
    next_label = G
    for k, cl in enumerate(claims):
        i = cl["idx"]
        won = i[decided[i] & (owner[i] == k)]
        lost = i[decided[i] & (owner[i] != k)]
        amb = i[~decided[i]]
        rivals = sorted({claims[o]["tag"] for o in owner[i[contested[i]]] if o != k} |
                        {c2["tag"] for c2 in claims if c2 is not cl and np.isin(i, c2["idx"]).any()})
        c, info = cl["c"], cl["info"]
        if len(won) < rc["min_candidate_voxels"]:
            log.append({"op": "reject", "family": "context", "candidate": c["cid"], "region": info["region"],
                        "reason": f"lost competing claims ({len(won)} voxels left; rivals {rivals})"})
            continue
        provisional = (len(lost) + len(amb)) > rc["provisional_frac"] * len(i)
        old = vox_group[won].copy()
        vox_group[won] = next_label
        cand_meta[next_label] = {"cid": c["cid"], "region": info["region"], "cls": c["cls"],
                                 "confidence": c["confidence"], "assembly": c.get("assembly"),
                                 "model": info.get("model"), "sam_scores": c.get("sam_scores"),
                                 "renderer": info.get("renderer"), "provisional": bool(provisional)}
        log.append({"op": "context_repair", "family": "context", "candidate": c["cid"], "region": info["region"],
                    "cls": c["cls"], "new_label": next_label, "voxels": int(len(won)),
                    "status": "provisional" if provisional else "accepted",
                    "conflict": {"claimed": int(len(i)), "lost": int(len(lost)), "ambiguous": int(len(amb)),
                                 "rivals": rivals},
                    "uncertain_unseen": cl["uncertain_unseen"], "corroborated_by": cl.get("corroborated_by", []),
                    "from_groups": {int(a): int(b) for a, b in zip(*np.unique(old, return_counts=True))},
                    "members": cl["members"], "splits": cl["splits"], "skipped": cl["skipped"],
                    "provenance": {k2: info.get(k2) for k2 in ("proposal_file", "proposal_key", "proposal_renderer",
                                                                "renderer", "render_key", "sam_key", "sam_model",
                                                                "seg_rev", "assoc_key", "model", "n_views")},
                    "moved_voxels": won.tolist()})
        next_label += 1
    return vox_group, log, cand_meta


def largest_components(xyz: np.ndarray, idx: np.ndarray, eps: float, min_frac: float) -> np.ndarray:
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[idx]))
    lab = np.asarray(pc.cluster_dbscan(eps=eps, min_points=3))
    if (lab >= 0).sum() == 0:
        return idx
    ids, cnt = np.unique(lab[lab >= 0], return_counts=True)
    keep = ids[cnt >= min_frac * cnt.max()]
    return idx[np.isin(lab, keep)]


# ---------- build a segmentation version ----------

def build(base: dict, vox_group: np.ndarray, groups: list[list[int]], cand_meta: dict[int, dict],
          cfg: Cfg, out_dir: str, log: list[dict], variant: str) -> None:
    out = ROOT / out_dir
    out.mkdir(exist_ok=True)
    for stale in ("assets_labeled.json", "path.json", "scene_graph.json", "scene_anchors.csv", "navgrid.npz"):
        if (out / stale).exists():       # downstream of the previous segmentation version: must be regenerated
            (out / stale).unlink()
            print(f"[build] removed stale {out_dir}/{stale} (rerun label -> navgrid -> mapping)")
    xyz, rgb, opac = base["xyz"].astype(np.float64), base["rgb"], base["opacity"]
    base_assets = base["assets"]
    structures = [a for a in base_assets if a["kind"] == "structure"]
    base_lab = base["labels"]
    frag = base["frag"]
    leaf_to_base = {}
    for a in base_assets:
        for f in a.get("fragments", []):
            leaf_to_base[f] = a["id"]
    labels = np.full(len(xyz), NOISE, np.int32)
    assets: list[dict] = []
    for i, a in enumerate(structures):
        m = base_lab == base_assets.index(a)
        labels[m] = len(assets)
        assets.append(a)
    objs = []
    for g in np.unique(vox_group[vox_group >= 0]):
        idx = np.where(vox_group == g)[0]
        if len(idx) < cfg["repair"]["min_object_points"]:
            continue
        objs.append((int(g), idx))
    objs.sort(key=lambda t: -len(t[1]))
    for g, idx in objs:
        lf = sorted({int(f) for f in np.unique(frag[idx]) if f >= 0})
        if g in cand_meta:
            cm = cand_meta[g]
            hint = {"valve": "fitting/valve"}.get(cm["cls"])
        else:
            hints = {base["frag_hint"][f] for f in groups[g]}
            hint = next(iter(hints)) if len(hints) == 1 else None
            hint = hint or None
        tags = sorted({t for f in lf if f in leaf_to_base
                       for t in base_assets[[a["id"] for a in base_assets].index(leaf_to_base[f])]["tags"]}) \
            if lf else []
        rec = cluster_record(len(assets), idx, xyz, rgb, opac, cfg, "object", hint, tags)
        rec["fragments"] = lf
        rec["baseline_ids"] = sorted({leaf_to_base[f] for f in lf if f in leaf_to_base})
        if g in cand_meta:
            cm = cand_meta[g]
            rec["repair"] = {"family": "context", **{k: cm.get(k) for k in ("cid", "region", "cls", "confidence", "model",
                                                                          "renderer", "provisional")},
                             "boundary_status": "provisional" if cm.get("provisional") else "accepted-unreviewed"}
            if cm.get("assembly"):
                rec["assembly"] = {"id": f"{cm['region']}-{cm['assembly']}", "note": "proposed by VLM; unverified"}
            vlm_label = {"pump": "pump", "motor": "motor", "valve": "manual valve/handwheel",
                         "control valve": "control valve", "vessel": "tank/vessel",
                         "sensor": "pressure transmitter/gauge"}.get(cm["cls"])
            if vlm_label and not cm.get("provisional"):        # class confidence kept separate from boundary
                rec["vlm_proposal"] = {"label": vlm_label, "confidence": cm["confidence"], "model": cm["model"],
                                       "reason": f"contextual proposal {cm['cid']} ({cm['region']}) + SAM mask support",
                                       "source": "featherless_vision"}
        elif len(groups[g]) > 1:
            rec["repair"] = {"family": "geometry", "leaves": groups[g]}
        labels[idx] = len(assets)
        assets.append(rec)
    np.savez_compressed(out / "segment_points.npz", xyz=base["xyz"], rgb=base["rgb"], opacity=base["opacity"],
                        labels=labels, labels_baseline=base_lab, normals=base["normals"], frag=frag,
                        frag_hint=base["frag_hint"], frag_axis=base["frag_axis"],
                        label_ids=np.array([a["id"] for a in assets]), src_rows=base["src_rows"],
                        src_vox=base["src_vox"], src_dist=base["src_dist"])
    (out / "assets.json").write_text(json.dumps(assets, indent=1), encoding="utf-8")
    slim = [{k: v for k, v in e.items() if k != "moved_voxels"} for e in log]
    (out / "repairs.json").write_text(json.dumps({"variant": variant, "base": "out_baseline",
                                                  "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                  "edits": slim}, indent=1), encoding="utf-8")
    np.savez_compressed(out / "repairs_voxels.npz", **{f"edit_{i}": np.array(e["moved_voxels"], np.int64)
                                                      for i, e in enumerate(log) if "moved_voxels" in e})
    man = ROOT / "out_baseline" / "segment_manifest.json"
    if man.exists():
        shutil.copy(man, out / "segment_manifest.json")
    print(f"[build] {out_dir}: {sum(a['kind'] == 'object' for a in assets)} objects "
          f"({sum('repair' in a for a in assets)} repaired) -> assets.json, segment_points.npz, repairs.json")


def run(variant: str) -> None:
    cfg = load_config()
    base = load_base("out_baseline")
    leaves = leaves_of(base)
    groups, glog = geo_group(base, leaves, cfg["repair"]["geometry"])
    if variant == "geo":
        vg = np.full(len(base["xyz"]), -1, np.int64)
        for gi, g in enumerate(groups):
            for f in g:
                vg[leaves[f]] = gi
        build(base, vg, groups, {}, cfg, "out_geo", glog, "geometry-only")
        return
    from pipeline.propose import PDIR, check_fresh
    cset = variant.split(":", 1)[1] if ":" in variant else cfg["repair"]["context"]["set"]
    seg_dir = ROOT / cfg["propose"]["seg_dir"]
    vg, clog, cm = context_edits(base, groups, leaves, cfg, PDIR / cset,
                                 freshness=lambda info: check_fresh(info, seg_dir))
    print(f"[context:{cset}] {sum(e['op'] == 'context_repair' for e in clog)} repairs "
          f"({sum(e.get('status') == 'provisional' for e in clog)} provisional), "
          f"{sum(e['op'].startswith('reject') for e in clog)} rejected")
    renderer = cset
    out_dir = cfg["paths"]["out_dir"] if cset == cfg["repair"]["context"]["set"] else f"out_ctx_{cset}"
    build(base, vg, groups, cm, cfg, out_dir, glog + clog, f"geometry+context ({renderer})")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "context")
