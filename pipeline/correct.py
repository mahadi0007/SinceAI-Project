"""Guided manual boundary correction: reviewer prompts -> SAM mask -> visible 3D support -> reversible edit.

Reuses the automatic pipeline instead of duplicating it:
  * clean calibrated renders + buffers         pipeline.propose (render_region / load_views / render_meta)
  * SAM with cached image embeddings           pipeline.propose.sam_embedding / sam_predict
  * contribution-weighted visible support      pipeline.propose.view_support (EWA alpha * transmittance)
  * whole-vs-subset claim rule                 pipeline.repair.claim_support (+ largest_components)

Identity is kept apart on purpose:
  component  : stable id MC-0001 (this boundary), optional class (default "unknown")
  assembly   : optional assembly id (e.g. pump + motor), never implied by class
  asset      : optional workbook tag, status "unassigned" / "unverified" until a real-scene method confirms it

Storage (never touches the PLY or the baseline/base segmentation):
  review/corrections/index.json                     components, drafts, revisions, accepted stacks, event log
  review/corrections/<cid>/draft/<view>.png          current draft mask per view (prompts in index.json)
  review/corrections/<cid>/rev_NNN/                  frozen revision: record.json, masks, selection.npz
  <out_seg>/ (default out_manual/)                   materialised ONLY by apply(): base labels + accepted
                                                     corrections, labels_before_manual, membership history,
                                                     manual_manifest.json (downstream artifacts marked stale)
Undo pops the component's accepted stack (the previous accepted revision, or none, becomes active again);
apply() always replays from the untouched base, so undo survives restarts.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pipeline.common import ROOT

Cfg = dict[str, Any]

DEFAULTS = {
    "base_seg": "out",                 # segmentation version being corrected (read-only here)
    "out_seg": "out_manual",           # separate version written by apply()
    "store": "review/corrections",
    "renderer": "ewa",
    "port": 8091,
    "viser_port": 8081,                # standalone mode only (the main viewer embeds the tool instead)
    "classes": ["unknown", "pump", "motor", "valve", "control valve", "sensor", "vessel", "pipe run", "other"],
    "two_view_min_agree": 0.3,         # >= this share of the selection seen AND inside the mask in >= 2 views ...
    "max_disagreement": 0.15,          # ... and disputed / (selected + disputed) <= this -> "two-view"
    "guide_max_points": 6000,
    "rank_depth_tol": 0.05,            # view ranking only (approximate depth test on the median depth buffer)
    "uncertain_radius": 0.08,          # unseen voxels of touched objects within this of the selection = uncertain
}
DOWNSTREAM = ["assets_labeled.json", "scene_graph.json", "scene_anchors.csv", "path.json", "navgrid.npz",
              "navgrid.png", "tour.json", "tour.png", "label_contact_sheet.png"]
REGENERATE = ("VALMET_OUT={seg} python -m pipeline.label  ->  pipeline.navgrid  ->  pipeline.mapping  ->  "
              "pipeline.tour (label is paid: keep the pilot limits)")


def ccfg(cfg: Cfg) -> Cfg:
    return {**DEFAULTS, **(cfg.get("correct") or {})}


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:20]


def sha_json(o: Any) -> str:
    return sha_bytes(json.dumps(o, sort_keys=True, default=str).encode())


_file_sha: dict[tuple[str, int, int], str] = {}


def file_sha(p: Path) -> str:
    st = p.stat()
    k = (str(p), st.st_size, st.st_mtime_ns)
    if k not in _file_sha:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        _file_sha[k] = h.hexdigest()[:20]
    return _file_sha[k]


def mask_sha(m: np.ndarray) -> str:
    return sha_bytes(np.packbits(np.asarray(m, bool)).tobytes() + str(m.shape).encode())


def mask_png(m: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray(np.asarray(m, bool).astype(np.uint8) * 255).save(b, "PNG")
    return b.getvalue()


def read_mask(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("L")) > 127


def safe(key: str) -> str:
    return key.replace("/", "_")


# ---------- prompts ----------

def clean_prompt(box: Any, pos: Any, neg: Any, W: int, H: int) -> dict:
    """Validate reviewer prompts given in ORIGINAL image pixel coordinates (the browser converts from its
    displayed size). Box corners are ordered and clipped; points must lie inside the image."""
    out: dict[str, Any] = {"box": None, "pos": [], "neg": []}
    if box is not None:
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            raise ValueError("box must be [x0, y0, x1, y1]")
        x0, y0, x1, y1 = (float(v) for v in box)
        if not all(np.isfinite([x0, y0, x1, y1])):
            raise ValueError("box has non-finite values")
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        x0, x1 = max(0.0, x0), min(W - 1.0, x1)
        y0, y1 = max(0.0, y0), min(H - 1.0, y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            raise ValueError("box is smaller than 2 px (or outside the image)")
        out["box"] = [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]
    for name, pts in (("pos", pos), ("neg", neg)):
        for p in pts or []:
            x, y = float(p[0]), float(p[1])
            if not (np.isfinite(x) and np.isfinite(y) and 0 <= x <= W - 1 and 0 <= y <= H - 1):
                raise ValueError(f"{name} point {p} outside the {W}x{H} image")
            out[name].append([round(x, 2), round(y, 2)])
    if out["box"] is None and not out["pos"]:
        raise ValueError("draw a box or add at least one include point")
    return out


def sam_prompt(pr: dict) -> dict:
    return {"box": pr["box"], "pts": pr["pos"] + pr["neg"], "lbl": [1] * len(pr["pos"]) + [0] * len(pr["neg"])}


# ---------- views ----------

def view_key(rid: str, k: int) -> str:
    return f"{rid}/v{k}"


def parse_key(key: str) -> tuple[str, int]:
    rid, v = key.rsplit("/", 1)
    return rid, int(v[1:])


def list_views(renderer: str) -> list[dict]:
    """Every existing calibrated view of `renderer` with a clean image + buffers, with freshness."""
    from pipeline import propose as pp
    regs = pp.regions()
    out = []
    for rid, r in regs.items():
        d = pp.PDIR / rid / renderer
        if not (d / "render_meta.json").exists():
            continue
        meta = json.loads((d / "render_meta.json").read_text(encoding="utf-8"))
        stale = None if meta["cams_key"] == pp.cams_key(pp.region_cameras(rid)) else \
            "cameras in regions.yaml changed since render (re-run pipeline.propose render)"
        for k, c in enumerate(meta["cameras"]):
            if (d / f"view{k}.png").exists() and (d / f"view{k}_buffers.npz").exists():
                out.append({"key": view_key(rid, k), "rid": rid, "k": k, "name": r.get("name", ""),
                            "W": c["W"], "H": c["H"], "stale": stale})
    return out


def view_hashes(rid: str, k: int, renderer: str) -> dict:
    from pipeline import propose as pp
    d = pp.PDIR / rid / renderer
    meta = pp.render_meta(rid, renderer)
    cam = meta["cameras"][k]
    return {"render_key": meta["render_key"], "cams_key": meta["cams_key"], "cam_sha": sha_json(cam),
            "camera": cam, "image_sha": file_sha(d / f"view{k}.png"),
            "buffers_sha": file_sha(d / f"view{k}_buffers.npz"), "W": cam["W"], "H": cam["H"]}


def check_view_fresh(stored: dict, rid: str, k: int, renderer: str) -> str | None:
    """Reason a stored view reference no longer matches what is on disk / in regions.yaml, or None."""
    from pipeline import propose as pp
    try:
        now = view_hashes(rid, k, renderer)
    except (FileNotFoundError, IndexError, KeyError) as e:
        return f"view {rid}/v{k} missing ({e})"
    if now["cams_key"] != pp.cams_key(pp.region_cameras(rid)):
        return f"{rid}: cameras in regions.yaml changed since render"
    for f, what in (("render_key", "render"), ("cam_sha", "camera"), ("image_sha", "clean image"),
                    ("buffers_sha", "visibility buffers")):
        if stored.get(f) != now[f]:
            return f"{rid}/v{k}: {what} changed since the prompts were drawn"
    return None


# ---------- engine: scene, views, SAM, support ----------

class Engine:
    """Lazy scene + view cache. All heavy state is read-only; SAM runs under its own lock."""

    def __init__(self, cfg: Cfg) -> None:
        self.cfg = cfg
        self.cc = ccfg(cfg)
        self._scene: Any = None
        self._views: OrderedDict[str, tuple] = OrderedDict()
        self._lock = threading.RLock()
        self.sam_lock = threading.Lock()

    # scene
    def scene(self) -> Any:
        with self._lock:
            if self._scene is None:
                from pipeline.scene import load_scene, seg_revision
                sc = load_scene(self.cfg, self.cc["base_seg"])
                self.seg_rev = seg_revision(sc.seg_dir)
                a = sc.assets
                rc = self.cfg["repair"]["context"]
                self.is_structure = np.array([x["kind"] == "structure" for x in a])
                long_pipe = np.array([x["kind"] == "object" and x.get("hint") == "pipe segment"
                                      and max(x["extent"]) >= rc["pipe_protect_min_len"] for x in a])
                self.protected = self.is_structure | long_pipe
                self._scene = sc
            return self._scene

    def base_rev(self) -> str:
        self.scene()
        return self.seg_rev

    # views
    def view(self, key: str) -> tuple[Any, np.ndarray, dict]:
        """(camera, clean RGB (read-only array), buffers) for 'RV-01/v0'."""
        from pipeline import propose as pp
        from pipeline.views import Camera
        with self._lock:
            if key in self._views:
                self._views.move_to_end(key)
                return self._views[key]
            rid, k = parse_key(key)
            d = pp.PDIR / rid / self.cc["renderer"]
            meta = pp.render_meta(rid, self.cc["renderer"])
            cam = Camera.from_json(meta["cameras"][k])
            rgb = np.asarray(Image.open(d / f"view{k}.png").convert("RGB")).copy()
            rgb.setflags(write=False)                       # SAM input is immutable
            buf = dict(np.load(d / f"view{k}_buffers.npz"))
            self._views[key] = (cam, rgb, buf)
            while len(self._views) > 4:
                self._views.popitem(last=False)
            return self._views[key]

    def depth_only(self, key: str) -> tuple[Any, np.ndarray]:
        from pipeline import propose as pp
        from pipeline.views import Camera
        rid, k = parse_key(key)
        meta = pp.render_meta(rid, self.cc["renderer"])
        with np.load(pp.PDIR / rid / self.cc["renderer"] / f"view{k}_buffers.npz") as z:
            return Camera.from_json(meta["cameras"][k]), z["depth"]

    # SAM
    def mask(self, key: str, pr: dict) -> tuple[np.ndarray, float]:
        from pipeline import propose as pp
        cam, rgb, _ = self.view(key)
        name = self.cfg["propose"]["sam_model"]
        before = rgb.copy()
        with self.sam_lock:
            emb, base, img = pp.sam_embedding(rgb, name)    # cached per (model, image hash)
            m, s = pp.sam_predict(emb, base, img, sam_prompt(pr), name)
        assert np.array_equal(before, rgb), "SAM input image was modified"
        if m.shape != (cam.H, cam.W):
            raise RuntimeError(f"SAM mask {m.shape} does not match the original image {(cam.H, cam.W)}")
        return m, s

    # support
    def support(self, masks: dict[str, np.ndarray], other_claims: dict[str, np.ndarray] | None = None) -> dict:
        from pipeline import propose as pp
        from pipeline.repair import claim_support, largest_components
        sc = self.scene()
        rc = self.cfg["repair"]["context"]
        pc = self.cfg["propose"]
        nvox = len(sc.vox_xyz)
        return compute_selection(sc, masks, {k: self.view(k) for k in masks}, self.protected, self.is_structure,
                                 rc, pc, self.cc, other_claims or {}, pp.view_support, claim_support,
                                 largest_components, nvox)

    def guide(self, key: str, sel: dict) -> dict:
        """Current selection projected into another view: visible points (contribution weight > seen_eps in
        THIS view's buffers) vs hidden ones, for the reviewer to confirm or correct."""
        from pipeline import propose as pp
        sc = self.scene()
        cam, _, buf = self.view(key)
        vis = pp.view_visibility(sc.g_vox, len(sc.vox_xyz), buf)
        eps = self.cfg["repair"]["context"]["seen_eps"]
        out: dict[str, Any] = {"view": key, "W": cam.W, "H": cam.H}
        for name in ("selected", "disputed"):
            idx = np.asarray(sel.get(name, []), np.int64)
            if not len(idx):
                out[name] = {"visible": [], "hidden": 0}
                continue
            u, v, z = cam.project(sc.vox_xyz[idx])
            inside = (z > 0) & (u >= 0) & (u <= cam.W - 1) & (v >= 0) & (v <= cam.H - 1)
            visible = inside & (vis[idx] > eps)
            pts = np.c_[u[visible], v[visible]]
            if len(pts) > self.cc["guide_max_points"]:
                pts = pts[np.random.default_rng(0).choice(len(pts), self.cc["guide_max_points"], replace=False)]
            out[name] = {"visible": np.round(pts, 1).tolist(), "n_visible": int(visible.sum()),
                         "hidden": int((~visible).sum()), "out_of_frame": int((~inside).sum())}
        return out

    def rank_views(self, sel_idx: np.ndarray, exclude: set[str]) -> list[dict]:
        """Other existing views ranked by how many selected voxels they see (approximate: centre projection +
        median-depth test; the exact contribution-weighted check runs when the view is opened)."""
        sc = self.scene()
        P = sc.vox_xyz[np.asarray(sel_idx, np.int64)]
        res = []
        if not len(P):
            return res
        for v in list_views(self.cc["renderer"]):
            if v["key"] in exclude or v["stale"]:
                continue
            cam, dep = self.depth_only(v["key"])
            u, vv, z = cam.project(P)
            ins = (z > 0) & (u >= 0) & (u <= cam.W - 1) & (vv >= 0) & (vv <= cam.H - 1)
            if not ins.any():
                continue
            ui, vi = np.round(u[ins]).astype(int), np.round(vv[ins]).astype(int)
            seen = z[ins] <= dep[vi, ui] + self.cc["rank_depth_tol"]
            if seen.sum():
                res.append({"key": v["key"], "visible_frac": round(float(seen.sum()) / len(P), 3),
                            "visible": int(seen.sum())})
        return sorted(res, key=lambda r: -r["visible"])


def compute_selection(sc: Any, masks: dict[str, np.ndarray], views: dict[str, tuple], protected: np.ndarray,
                      is_structure: np.ndarray, rc: Cfg, pc: Cfg, cc: Cfg, other_claims: dict[str, np.ndarray],
                      view_support: Any, claim_support: Any, largest_components: Any, nvox: int) -> dict:
    """Combine per-view mask support into one proposed selection and report disagreement explicitly.
      per view: in_v = seen & pos_v > 0 & pos_v >= neg_v ; out_v = seen & not in_v
      claim     : claim_support on the summed pos/neg (whole group only with support+coverage+spatial)
      disputed  : claimed voxels inside the mask in one view but visibly outside it in another -> NOT selected
      inferred  : unseen voxels of a group claimed whole (never verified; counted separately)
      uncertain : unseen voxels of partially claimed groups -> stay with their old object
    Structure voxels (walls/floor/ceiling planes) are never moved."""
    xyz = sc.vox_xyz
    G = len(sc.assets)
    vox_group = np.asarray(sc.vox_label, np.int64)
    pos = np.zeros(nvox)
    neg = np.zeros(nvox)
    n_in = np.zeros(nvox, np.int32)
    n_out = np.zeros(nvox, np.int32)
    per_view = []
    for key, m in masks.items():
        cam, _, buf = views[key]
        res = view_support(sc.xyz, sc.g_vox, nvox, cam, buf, m, pc)
        if res is None:
            per_view.append({"view": key, "empty": True})
            continue
        p, q, st = res
        pos += p
        neg += q
        seen = (p + q) > rc["seen_eps"]
        inn = seen & (p > 0) & (p >= q)
        n_in += inn
        n_out += seen & ~inn
        per_view.append({"view": key, **st, "voxels_in": int(inn.sum())})
    names = {i: a["id"] for i, a in enumerate(sc.assets)}
    cl = claim_support(pos, neg, vox_group, G, protected, xyz, rc, protected_subset=True,
                       describe=lambda gi: {"object": names[gi], "label": sc.assets[gi].get("label"),
                                            "kind": sc.assets[gi]["kind"]})
    take = cl["take"]
    struct_vox = (vox_group >= 0) & is_structure[np.maximum(vox_group, 0)]
    n_struct = int((take & struct_vox).sum())
    take &= ~struct_vox
    idx = np.where(take)[0]
    dropped = 0
    if len(idx) >= 3:
        kept = largest_components(xyz, idx, rc["cc_eps"], rc["cc_min_frac"])
        dropped = len(idx) - len(kept)
        take[:] = False
        take[kept] = True
    disputed = take & (n_in > 0) & (n_out > 0)
    final = take & ~disputed
    n_seen = n_in + n_out
    inferred = final & cl["inferred"]
    single = final & (n_seen == 1)
    multi = final & (n_seen >= 2)
    touched = {e["group"] for e in cl["members"] + cl["splits"]}
    in_touched = np.isin(vox_group, list(touched)) if touched else np.zeros(nvox, bool)
    uncertain = in_touched & ~final & ~cl["seen"]
    sel = np.where(final)[0]
    if uncertain.any() and len(sel):                   # only unseen points next to the selection are in doubt
        from scipy.spatial import cKDTree
        u = np.where(uncertain)[0]
        dd, _ = cKDTree(xyz[sel]).query(xyz[u], distance_upper_bound=cc["uncertain_radius"])
        uncertain[:] = False
        uncertain[u[np.isfinite(dd)]] = True
    n_sel, n_disp = len(sel), int(disputed.sum())
    views_used = sum(1 for v in per_view if not v.get("empty"))
    agree = float(multi.sum()) / max(n_sel, 1)
    disagree = n_disp / max(n_sel + n_disp, 1)
    if views_used >= 2 and agree >= cc["two_view_min_agree"] and disagree <= cc["max_disagreement"]:
        level = "two-view"
    elif views_used >= 2:
        level = "views-disagree" if disagree > cc["max_disagreement"] else "two-view-weak-overlap"
    else:
        level = "single-view-provisional"
    warnings = []
    if n_sel < rc["min_candidate_voxels"]:
        warnings.append(f"only {n_sel} voxels selected (< {rc['min_candidate_voxels']}): cannot accept")
    if n_sel:
        ext = np.ptp(xyz[sel], 0)
        if ext.max() > rc["max_extent"]:
            warnings.append(f"selection extent {ext.round(2).tolist()} exceeds {rc['max_extent']} units: check for "
                            "background or neighbouring equipment")
    for s in cl["splits"]:
        if s["coverage"] < rc["min_coverage"]:
            warnings.append(f"{s['object']}: only {s['coverage']:.0%} observed in these views, so only its visible "
                            f"supported part moves; {s['unseen_left_uncertain']} unseen voxels stay (uncertain)")
    if n_disp:
        warnings.append(f"{n_disp} voxels disputed between views (inside the mask in one view, outside in "
                        "another): excluded; add include/exclude points to resolve")
    if n_struct:
        warnings.append(f"{n_struct} wall/floor/ceiling voxels inside the mask left untouched")
    if inferred.any():
        warnings.append(f"{int(inferred.sum())} unseen voxels come with whole objects (support, coverage and "
                        "spatial checks passed); they are inferred, not verified")
    overlaps = {c: int(np.isin(sel, v, assume_unique=False).sum()) for c, v in other_claims.items()}
    overlaps = {c: n for c, n in overlaps.items() if n}
    for c, n in overlaps.items():
        warnings.append(f"{n} voxels are already in accepted component {c}; accepting moves them here")
    if level != "two-view":
        warnings.append({"single-view-provisional": "single view: provisional until confirmed in a second view",
                         "views-disagree": "views disagree: resolve in the second view before relying on it",
                         "two-view-weak-overlap": "few voxels confirmed by both views: mostly single-view evidence"}[level])
    bb = (xyz[sel].min(0).round(4).tolist(), xyz[sel].max(0).round(4).tolist()) if n_sel else None
    return {"selected": sel, "disputed": np.where(disputed)[0], "inferred": np.where(inferred)[0],
            "single_view": np.where(single)[0], "uncertain": np.where(uncertain)[0],
            "stats": {"selected": n_sel, "disputed": n_disp, "inferred_unseen": int(inferred.sum()),
                      "single_view_only": int(single.sum()), "multi_view_agree": int(multi.sum()),
                      "uncertain_left": int(uncertain.sum()), "speckle_dropped": dropped,
                      "structure_untouched": n_struct, "views_used": views_used, "agree_frac": round(agree, 3),
                      "disagree_frac": round(disagree, 3), "evidence_level": level, "bbox": bb,
                      "overlaps": overlaps},
            "per_view": per_view, "members": cl["members"], "splits": cl["splits"],
            "skipped": cl["skipped"], "warnings": warnings}


# ---------- persistent store ----------

class Store:
    """Corrections live in their own folder; every action is appended to an event log."""

    def __init__(self, cfg: Cfg, root: Path | None = None) -> None:
        self.cc = ccfg(cfg)
        self.root = Path(root) if root else ROOT / self.cc["store"]
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.path = self.root / "index.json"
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else \
            {"next": 1, "components": {}, "events": [], "applied": None}

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def event(self, op: str, cid: str | None, **kw: Any) -> None:
        self.data["events"].append({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "op": op, "cid": cid, **kw})

    def comp(self, cid: str) -> dict:
        if cid not in self.data["components"]:
            raise KeyError(f"unknown component {cid}")
        return self.data["components"][cid]

    def new(self, region: str, component_class: str = "unknown", assembly_id: str | None = None,
            asset_tag: str | None = None, note: str = "") -> dict:
        with self.lock:
            cid = f"MC-{self.data['next']:04d}"
            self.data["next"] += 1
            c = {"id": cid, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "region": region,
                 "component_class": component_class or "unknown", "assembly_id": assembly_id or None,
                 "asset": {"tag": asset_tag or None, "status": "unverified" if asset_tag else "unassigned",
                           "note": "workbook identity is separate from this boundary; not verified by it"},
                 "note": note, "draft": {"views": {}, "preview": None}, "revisions": [], "accepted_stack": []}
            self.data["components"][cid] = c
            self.event("create", cid, region=region)
            self.save()
            return c

    def update_meta(self, cid: str, **kw: Any) -> dict:
        with self.lock:
            c = self.comp(cid)
            ch = {}
            for k in ("component_class", "assembly_id", "note"):
                if k in kw and kw[k] is not None and kw[k] != c.get(k):
                    ch[k] = [c.get(k), kw[k] or None]
                    c[k] = kw[k] or (None if k != "component_class" else "unknown")
            if "asset_tag" in kw and kw["asset_tag"] is not None and (kw["asset_tag"] or None) != c["asset"]["tag"]:
                ch["asset_tag"] = [c["asset"]["tag"], kw["asset_tag"] or None]
                c["asset"]["tag"] = kw["asset_tag"] or None
                c["asset"]["status"] = "unverified" if c["asset"]["tag"] else "unassigned"
            if ch:
                self.event("meta", cid, changes=ch)
                self.save()
            return c

    def cdir(self, cid: str) -> Path:
        d = self.root / cid
        d.mkdir(exist_ok=True)
        return d

    def set_draft_view(self, cid: str, key: str, entry: dict, mask: np.ndarray) -> None:
        with self.lock:
            c = self.comp(cid)
            d = self.cdir(cid) / "draft"
            d.mkdir(exist_ok=True)
            (d / f"{safe(key)}.png").write_bytes(mask_png(mask))
            c["draft"]["views"][key] = {**entry, "mask_file": f"{cid}/draft/{safe(key)}.png",
                                        "mask_sha": mask_sha(mask)}
            c["draft"]["preview"] = None                      # 3D preview is out of date
            self.save()

    def drop_draft_view(self, cid: str, key: str) -> None:
        with self.lock:
            c = self.comp(cid)
            if c["draft"]["views"].pop(key, None) is not None:
                f = self.cdir(cid) / "draft" / f"{safe(key)}.png"
                if f.exists():
                    f.unlink()
                c["draft"]["preview"] = None
                self.save()

    def draft_masks(self, cid: str) -> dict[str, np.ndarray]:
        c = self.comp(cid)
        out = {}
        for key, e in c["draft"]["views"].items():
            m = read_mask(self.root / e["mask_file"])
            if mask_sha(m) != e["mask_sha"]:
                raise ValueError(f"{cid} {key}: draft mask file was modified")
            out[key] = m
        return out

    def draft_hash(self, cid: str) -> str:
        c = self.comp(cid)
        return sha_json({k: [e["prompt_hash"], e["mask_sha"]] for k, e in sorted(c["draft"]["views"].items())})

    def active_rev(self, cid: str) -> dict | None:
        c = self.comp(cid)
        if not c["accepted_stack"]:
            return None
        n = c["accepted_stack"][-1]
        return next(r for r in c["revisions"] if r["rev"] == n)

    def freeze(self, cid: str, status: str, extra: dict, selection: dict | None) -> dict:
        """Copy the draft (prompts + masks) into rev_NNN and record it."""
        with self.lock:
            c = self.comp(cid)
            n = len(c["revisions"]) + 1
            d = self.cdir(cid) / f"rev_{n:03d}"
            d.mkdir(exist_ok=True)
            views = []
            for key, e in c["draft"]["views"].items():
                src = self.root / e["mask_file"]
                dst = d / f"mask_{safe(key)}.png"
                dst.write_bytes(src.read_bytes())
                views.append({**e, "mask_file": f"{cid}/rev_{n:03d}/{dst.name}"})
            rec = {"rev": n, "status": status, "t": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "component": {k: copy.deepcopy(c.get(k)) for k in ("component_class", "assembly_id", "asset",
                                                                       "note", "region")},
                   "views": views, **extra}
            if selection is not None:
                arrs = {k: np.asarray(selection[k], np.int64) for k in
                        ("selected", "disputed", "inferred", "single_view", "uncertain")}
                np.savez_compressed(d / "selection.npz", **arrs)
                rec["selection_file"] = f"{cid}/rev_{n:03d}/selection.npz"
                rec["selection_sha"] = sha_bytes(arrs["selected"].tobytes())
            (d / "record.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
            c["revisions"].append(rec)
            return rec

    def clear_draft(self, cid: str) -> None:
        c = self.comp(cid)
        for e in c["draft"]["views"].values():
            f = self.root / e["mask_file"]
            if f.exists():
                f.unlink()
        c["draft"] = {"views": {}, "preview": None}

    def selection(self, rec: dict) -> dict[str, np.ndarray]:
        with np.load(self.root / rec["selection_file"]) as z:
            out = {k: z[k] for k in z.files}
        if sha_bytes(out["selected"].astype(np.int64).tobytes()) != rec["selection_sha"]:
            raise ValueError(f"{rec['selection_file']} was modified after acceptance")
        return out

    def accepted(self) -> list[tuple[str, dict]]:
        """Active accepted revision per component, in acceptance order (later acceptance wins overlaps)."""
        out = []
        for cid, c in self.data["components"].items():
            r = self.active_rev(cid)
            if r is not None:
                out.append((cid, r))
        return sorted(out, key=lambda t: (t[1]["accepted_at"], t[0]))

    def status(self, cid: str) -> dict:
        c = self.comp(cid)
        r = self.active_rev(cid)
        ap = self.data.get("applied") or {}
        applied_rev = (ap.get("revisions") or {}).get(cid)
        st = {"draft_views": sorted(c["draft"]["views"]), "preview_current": bool(c["draft"]["preview"]),
              "accepted_rev": r["rev"] if r else None, "evidence_level": r["stats"]["evidence_level"] if r else None,
              "applied_rev": applied_rev}
        st["state"] = ("draft" if c["draft"]["views"] and not r else
                       "accepted, not applied" if r and applied_rev != r["rev"] else
                       "applied" if r else
                       "applied state out of date (undone)" if applied_rev else
                       "rejected" if c["revisions"] and c["revisions"][-1]["status"] == "rejected" else
                       "original assignment (correction undone)" if any(x["status"] == "undone" for x in c["revisions"])
                       else "empty")
        if c["draft"]["views"] and r:
            st["state"] += " + new draft"
        return st


# ---------- accept / reject / undo ----------

def validate_rev_views(views: list[dict], renderer: str) -> list[str]:
    errs = []
    for e in views:
        rid, k = parse_key(e["view"])
        why = check_view_fresh(e, rid, k, renderer)
        if why:
            errs.append(why)
    return errs


def accept(store: Store, eng: Engine, cid: str, preview_hash: str, accept_provisional: bool = False) -> dict:
    with store.lock:
        c = store.comp(cid)
        if not c["draft"]["views"]:
            raise ValueError("nothing to accept: preview a mask first")
        if c["draft"]["preview"] is None or c["draft"]["preview"]["hash"] != store.draft_hash(cid) \
                or preview_hash != c["draft"]["preview"]["hash"]:
            raise ValueError("3D preview is out of date: run 'Preview in 3D' again before accepting")
        errs = validate_rev_views([{**e, "view": k} for k, e in c["draft"]["views"].items()], eng.cc["renderer"])
        if errs:
            raise ValueError("stale prompts: " + "; ".join(errs))
        if c["draft"]["preview"]["base_seg_rev"] != eng.base_rev():
            raise ValueError("base segmentation changed since the preview")
        others = {o: store.selection(r)["selected"] for o, r in store.accepted() if o != cid}
        sel = eng.support(store.draft_masks(cid), others)
        st = sel["stats"]
        if st["selected"] < eng.cfg["repair"]["context"]["min_candidate_voxels"]:
            raise ValueError(f"selection too small ({st['selected']} voxels)")
        if st["evidence_level"] != "two-view" and not accept_provisional:
            raise ValueError(f"evidence is '{st['evidence_level']}': tick 'accept as provisional' to accept it "
                             "explicitly (unseen geometry stays unverified)")
        rec = store.freeze(cid, "accepted", {
            "accepted_at": time.time(), "base_seg": eng.cc["base_seg"], "base_seg_rev": eng.base_rev(),
            "renderer": eng.cc["renderer"], "sam_model": eng.cfg["propose"]["sam_model"],
            "boundary_status": "accepted-two-view" if st["evidence_level"] == "two-view" else "accepted-provisional",
            "stats": st, "members": sel["members"], "splits": sel["splits"], "warnings": sel["warnings"],
            "per_view": sel["per_view"]}, sel)
        c["accepted_stack"].append(rec["rev"])
        store.clear_draft(cid)
        store.event("accept", cid, rev=rec["rev"], evidence=st["evidence_level"], voxels=st["selected"])
        store.save()
        return rec


def reject(store: Store, cid: str, reason: str = "") -> dict:
    with store.lock:
        c = store.comp(cid)
        if not c["draft"]["views"]:
            raise ValueError("no pending draft to reject")
        rec = store.freeze(cid, "rejected", {"reason": reason}, None)
        store.clear_draft(cid)
        store.event("reject", cid, rev=rec["rev"])
        store.save()
        return rec


def undo(store: Store, cid: str) -> dict:
    """Pop the active accepted revision; the previous one (or the original assignment) becomes active."""
    with store.lock:
        c = store.comp(cid)
        if not c["accepted_stack"]:
            raise ValueError(f"{cid} has no accepted correction to undo")
        n = c["accepted_stack"].pop()
        r = next(r for r in c["revisions"] if r["rev"] == n)
        r["status"] = "undone"
        r["undone_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        prev = c["accepted_stack"][-1] if c["accepted_stack"] else None
        store.event("undo", cid, rev=n, now_active=prev)
        store.save()
        return {"undone": n, "active": prev}


# ---------- materialise a separate segmentation version ----------

def apply(store: Store, cfg: Cfg, eng: Engine | None = None) -> dict:
    """Replay every active accepted correction over the UNTOUCHED base segmentation into out_seg.
    Refuses (writes nothing) if any correction is stale. Downstream artifacts of out_seg are removed and
    listed as stale in manual_manifest.json until they are regenerated against this version."""
    from pipeline.scene import seg_revision
    from pipeline.segment import cluster_record
    cc = ccfg(cfg)
    base_dir, out_dir = ROOT / cc["base_seg"], ROOT / cc["out_seg"]
    with store.lock:
        base_rev = seg_revision(base_dir)
        acc = store.accepted()
        errs = []
        for cid, r in acc:
            if r["base_seg_rev"] != base_rev:
                errs.append(f"{cid} rev {r['rev']}: base segmentation {cc['base_seg']} changed "
                            f"({r['base_seg_rev']} -> {base_rev})")
            errs += [f"{cid} rev {r['rev']}: {e}" for e in validate_rev_views(r["views"], r["renderer"])]
        if errs:
            raise ValueError("cannot apply, stale corrections: " + "; ".join(errs))
        s = dict(np.load(base_dir / "segment_points.npz"))
        assets = json.loads((base_dir / "assets.json").read_text(encoding="utf-8"))
        lp = base_dir / "assets_labeled.json"
        if lp.exists():
            lab = json.loads(lp.read_text(encoding="utf-8"))
            if [a["id"] for a in lab] == [a["id"] for a in assets]:
                assets = lab
        labels0 = s["labels"].astype(np.int64)
        owner = np.full(len(labels0), -1, np.int64)        # index into acc
        conflicts = []
        for j, (cid, r) in enumerate(acc):
            sel = store.selection(r)["selected"]
            prev = owner[sel]
            for o in np.unique(prev[prev >= 0]):
                conflicts.append({"voxels": int((prev == o).sum()), "from": acc[o][0], "to": cid,
                                  "rule": "later acceptance wins"})
            owner[sel] = j
        xyz, rgb, op = s["xyz"].astype(np.float64), s["rgb"], s["opacity"]
        changed = np.unique(labels0[(owner >= 0) & (labels0 >= 0)])
        new_assets, new_labels = [], np.full(len(labels0), -1, np.int32)
        absorbed = {}
        remap = {}
        for i, a in enumerate(assets):
            idx = np.where((labels0 == i) & (owner < 0))[0]
            if not len(idx):
                absorbed[a["id"]] = sorted({acc[o][0] for o in np.unique(owner[labels0 == i])})
                continue
            rec = a
            if i in changed:
                rec = cluster_record(0, idx, xyz, rgb, op, cfg, a["kind"], a.get("hint"), a.get("tags", []))
                rec["id"] = a["id"]
                rec.update({k: a[k] for k in ("fragments", "baseline_ids", "repair", "name") if k in a})
                if "label" in a:
                    rec["label_before_manual"] = a.get("label")
                rec["label_stale"] = True
                rec["manual_changes"] = [{"to": acc[o][0], "rev": acc[o][1]["rev"],
                                          "voxels": int(((labels0 == i) & (owner == o)).sum())}
                                         for o in np.unique(owner[(labels0 == i) & (owner >= 0)])]
                rec["n_points_before_manual"] = a["n_points"]
            remap[i] = len(new_assets)
            new_labels[idx] = len(new_assets)
            new_assets.append(rec)
        for j, (cid, r) in enumerate(acc):
            idx = np.where(owner == j)[0]
            if not len(idx):
                continue
            c = store.comp(cid)
            cls = r["component"]["component_class"] or "unknown"
            rec = cluster_record(0, idx, xyz, rgb, op, cfg, "object", None, [])
            rec["id"] = cid
            src = labels0[idx]
            rec["manual"] = {
                "component_id": cid, "revision": r["rev"], "boundary_status": r["boundary_status"],
                "evidence_level": r["stats"]["evidence_level"], "component_class": cls,
                "assembly_id": r["component"]["assembly_id"], "asset": r["component"]["asset"],
                "inferred_unseen_voxels": r["stats"]["inferred_unseen"],
                "uncertain_left_in_source": r["stats"]["uncertain_left"],
                "membership": [{"from": assets[int(o)]["id"] if o >= 0 else "noise", "voxels": int((src == o).sum()),
                                "whole": bool(o >= 0 and not ((labels0 == o) & (owner != j)).any())}
                               for o in np.unique(src)]}
            rec["label"] = cls if cls != "unknown" else "unknown"
            rec["label_source"] = "manual_review"
            rec["fragments"] = sorted({int(f) for f in np.unique(s["frag"][idx]) if f >= 0})
            new_labels[idx] = len(new_assets)
            new_assets.append(rec)
        assert (new_labels >= 0).sum() == ((labels0 >= 0) | (owner >= 0)).sum(), "points lost while applying"
        out_dir.mkdir(exist_ok=True)
        removed = []
        for f in DOWNSTREAM:
            if (out_dir / f).exists():
                (out_dir / f).unlink()
                removed.append(f)
        arrs = {k: v for k, v in s.items() if k not in ("labels", "label_ids")}
        arrs["labels_before_manual"] = labels0.astype(np.int32)
        arrs["label_ids_before_manual"] = s["label_ids"]
        arrs["manual_owner"] = owner.astype(np.int32)
        np.savez_compressed(out_dir / "segment_points.npz", labels=new_labels,
                            label_ids=np.array([a["id"] for a in new_assets]), **arrs)
        (out_dir / "assets.json").write_text(json.dumps(new_assets, indent=1), encoding="utf-8")
        for f in ("segment_manifest.json",):
            if (base_dir / f).exists():
                (out_dir / f).write_bytes((base_dir / f).read_bytes())
        new_rev = seg_revision(out_dir)
        man = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "applied_at": time.time(),
               "base_seg": cc["base_seg"], "base_seg_rev": base_rev, "seg_rev": new_rev,
               "corrections": [{"cid": cid, "rev": r["rev"], "boundary_status": r["boundary_status"],
                                "evidence_level": r["stats"]["evidence_level"],
                                "voxels": int((owner == j).sum())} for j, (cid, r) in enumerate(acc)],
               "absorbed_objects": absorbed, "conflicts": conflicts,
               "objects": {"before": sum(a["kind"] == "object" for a in assets),
                           "after": sum(a["kind"] == "object" for a in new_assets)},
               "downstream_stale": DOWNSTREAM, "removed": removed,
               "regenerate": REGENERATE.format(seg=cc["out_seg"]),
               "note": "base segmentation and PLY untouched; labels_before_manual + manual_owner keep membership"}
        (out_dir / "manual_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
        store.data["applied"] = {"t": man["created"], "seg_rev": new_rev, "base_seg_rev": base_rev,
                                 "revisions": {cid: r["rev"] for cid, r in acc}}
        store.event("apply", None, seg_rev=new_rev, corrections=len(acc))
        store.save()
        return man


def downstream_stale(out_dir: Path, name: str) -> str | None:
    """Reason `name` in out_dir must not be used: it predates the last manual apply (or is missing)."""
    man = Path(out_dir) / "manual_manifest.json"
    if not man.exists():
        return None
    m = json.loads(man.read_text(encoding="utf-8"))
    p = Path(out_dir) / name
    if name in m.get("downstream_stale", []) and (not p.exists() or p.stat().st_mtime < m["applied_at"]):
        return f"{name} is stale after manual corrections were applied ({m['regenerate']})"
    return None
