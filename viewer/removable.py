"""Which scan objects can be removed cleanly? Only these can be moved, removed or duplicated in viewer.app.

1. geometry screen (all objects): size, not a pipe/wall, few of its voxels touching other objects / walls (the
   floor is fine), and a clean hole (few foreign Gaussians left inside its volume when it is removed);
2. SAM check (survivors): render the object from several sides in a real browser (GPU), prompt SAM 2.1 at the
   object (footprint box + centre point) and compare SAM's mask with the object's visible footprint: SAM's object
   must lie inside ours (not split) and ours mostly inside SAM's (not fused), in most views.
Writes out/removable.json, out/removable_rows.npz (splat rows per object, read by viewer.app) and
out/removable_sheet.png (each accepted object: view, SAM mask, after removal). viewer.removable.include skips
the SAM check and keeps exactly those hand-confirmed objects. Those ids belong to the segmentation named in
viewer.removable.include_source; they are mapped to the current segmentation by their points (same voxel points,
IoU >= include_min_iou), never by id. Outputs record the scene revision; previous outputs are backed up first.
Run: python -m viewer.removable   (Chrome or `playwright install chromium`; viewer.bim_view must not be needed)
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from viser.transforms import SO3

from pipeline.bim_provenance import ArtifactMismatch, backup, label_revision, scene_revision
from pipeline.common import ROOT, load_config, out_path
from pipeline.propose import sam_model
from pipeline.views import Camera, render
from viewer.bim_view import BimView

Cfg = dict[str, Any]


def geometry(v: BimView, rc: Cfg) -> list[dict]:
    """Per object (group leader): size, share of its voxels touching other objects / walls, foreign Gaussians left
    in its volume."""
    floor = {k for k, o in enumerate(v.objs) if o["kind"] != "object" and o.get("label") == "floor"}
    seg_lab = v.seg_labels
    code_of = v._code(v._vox(v.pts))
    occ_lab: dict[int, set[int]] = {}                                      # voxel -> labels (objects / -2 structure)
    for c, k, raw in zip(code_of.tolist(), v.grp_lab.tolist(), seg_lab.tolist()):
        if k >= 0:
            occ_lab.setdefault(c, set()).add(k)
        elif raw >= 0 and raw not in floor:
            occ_lab.setdefault(c, set()).add(-2)
    sp = v.splat
    g_code = v._code(v._vox(sp["xyz"]))
    g_floor = np.isin(seg_lab[v.g_nearest], list(floor))
    out = []
    for k in v.src:
        if k != v.src[k] or k in v.pieces:
            continue
        o = v.objs[k]
        P = v.pts[v.grp_lab == k]
        ext = np.ptp(P, 0)
        rec = {"key": k, "id": o["id"], "label": o.get("label"), "points": len(P), "extent_m": ext.round(2).tolist()}
        out.append(rec)
        if (o.get("label") in rc["skip_labels"] or len(P) < rc["min_points"] or ext.max() < rc["min_extent_m"]
                or ext.max() > rc["max_extent_m"]):
            rec["reject"] = "size/type"
            continue
        vox = np.unique(v._vox(P), axis=0)
        own = set(v._code(vox).tolist())
        nb = v._code((vox[:, None] + v.off).reshape(-1, 3)).reshape(len(vox), -1)
        foreign = np.array([[bool(occ_lab.get(c, set()) - {k}) for c in row] for row in nb.tolist()])
        rec["contact"] = round(float(foreign.any(1).mean()), 3)            # own voxels next to another object / wall
        rows = v._rows(k)
        left = np.isin(g_code, list(own)) & ~g_floor
        left[rows] = False
        rec["hole"] = round(int(left.sum()) / max(len(rows), 1), 3)
        if rec["contact"] > rc["max_contact"]:
            rec["reject"] = "contact"
        elif rec["hole"] > rc["max_hole"]:
            rec["reject"] = "hole"
    return out


def look(client: Any, eye: np.ndarray, target: np.ndarray, W: int, fov: float) -> Camera:
    """Point the browser camera at target and return the matching camera (read back: get_render's own pose
    arguments do not match, and the frame must keep the browser's aspect)."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z], 1)                                # camera -> world (OpenCV axes)
    cam_h = client.camera
    cam_h.fov, cam_h.position, cam_h.wxyz = fov, tuple(eye), tuple(SO3.from_matrix(R).wxyz)
    time.sleep(1.0)
    H = int(round(W / cam_h.aspect))
    f = 0.5 * H / np.tan(cam_h.fov / 2)
    return Camera(np.asarray(cam_h.position), SO3(np.asarray(cam_h.wxyz)).as_matrix().T, f, f, (W - 1) / 2,
                  (H - 1) / 2, W, H)


def frame(client: Any, cam: Camera, ref: np.ndarray | None = None, min_ncc: float = 0.4) -> np.ndarray | None:
    """Browser frame; with ref (our render of the same camera) retried until it matches (the browser may lag)."""
    def small(a: np.ndarray) -> np.ndarray:
        g = np.asarray(Image.fromarray(a).convert("L").resize((80, 50)), float).ravel()
        return (g - g.mean()) / (g.std() + 1e-6)
    for _ in range(4):
        img = np.asarray(client.get_render(height=cam.H, width=cam.W, transport_format="png", timeout=20))[..., :3]
        if ref is None or float(np.mean(small(img) * small(ref))) >= min_ncc:
            return img
        time.sleep(1.0)
    return None


def sam_check(v: BimView, client: Any, rec: dict, rc: Cfg, cfg: Cfg) -> dict:
    """SAM in rc['views'] directions around the object; IoU of the best mask with the object's visible footprint."""
    import torch
    proc, model = sam_model(cfg["propose"]["sam_model"])
    k = rec["key"]
    fov = np.radians(rc["fov_deg"])
    c = v.centre[k]
    ext = np.array(rec["extent_m"])
    dist = max(rc["min_dist_m"], rc["dist_factor"] * float(np.linalg.norm(ext)))
    rows, xyz, key = v._shown()
    sp = v.splat
    W, H = rc["width"], int(round(rc["width"] / client.camera.aspect))
    f = 0.5 * H / np.tan(fov / 2)
    cands = []
    for i in range(rc["candidates"]):                                       # keep views from free space
        a = 2 * np.pi * i / rc["candidates"] + 0.4
        eye = c + dist * np.array([np.cos(a) * np.cos(rc["elev"]), np.sin(a) * np.cos(rc["elev"]), np.sin(rc["elev"])])
        z = (c - eye) / dist
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        cv = render(Camera(eye, np.stack([x, np.cross(z, x), z], 1).T, f, f, (W - 1) / 2, (H - 1) / 2, W, H),
                    xyz, sp["rgb"][rows], sp["scale"][rows], sp["op"][rows, 0])
        fp = (cv.gid >= 0) & (key[np.maximum(cv.gid, 0)] == k)
        near = float(((cv.depth < rc["near_frac"] * dist) & ~fp).mean())
        empty = float(np.isinf(cv.depth).mean())
        if near <= rc["max_near"] and empty <= rc["max_empty"]:
            cands.append((int(fp.sum()), eye))
    cands.sort(key=lambda q: -q[0])
    views = []
    for _, eye in cands:                                                   # next candidate when a view is unusable
        if sum("prec" in x for x in views) >= rc["views"]:
            break
        cam = look(client, eye, c, rc["width"], fov)
        view = render(cam, xyz, sp["rgb"][rows], sp["scale"][rows], sp["op"][rows, 0])
        foot = (view.gid >= 0) & (key[np.maximum(view.gid, 0)] == k)
        if foot.sum() < rc["min_visible_px"]:
            views.append({"skip": "hidden"})
            continue
        ys, xs = np.nonzero(foot)
        j = int(np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2))      # visible pixel nearest its centre
        img = frame(client, cam, view.rgb, rc["min_ncc"])
        if img is None:
            views.append({"skip": "frame"})
            continue
        e = rc["box_expand"]
        bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
        box = [max(0, xs.min() - e * bw), max(0, ys.min() - e * bh), min(cam.W - 1, xs.max() + e * bw),
               min(cam.H - 1, ys.max() + e * bh)]
        inp = proc(images=Image.fromarray(img), input_boxes=[[[float(x) for x in box]]],
                   input_points=[[[[float(xs[j]), float(ys[j])]]]], input_labels=[[[1]]], return_tensors="pt")
        with torch.no_grad():
            o = model(**inp, multimask_output=True)
        m = proc.post_process_masks(o.pred_masks, inp["original_sizes"])[0][0].numpy().astype(bool)
        s = o.iou_scores[0, 0].numpy()
        b = int(np.argmax(s))
        x0, y0, x1, y1 = (int(round(q)) for q in box)
        edge = np.concatenate([m[b][y0, x0:x1 + 1], m[b][y1, x0:x1 + 1], m[b][y0:y1 + 1, x0], m[b][y0:y1 + 1, x1]])
        inner = [x0 > 0, y0 > 0, x1 < cam.W - 1, y1 < cam.H - 1]           # box sides inside the frame
        extends = all(inner) and float(edge.mean()) > rc["max_edge_frac"]   # mask runs into the box: object continues
        inter = float((m[b] & foot).sum())
        prec, rec_ = inter / max(m[b].sum(), 1), inter / max(foot.sum(), 1)  # SAM inside ours / ours inside SAM
        views.append({"prec": round(prec, 2), "rec": round(rec_, 2), "score": round(float(s[b]), 3), "extends": extends, "px": int(foot.sum()),
                      "img": img, "mask": m[b], "foot": foot, "cam": cam})
    ok = [x for x in views if "prec" in x]
    for x in ok:
        x["good"] = (x["prec"] >= rc["min_prec"] and x["rec"] >= rc["min_rec"] and x["score"] >= rc["min_score"]
                     and not x["extends"])
    good = [x for x in ok if x["good"]]
    rec["sam_views"] = [(f"{x['prec']}/{x['rec']}{'+' if x['extends'] else ''}" if "prec" in x else x["skip"])
                        for x in views]
    rec["sam_ok"] = len(ok) >= rc["min_views"] and len(good) >= rc["agree_frac"] * len(ok)
    return {"views": ok}


def outline(img: np.ndarray, mask: np.ndarray, col: tuple[int, int, int]) -> np.ndarray:
    e = mask & ~(np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1))
    e = e | np.roll(e, 1, 0) | np.roll(e, 1, 1)
    out = img.copy()
    out[e] = col
    return out


def remap_include(cfg: Cfg, v: BimView, rc: Cfg) -> tuple[list[str], dict[str, dict]]:
    """Hand-confirmed ids from include_source's segmentation -> current ids, by point overlap (IoU)."""
    src = rc.get("include_source") or {}
    if not src:
        return list(rc["include"]), {i: {"id": i, "iou": None, "note": "unscoped list used as-is"} for i in rc["include"]}
    d = ROOT / src["seg_dir"]
    seg = np.load(d / "segment_points.npz")
    if label_revision(seg["labels"]) != src["segmentation"]:
        raise ArtifactMismatch(f"include_source {src['seg_dir']} is segmentation {label_revision(seg['labels'])}, "
                               f"not {src['segmentation']}: the hand-confirmed ids no longer apply")
    cur = np.load(out_path(cfg, "segment_points.npz"))
    if not np.array_equal(seg["xyz"], cur["xyz"]):
        raise ArtifactMismatch(f"{src['seg_dir']} has other voxel points than the current scene; ids can't be mapped")
    ids = {a["id"]: k for k, a in enumerate(json.loads((d / "assets.json").read_text(encoding="utf-8")))}
    out, mapping = [], {}
    for oid in rc["include"]:
        m = seg["labels"] == ids[oid]
        lab, cnt = np.unique(cur["labels"][m], return_counts=True)
        j = int(lab[np.argmax(cnt)])
        iou = float(cnt.max() / (m.sum() + (cur["labels"] == j).sum() - cnt.max())) if j >= 0 else 0.0
        new = v.objs[j]["id"] if j >= 0 else None
        mapping[oid] = {"id": new, "iou": round(iou, 4)}
        if new is None or iou < rc.get("include_min_iou", 0.9) or v.objs[j]["kind"] != "object":
            print(f"[removable] {src['seg_dir']} {oid}: best match {new} IoU {iou:.3f}: rejected")
            mapping[oid]["rejected"] = True
            continue
        print(f"[removable] {src['seg_dir']} {oid} -> {new} (IoU {iou:.3f}, {v.objs[j].get('label')})")
        out.append(new)
    return out, mapping


def write(cfg: Cfg, v: BimView, keep: list[dict], recs: list[dict], mapping: dict | None = None) -> None:
    clean = lambda r: {k: r[k] for k in r if k != "key"}                 # noqa: E731
    backup(cfg, ["removable.json", "removable_rows.npz", "removable_sheet.png"], "removable")
    out_path(cfg, "removable.json").write_text(json.dumps(
        {"version": 2, "revision": scene_revision(cfg), "removable": [r["id"] for r in keep],
         "parts": {r["id"]: [v.objs[m]["id"] for m in v.parts.get(r["key"], [])] for r in keep},
         "include_mapping": mapping or {}, "objects": [clean(r) for r in recs]}, indent=1), encoding="utf-8")
    np.savez_compressed(out_path(cfg, "removable_rows.npz"),                # splat rows per object, for viewer.app
                        **{r["id"]: v.splat["row_ids"][v._rows(r["key"])] for r in keep})


def run(cfg: Cfg, limit: int | None = None) -> None:
    rc = cfg["viewer"]["removable"]
    cfg["viewer"]["bim_port"] = rc["port"]
    v = BimView(cfg, apply_edits=False)
    t = time.time()
    recs = geometry(v, rc)
    if rc.get("include"):                                                   # hand-confirmed list: no SAM pass
        ids, mapping = remap_include(cfg, v, rc)
        keep = [r for i in ids for r in recs if r["id"] == i]
        for r in keep:
            if "reject" in r:
                print(f"[removable] note: {r['id']} fails the geometry screen ({r['reject']}, contact "
                      f"{r.get('contact')}, hole {r.get('hole')}); kept because it is hand-confirmed")
        write(cfg, v, keep, recs, mapping)
        v.server.stop()
        print(f"[removable] {len(keep)} confirmed objects: {[r['id'] for r in keep]} -> removable.json, removable_rows.npz")
        return
    from playwright.sync_api import sync_playwright
    cand = sorted([r for r in recs if "reject" not in r], key=lambda r: r["contact"] + r["hole"])[:limit]
    print(f"[geo] {len(recs)} objects, {len(cand)} pass the geometry screen ({time.time() - t:.0f}s); "
          f"rejected {dict((x, sum(r.get('reject') == x for r in recs)) for x in ('size/type', 'contact', 'hole'))}")
    sheet = []
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="chrome", args=["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"])
        except Exception:  # noqa: BLE001
            b = p.chromium.launch(args=["--enable-gpu", "--ignore-gpu-blocklist"])
        page = b.new_page(viewport={"width": 1280, "height": 800})
        page.goto(f"http://localhost:{rc['port']}")
        while not v.server.get_clients():
            time.sleep(0.5)
        client = next(iter(v.server.get_clients().values()))
        time.sleep(rc["load_wait_s"])
        for i, r in enumerate(cand):
            res = sam_check(v, client, r, rc, cfg)
            print(f"[sam] {i + 1}/{len(cand)} {r['id']} {r['label']} contact {r['contact']} hole {r['hole']} "
                  f"prec/rec {r['sam_views']} -> {'OK' if r['sam_ok'] else 'no'}", flush=True)
            if r["sam_ok"]:
                best = max(res["views"], key=lambda x: x["good"] * (x["prec"] + x["rec"]))
                h = v.splat_h[r["key"]]
                h.visible = False
                look(client, best["cam"].C, best["cam"].C + best["cam"].R[2], rc["width"], np.radians(rc["fov_deg"]))
                after = frame(client, best["cam"])
                h.visible = True
                sheet.append((r, outline(outline(best["img"], best["foot"], (0, 160, 255)), best["mask"], (255, 40, 40)),
                              after))
        b.close()
    v.server.stop()
    keep = [r for r in cand if r["sam_ok"]]
    keep.sort(key=lambda r: r["contact"] + r["hole"])
    write(cfg, v, keep, recs)
    if sheet:
        h, w = sheet[0][1].shape[:2]
        im = Image.new("RGB", (2 * w, h * len(sheet)), "white")
        d = ImageDraw.Draw(im)
        for i, (r, a, z) in enumerate(sheet):
            im.paste(Image.fromarray(a), (0, i * h))
            im.paste(Image.fromarray(z), (w, i * h))
            d.text((6, i * h + 4), f"{r['id']} {r['label']}  prec/rec {r['sam_views']}  (blue: object, red: SAM)",
                   fill=(255, 255, 0))
            d.text((w + 6, i * h + 4), "after removal", fill=(255, 255, 0))
        im.save(out_path(cfg, "removable_sheet.png"))
    print(f"[removable] {len(keep)} objects: {[r['id'] for r in keep]} -> removable.json, removable_rows.npz, removable_sheet.png")


if __name__ == "__main__":
    run(load_config(), int(sys.argv[1]) if len(sys.argv) > 1 else None)
