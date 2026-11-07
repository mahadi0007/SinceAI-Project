"""Contextual proposals -> SAM 2.1 masks -> visible 3D support, per review region and renderer.

Renderers (same calibrated cameras): "disc" (baseline: opaque discs, z-buffer winner) and "ewa"
(alpha-composited anisotropic Gaussians from quaternions/scales/opacity, explicit per-Gaussian weights).
Artifacts are bound by hashes and stale combinations are rejected:
  render_key = hash(renderer params, cameras, PLY checksum, load filters)    <rid>/<renderer>/render_meta.json
  proposal   : key = hash(model, prompt, exact image bytes); records cams_key/render_key it saw
  sam_key    = hash(render_key, proposal key, SAM model, box expansion)      <rid>/<renderer>/sam_meta.json
  assoc      : sam_key + segmentation revision (labels hash)      <renderer>-<proposal source>/<rid>_assoc.json
Proposal boxes are camera-space hypotheses: reusing them on another renderer is allowed only for
identical cameras (cams_key) and is recorded. SAM always gets the clean, immutable RGB render.
Run: python -m pipeline.propose RV-01 [render|ask|sam|assoc|all] [--renderer=ewa] [--proposal=legacy|ewa] [--metadata]
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from pipeline.common import ROOT, load_config
from pipeline.scene import Scene, load_scene, seg_revision
from pipeline.survey import short
from pipeline.views import Camera, View, draw_boxes, id_overlay, mask_outline, render, render_ewa

Cfg = dict[str, Any]
W, H = 768, 576
PDIR = ROOT / "out_review" / "proposals"
CLASSES = ["pump", "motor", "valve", "control valve", "sensor", "vessel", "unknown"]
REPAIRABLE = {"pump", "motor", "valve", "control valve", "sensor", "vessel"}

PROMPT = """You are inspecting renders of a 3D Gaussian-splat scan of an industrial equipment room. The renders are
sparse and dotted (scan noise), but shapes and colours are real.
Images: (1) view A, (2) view B of the same area from another angle, (3) view A again with the CURRENT automatic
segmentation: each fragment coloured and numbered. Fragments are often only pieces of one physical component,
and a single large fragment may wrongly mix several components with pipes and wall.

Task: find WHOLE physical components of these classes that are clearly visible: pump, motor, valve (body together
with its handwheel or actuator), control valve, sensor (gauge/transmitter), vessel. Do NOT just name existing
fragments: give the full visible extent of each component. A pump and its motor are separate components; give them
the same "assembly" id if they belong together. Do not list straight pipes, walls, floor, ladders or furniture.
Use class "unknown" when unsure. At most 6 components, clearest first. Empty list if none is visible.
{metadata}
Coordinates: integers normalised to 0-1000 per image (x to the right, y down). For view A: box_a [x0,y0,x1,y1]
tightly around the component and point_a [x,y] ON the component; same for view B (box_b, point_b) or null when
not visible there.

Return ONLY JSON:
{{"components": [{{"cid": "C1", "class": "pump", "box_a": [0,0,0,0], "point_a": [0,0], "box_b": null,
"point_b": null, "fragment_ids": [12, 40], "relation": "fragments 12 and 40 are one pump; fragment 9 also contains
wall pipes", "completeness": "complete|partial|occluded", "assembly": null, "confidence": 0.0,
"notes": "max 20 words"}}]}}"""

METADATA_HINT = """Plant inventory vocabulary (may or may not be visible here; do not invent components to match it):
process pumps with electric motors, control valves, pressure transmitters, vibration sensors."""




RENDERERS = ("disc", "ewa")


def regions() -> dict[str, dict]:
    reg = yaml.safe_load((ROOT / "review" / "regions.yaml").read_text(encoding="utf-8"))["regions"]
    return {r["id"]: r for r in reg}


def rdir(rid: str, renderer: str | None = None) -> Path:
    d = PDIR / rid / renderer if renderer else PDIR / rid
    d.mkdir(parents=True, exist_ok=True)
    return d


def png(im: Image.Image) -> bytes:
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def sha(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, (bytes, bytearray)) else json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:20]


def region_cameras(rid: str) -> list[Camera]:
    r = regions()[rid]
    return [Camera.look_at(v["eye"], v["target"], W, H, v["fov"], f"{rid}-v{k}") for k, v in enumerate(r["views"])]


def cams_key(cams: list[Camera]) -> str:
    return sha([c.to_json() for c in cams])


def renderer_params(cfg: Cfg, renderer: str) -> dict:
    return {"renderer": renderer, **(cfg["propose"].get("render", {}).get(renderer, {}))}


# ---------- render ----------

def render_view(sc: Scene, cam: Camera, renderer: str, params: dict) -> dict:
    """Unified buffers: clean RGB, depth (z-buffer / median), alpha, sparse (pixel, Gaussian, weight)."""
    if renderer == "ewa":
        kw = {k: v for k, v in params.items() if k != "renderer"}
        v = render_ewa(cam, sc.xyz, sc.rgb, sc.scales, sc.quats, sc.opacity, **kw)
        return {"rgb": v.rgb, "depth": v.depth, "depth_exp": v.depth_exp, "alpha": v.alpha,
                "frag_pix": v.frag_pix, "frag_gid": v.frag_gid, "frag_w": v.frag_w, "top_id": v.top_id()}
    v = render(cam, sc.xyz, sc.rgb, sc.scales, sc.opacity)
    gid = v.gid.ravel()
    pix = np.where(gid >= 0)[0]
    return {"rgb": v.rgb, "depth": v.depth, "depth_exp": v.depth, "alpha": (v.gid >= 0).astype(np.float32),
            "frag_pix": pix.astype(np.int32), "frag_gid": gid[pix], "frag_w": np.ones(len(pix), np.float32),
            "top_id": v.gid}


def depth_image(dep: np.ndarray) -> Image.Image:
    import matplotlib.cm as cm
    f = np.isfinite(dep)
    out = np.full(dep.shape + (3,), 255, np.uint8)
    if f.any():
        lo, hi = np.percentile(dep[f], [2, 98])
        out[f] = (cm.viridis(np.clip((dep[f] - lo) / max(hi - lo, 1e-6), 0, 1))[:, :3] * 255).astype(np.uint8)
    return Image.fromarray(out)


def render_region(sc: Scene, rid: str, renderer: str, cfg: Cfg) -> dict:
    """Writes <rid>/<renderer>/: view{k}.png (clean RGB, the only image SAM/VLM ever see),
    view{k}_ids.png (fragment overlay), view{k}_depth.png / _alpha.png (diagnostics), view{k}_buffers.npz,
    render_meta.json with render_key = hash(renderer params, cameras, PLY checksum)."""
    import time
    d = rdir(rid, renderer)
    cams = region_cameras(rid)
    params = renderer_params(cfg, renderer)
    man = json.loads((sc.seg_dir / "segment_manifest.json").read_text(encoding="utf-8"))
    rkey = sha(params, cams_key(cams), man["ply_sha256"], cfg["load"])
    gl = sc.g_label
    names = {i: short(a["id"]) for i, a in enumerate(sc.assets)}
    objs = {i for i, a in enumerate(sc.assets) if a["kind"] == "object"}
    t0 = time.time()
    for k, cam in enumerate(cams):
        b = render_view(sc, cam, renderer, params)
        Image.fromarray(b["rgb"]).save(d / f"view{k}.png")
        tid = b["top_id"]
        pix = np.where(tid >= 0, gl[np.maximum(tid, 0)], -1)
        id_overlay(View(cam, b["rgb"], b["depth"], tid), pix, names, focus=objs, min_px=80).save(d / f"view{k}_ids.png")
        depth_image(b["depth"]).save(d / f"view{k}_depth.png")
        Image.fromarray((np.clip(b["alpha"], 0, 1) * 255).astype(np.uint8)).save(d / f"view{k}_alpha.png")
        np.savez_compressed(d / f"view{k}_buffers.npz", **{k2: b[k2] for k2 in
                            ("depth", "depth_exp", "alpha", "frag_pix", "frag_gid", "frag_w", "top_id")})
    meta = {"rid": rid, "renderer": renderer, "params": params, "cams_key": cams_key(cams), "render_key": rkey,
            "overlay_seg_rev": seg_revision(sc.seg_dir), "cameras": [c.to_json() for c in cams],
            "seconds": round(time.time() - t0, 1)}
    (d / "render_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[render] {rid}/{renderer}: {len(cams)} views in {meta['seconds']}s key {rkey}")
    return meta


def render_meta(rid: str, renderer: str) -> dict:
    return json.loads((rdir(rid, renderer) / "render_meta.json").read_text(encoding="utf-8"))


def load_views(rid: str, renderer: str) -> tuple[list[Camera], list[np.ndarray], list[dict]]:
    d = rdir(rid, renderer)
    meta = render_meta(rid, renderer)
    cams = [Camera.from_json(c) for c in meta["cameras"]]
    rgbs = [np.asarray(Image.open(d / f"view{k}.png").convert("RGB")) for k in range(len(cams))]
    bufs = [dict(np.load(d / f"view{k}_buffers.npz")) for k in range(len(cams))]
    return cams, rgbs, bufs


# ---------- ask (paid) ----------
def norm_box(b: Any) -> list[int] | None:
    if not (isinstance(b, list) and len(b) == 4):
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in b)
    except (TypeError, ValueError):
        return None
    if not all(0 <= v <= 1000 for v in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
        return None
    return [int(x0), int(y0), int(x1), int(y1)]


def norm_pt(p: Any) -> list[int] | None:
    if not (isinstance(p, list) and len(p) == 2):
        return None
    try:
        x, y = float(p[0]), float(p[1])
    except (TypeError, ValueError):
        return None
    return [int(x), int(y)] if 0 <= x <= 1000 and 0 <= y <= 1000 else None


def validate(res: Any) -> tuple[list[dict], list[str]]:
    comps = res.get("components", []) if isinstance(res, dict) else []
    ok, bad = [], []
    for i, c in enumerate(comps[:6]):
        if not isinstance(c, dict):
            bad.append(f"#{i}: not an object")
            continue
        cls = str(c.get("class", "")).strip().lower()
        try:
            conf = float(c.get("confidence"))
        except (TypeError, ValueError):
            conf = float("nan")
        if cls not in CLASSES:
            bad.append(f"#{i}: class {cls!r}")
            continue
        if not (conf == conf and 0 <= conf <= 1):
            bad.append(f"#{i}: confidence")
            continue
        ba, bb = norm_box(c.get("box_a")), norm_box(c.get("box_b"))
        if ba is None and bb is None:
            bad.append(f"#{i}: no valid box")
            continue
        frs = [int(f) for f in c.get("fragment_ids") or [] if isinstance(f, (int, float)) or str(f).isdigit()]
        ok.append({"cid": str(c.get("cid") or f"C{i + 1}")[:8], "cls": cls, "confidence": round(conf, 3),
                   "box": [ba, bb], "point": [norm_pt(c.get("point_a")) if ba else None,
                                              norm_pt(c.get("point_b")) if bb else None],
                   "fragment_ids": frs[:30], "relation": str(c.get("relation", ""))[:200],
                   "completeness": str(c.get("completeness", ""))[:20],
                   "assembly": (str(c["assembly"])[:8] if c.get("assembly") else None),
                   "notes": str(c.get("notes", ""))[:160]})
    return ok, bad


def ask(rid: str, cfg: Cfg, renderer: str, metadata: bool = False) -> dict:
    """One paid request. Images = clean RGB views of `renderer` + its fragment overlay; cache key covers
    model, prompt and the exact image bytes, so new renders always get new keys."""
    from copilot.llm import generate
    d = rdir(rid, renderer)
    meta = render_meta(rid, renderer)
    imgs = [(d / "view0.png").read_bytes()] + ([(d / "view1.png").read_bytes()] if (d / "view1.png").exists() else []) \
        + [(d / "view0_ids.png").read_bytes()]
    prompt = PROMPT.format(metadata=METADATA_HINT if metadata else "")
    key = proposal_key(cfg["copilot"]["model"], prompt, imgs)
    tag = "meta" if metadata else "visual"
    cache = rdir(rid) / f"proposal_{tag}_{renderer}_{key}.json"
    if cache.exists():
        print(f"[ask] {rid}: cached {cache.name}")
        return json.loads(cache.read_text(encoding="utf-8"))
    parts: list[Any] = [prompt, "Image 1 (view A):", imgs[0]]
    if len(imgs) == 3:
        parts += ["Image 2 (view B):", imgs[1]]
    parts += [f"Image {len(imgs)} (view A, fragment IDs):", imgs[-1]]
    text, model = generate(parts, cfg["copilot"], json_mode=True, temperature=0.1,
                           purpose=f"proposal_{rid}_{tag}_{renderer}", max_tokens=1500)
    from copilot.agent import parse_json
    try:
        raw = parse_json(text)
    except Exception:  # noqa: BLE001
        raw = {}
    comps, bad = validate(raw)
    out = {"region": rid, "model": model, "mode": tag, "key": key, "renderer": renderer,
           "cams_key": meta["cams_key"], "render_key": meta["render_key"], "overlay_seg_rev": meta["overlay_seg_rev"],
           "raw_text": text, "components": comps, "rejected": bad, "n_images": len(imgs)}
    cache.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[ask] {rid}/{renderer}: {len(comps)} valid components, rejected {bad}")
    return out


def proposal_key(model: str, prompt: str, imgs: list[bytes]) -> str:
    h = hashlib.sha256((model + prompt).encode())
    for i in imgs:
        h.update(hashlib.sha256(i).digest())
    return h.hexdigest()[:20]


def bind_legacy(rid: str, prop: dict, cfg: Cfg) -> dict | None:
    """Proposals made before render binding (top-level disc renders): accept only if the stored key is
    reproduced from the top-level images it was asked with, then bind it to those images' cameras."""
    d = rdir(rid)
    try:
        imgs = [(d / "view0.png").read_bytes()] + ([(d / "view1.png").read_bytes()] if (d / "view1.png").exists() else []) \
            + [(d / "view0_ids.png").read_bytes()]
        cams = [Camera.from_json(c) for c in json.loads((d / "cameras.json").read_text(encoding="utf-8"))["cameras"]]
    except FileNotFoundError:
        return None
    prompt = PROMPT.format(metadata=METADATA_HINT if prop.get("mode") == "meta" else "")
    if proposal_key(prop["model"], prompt, imgs) != prop["key"]:
        return None
    return {**prop, "renderer": "disc-legacy", "cams_key": cams_key(cams), "render_key": None}


def latest_proposal(rid: str, cfg: Cfg, source: str, tag: str = "visual") -> dict:
    """source = 'legacy' (pilot proposals on disc renders) or a renderer name ('ewa', 'disc')."""
    pat = f"proposal_{tag}_[0-9a-f]*.json" if source == "legacy" else f"proposal_{tag}_{source}_*.json"
    fs = sorted(rdir(rid).glob(pat), key=lambda p: p.stat().st_mtime)
    if not fs:
        raise FileNotFoundError(f"no {source} {tag} proposal for {rid}")
    prop = json.loads(fs[-1].read_text(encoding="utf-8"))
    if source == "legacy":
        prop = bind_legacy(rid, prop, cfg)
        if prop is None:
            raise ValueError(f"{rid}: legacy proposal no longer matches its images (stale)")
    prop["proposal_file"] = str(fs[-1].relative_to(ROOT))
    return prop


# ---------- SAM (clean image, embeddings reused) ----------

_sam: dict = {}


def sam_model(name: str) -> tuple[Any, Any]:
    if name not in _sam:
        from transformers import Sam2Model, Sam2Processor
        _sam[name] = (Sam2Processor.from_pretrained(name), Sam2Model.from_pretrained(name).eval())
    return _sam[name]


def to_px(v: list[int], w: int, h: int) -> list[float]:
    return [v[0] / 1000 * (w - 1), v[1] / 1000 * (h - 1)] + ([v[2] / 1000 * (w - 1), v[3] / 1000 * (h - 1)]
                                                             if len(v) == 4 else [])


def sam_prompts(prop: dict, v: int, expand: float) -> list[dict]:
    """Per candidate: expanded box, own positive point, other candidates' points inside the box as negatives."""
    out = []
    for ci, c in enumerate(prop["components"]):
        if v >= len(c["box"]) or c["box"][v] is None:
            continue
        raw = to_px(c["box"][v], W, H)
        bw, bh = raw[2] - raw[0], raw[3] - raw[1]
        box = [max(0, raw[0] - expand * bw), max(0, raw[1] - expand * bh),
               min(W - 1, raw[2] + expand * bw), min(H - 1, raw[3] + expand * bh)]
        pts, lbl = [], []
        if c["point"][v]:
            pts.append(to_px(c["point"][v], W, H))
            lbl.append(1)
        for cj, o in enumerate(prop["components"]):
            if cj != ci and v < len(o["point"]) and o["point"][v]:
                p = to_px(o["point"][v], W, H)
                if box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]:
                    pts.append(p)
                    lbl.append(0)
        out.append({"ci": ci, "cid": c["cid"], "box": box, "raw_box": raw, "pts": pts, "lbl": lbl})
    return out


_emb_cache: dict[tuple[str, str], tuple[Any, Any]] = {}
EMB_CACHE_MAX = 8


def image_sha(image: np.ndarray) -> str:
    a = np.ascontiguousarray(image)
    return hashlib.sha256(a.tobytes() + str(a.shape).encode()).hexdigest()[:20]


def sam_embedding(image: np.ndarray, name: str, cache: bool = True) -> tuple[Any, Any, Image.Image]:
    """(embedding, processor output with original_sizes, PIL copy) for ONE clean image. Cached by
    (model, image content hash), so repeated prompts on the same render never re-encode it."""
    import torch
    proc, model = sam_model(name)
    img = Image.fromarray(np.array(image, copy=True))
    key = (name, image_sha(image))
    if cache and key in _emb_cache:
        emb, base = _emb_cache[key]
        return emb, base, img
    base = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        emb = model.get_image_embeddings(base["pixel_values"])
    if cache:
        if len(_emb_cache) >= EMB_CACHE_MAX:
            _emb_cache.pop(next(iter(_emb_cache)))
        _emb_cache[key] = (emb, base)
    return emb, base, img


def sam_predict(emb: Any, base: Any, img: Image.Image, prompt: dict, name: str) -> tuple[np.ndarray, float]:
    """One prompt = optional box [x0,y0,x1,y1] + points/labels (1 include, 0 exclude), all in ORIGINAL image
    pixel coordinates (the processor rescales them). Returns the best-IoU mask of the multimask output."""
    import torch
    proc, model = sam_model(name)
    kw: dict[str, Any] = {"images": img, "return_tensors": "pt"}
    if prompt.get("box") is not None:
        kw["input_boxes"] = [[list(map(float, prompt["box"]))]]
    if prompt.get("pts"):
        kw["input_points"] = [[[list(map(float, p)) for p in prompt["pts"]]]]
        kw["input_labels"] = [[list(map(int, prompt["lbl"]))]]
    if "input_boxes" not in kw and "input_points" not in kw:
        raise ValueError("SAM prompt needs a box or at least one point")
    inp = proc(**kw)
    inp.pop("pixel_values", None)
    with torch.no_grad():
        out = model(image_embeddings=emb, multimask_output=True,
                    **{k: v for k, v in inp.items() if k in ("input_points", "input_labels", "input_boxes")})
    m = proc.post_process_masks(out.pred_masks, base["original_sizes"])[0][0].numpy()
    s = out.iou_scores[0, 0].numpy()
    k = int(s.argmax())
    return m[k].astype(bool), float(s[k])


def run_sam(image: np.ndarray, prompts: list[dict], name: str, cache: bool = False) -> list[tuple[np.ndarray, float]]:
    """SAM on ONE immutable clean image: the embedding is computed once and reused for every prompt.
    Returns (mask, iou score) per prompt."""
    emb, base, img = sam_embedding(image, name, cache=cache)
    return [sam_predict(emb, base, img, p, name) for p in prompts]


def sam_masks(rid: str, renderer: str, cfg: Cfg, prop: dict) -> dict:
    meta = render_meta(rid, renderer)
    if prop["cams_key"] != meta["cams_key"]:
        raise ValueError(f"{rid}: proposal cameras {prop['cams_key']} != render cameras {meta['cams_key']}")
    cams, rgbs, _ = load_views(rid, renderer)
    name, e = cfg["propose"]["sam_model"], cfg["propose"]["box_expand"]
    d = rdir(rid, renderer)
    masks: dict[str, np.ndarray] = {}
    scores: dict[str, list] = {}
    for v, rgb in enumerate(rgbs):
        prompts = sam_prompts(prop, v, e)
        clean = rgb.copy()
        results = run_sam(clean, prompts, name)
        assert np.array_equal(clean, rgb), "SAM input image was modified"
        overlay = Image.fromarray(rgb)                      # drawing happens on a separate copy only
        boxes_draw = []
        for p, (m, s) in zip(prompts, results):
            masks[f"c{p['ci']}_v{v}"] = m
            scores.setdefault(p["cid"], []).append({"view": v, "score": round(s, 3), "mask_px": int(m.sum())})
            col = tuple(int(x) for x in np.random.default_rng(p["ci"]).integers(40, 255, 3))
            overlay = mask_outline(overlay, m, col)
            c = prop["components"][p["ci"]]
            boxes_draw.append((p["raw_box"], f"{c['cid']} {c['cls']} {c['confidence']:.2f} sam {s:.2f}", col))
        draw_boxes(overlay, boxes_draw, 2).save(d / f"sam_view{v}.png")
    skey = sha(meta["render_key"], prop["key"], name, e)
    np.savez_compressed(d / "sam_masks.npz", **masks)
    (d / "sam_meta.json").write_text(json.dumps({"model": name, "box_expand": e, "sam_key": skey,
                                                 "render_key": meta["render_key"], "proposal_key": prop["key"],
                                                 "proposal_file": prop.get("proposal_file"), "scores": scores},
                                                indent=1), encoding="utf-8")
    print(f"[sam] {rid}/{renderer}: {len(masks)} masks, key {skey}")
    return scores


# ---------- association (weighted visible support) ----------

def view_support(xyz: np.ndarray, g_vox: np.ndarray, nvox: int, cam: Camera, buf: dict, mask: np.ndarray,
                 pc: Cfg) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Visible 2D->3D support of ONE mask in ONE calibrated view (shared by automatic association and manual
    correction). pos/neg per voxel = summed contribution weights (EWA: alpha * transmittance, so occluded
    Gaussians add ~0) of its Gaussians inside / outside the mask. Mask pixels deeper than
    front + max(depth_gate_min, depth_gate_size * size) are treated as background seen through gaps.
    Returns None for an empty mask."""
    m = np.asarray(mask, bool)
    if m.shape != (cam.H, cam.W):
        raise ValueError(f"mask shape {m.shape} != view size {(cam.H, cam.W)}")
    dm = buf["depth"][m]
    dm = dm[np.isfinite(dm)]
    if not len(dm):
        return None
    mf = m.ravel()
    fp, fg, fw = buf["frag_pix"], buf["frag_gid"], buf["frag_w"]
    fz = (xyz[fg] - cam.C) @ cam.R[2]
    inm = mf[fp]
    front = float(np.percentile(dm, 10))
    ys, xs = np.nonzero(m)
    size = max(np.ptp(xs), np.ptp(ys)) * front / cam.fx
    gate = front + max(pc["depth_gate_min"], pc["depth_gate_size"] * size)
    vox = g_vox[fg]
    ok = vox >= 0
    pin = ok & inm & (fz <= gate)
    nout = ok & ~inm
    pos = np.bincount(vox[pin], weights=fw[pin], minlength=nvox).astype(np.float32)
    neg = np.bincount(vox[nout], weights=fw[nout], minlength=nvox).astype(np.float32)
    return pos, neg, {"mask_px": int(mf.sum()), "w_in": round(float(fw[pin].sum()), 1),
                      "w_gated": round(float(fw[ok & inm & (fz > gate)].sum()), 1), "gate": round(gate, 3)}


def view_visibility(g_vox: np.ndarray, nvox: int, buf: dict) -> np.ndarray:
    """Summed visible contribution weight per voxel in one view (0 = occluded or out of frame)."""
    vox = g_vox[buf["frag_gid"]]
    ok = vox >= 0
    return np.bincount(vox[ok], weights=buf["frag_w"][ok], minlength=nvox).astype(np.float32)


def associate(sc: Scene, rid: str, renderer: str, cfg: Cfg, prop: dict) -> dict:
    """pos/neg per voxel = summed contribution weights of its Gaussians inside / outside each mask.
    EWA weights are alpha*transmittance, so occluded Gaussians contribute ~0; disc weights are 1 for the
    z-buffer winner. A depth gate drops background seen through gaps."""
    pc = cfg["propose"]
    meta = render_meta(rid, renderer)
    d = rdir(rid, renderer)
    sm = json.loads((d / "sam_meta.json").read_text(encoding="utf-8"))
    if sm["render_key"] != meta["render_key"] or sm["proposal_key"] != prop["key"]:
        raise ValueError(f"{rid}/{renderer}: SAM masks are stale for this render/proposal")
    cams, _, bufs = load_views(rid, renderer)
    M = np.load(d / "sam_masks.npz")
    nvox = len(sc.vox_xyz)
    arrays: dict[str, np.ndarray] = {}
    cands = []
    for ci, c in enumerate(prop["components"]):
        pos = np.zeros(nvox, np.float32)
        neg = np.zeros(nvox, np.float32)
        used = []
        for v, (cam, b) in enumerate(zip(cams, bufs)):
            k = f"c{ci}_v{v}"
            s = next((x["score"] for x in sm["scores"].get(c["cid"], []) if x["view"] == v), 0)
            if k not in M.files or s < pc["min_sam_score"]:
                continue
            res = view_support(sc.xyz, sc.g_vox, nvox, cam, b, M[k], pc)
            if res is None:
                continue
            p_v, n_v, st = res
            pos += p_v
            neg += n_v
            used.append({"view": v, "sam": s, **st})
        usable = bool(used) and c["cls"] in REPAIRABLE and c["confidence"] >= pc["min_confidence"]
        arrays[f"pos_{ci}"], arrays[f"neg_{ci}"] = pos, neg
        cands.append({**c, "usable": usable, "views_used": used, "sam_scores": sm["scores"].get(c["cid"])})
    seg_rev = seg_revision(sc.seg_dir)
    akey = sha(sm["sam_key"], seg_rev)
    src = "legacy" if prop.get("renderer") == "disc-legacy" else prop.get("renderer")
    out_dir = PDIR / f"{renderer}-{src}"                     # association set = SAM/visibility renderer x proposal source
    out_dir.mkdir(exist_ok=True)
    np.savez_compressed(out_dir / f"{rid}_assoc.npz", **arrays)
    info = {"region": rid, "renderer": renderer, "model": prop["model"], "proposal_file": prop.get("proposal_file"),
            "proposal_key": prop["key"], "proposal_renderer": prop.get("renderer"), "cams_key": meta["cams_key"],
            "render_key": meta["render_key"], "sam_key": sm["sam_key"], "sam_model": sm["model"],
            "seg_dir": sc.seg_dir.name, "seg_rev": seg_rev, "assoc_key": akey, "n_views": len(cams),
            "candidates": cands}
    (out_dir / f"{rid}_assoc.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"[assoc] {rid}/{renderer}: " + ", ".join(f"{c['cid']} {c['cls']} usable={c['usable']}" for c in cands))
    return info


def check_fresh(info: dict, seg_dir: Path) -> str | None:
    """Reason the association is stale, or None. Every link must match what is on disk now."""
    rid, r = info["region"], info["renderer"]
    try:
        meta = render_meta(rid, r)
        sm = json.loads((rdir(rid, r) / "sam_meta.json").read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        return f"missing {e.filename}"
    if meta["render_key"] != info["render_key"]:
        return "render changed since association"
    if meta["cams_key"] != info["cams_key"] or meta["cams_key"] != cams_key(region_cameras(rid)):
        return "cameras changed (regions.yaml) since render"
    if sm["sam_key"] != info["sam_key"] or sm["proposal_key"] != info["proposal_key"]:
        return "SAM masks changed since association"
    if seg_revision(seg_dir) != info["seg_rev"]:
        return "segmentation revision changed since association"
    return None


if __name__ == "__main__":
    rid = sys.argv[1]
    step = sys.argv[2] if len(sys.argv) > 2 else "all"
    cfg = load_config()
    pc = cfg["propose"]
    renderer = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--renderer=")), pc["renderer"])
    source = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--proposal=")), pc["proposal_source"])
    sc = load_scene(cfg, pc["seg_dir"])
    if step in ("render", "all"):
        render_region(sc, rid, renderer, cfg)
    if step == "ask":
        ask(rid, cfg, renderer, metadata="--metadata" in sys.argv)
    if step in ("sam", "all", "assoc"):
        prop = latest_proposal(rid, cfg, source)
        if step in ("sam", "all"):
            sam_masks(rid, renderer, cfg, prop)
        associate(sc, rid, renderer, cfg, prop)
