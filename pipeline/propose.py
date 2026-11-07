"""Contextual proposals -> SAM 2.1 masks -> visible 3D support, per review region.

  render  : calibrated views of the region (RGB, depth, visible-Gaussian ID) + a fragment-ID overlay
            of the current (geometry-repaired) segmentation; cameras saved as JSON.
  ask     : ONE vision-LLM request per region (Featherless Qwen3.5): view A, view B, overlay of A.
            Asks for WHOLE components (pump / motor / valve...) with per-view boxes + points;
            output validated; cached by hash(model, prompt, image bytes). Coordinates are hypotheses.
  sam     : SAM 2.1 tiny (transformers, CPU) prompted with each box + positive point, and negative
            points on the other candidates; best-IoU mask kept; box-vs-mask comparison image saved.
  assoc   : per voxel, visible pixels inside (pos) / outside (neg) each candidate's masks, using the
            ID buffer only (occluded Gaussians get nothing) and a depth gate that ignores background
            seen through gaps in the sparse splat.
Run: python -m pipeline.propose RV-01 [render|ask|sam|assoc|all] [--metadata]
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
from PIL import Image, ImageDraw

from pipeline.common import ROOT, load_config
from pipeline.scene import Scene, load_scene
from pipeline.survey import short
from pipeline.views import Camera, View, draw_boxes, id_overlay, mask_outline, render

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


def regions() -> dict[str, dict]:
    reg = yaml.safe_load((ROOT / "review" / "regions.yaml").read_text(encoding="utf-8"))["regions"]
    return {r["id"]: r for r in reg}


def rdir(rid: str) -> Path:
    d = PDIR / rid
    d.mkdir(parents=True, exist_ok=True)
    return d


def png(im: Image.Image) -> bytes:
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


# ---------- render ----------

def render_region(sc: Scene, rid: str) -> list[View]:
    r = regions()[rid]
    d = rdir(rid)
    gl = sc.g_label
    names = {i: short(a["id"]) for i, a in enumerate(sc.assets)}
    objs = {i for i, a in enumerate(sc.assets) if a["kind"] == "object"}
    views, cams = [], []
    for k, v in enumerate(r["views"]):
        cam = Camera.look_at(v["eye"], v["target"], W, H, v["fov"], f"{rid}-v{k}")
        vw = render(cam, sc.xyz, sc.rgb, sc.scales, sc.opacity)
        Image.fromarray(vw.rgb).save(d / f"view{k}.png")
        pix = np.where(vw.gid >= 0, gl[np.maximum(vw.gid, 0)], -1)
        id_overlay(vw, pix, names, focus=objs, min_px=80).save(d / f"view{k}_ids.png")
        np.savez_compressed(d / f"view{k}_buffers.npz", gid=vw.gid, depth=vw.depth)
        cams.append(cam.to_json())
        views.append(vw)
    (d / "cameras.json").write_text(json.dumps({"seg_dir": str(sc.seg_dir.name), "cameras": cams}, indent=1),
                                    encoding="utf-8")
    print(f"[render] {rid}: {len(views)} views -> {d}")
    return views


def load_views(rid: str) -> tuple[list[Camera], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    d = rdir(rid)
    cams = [Camera.from_json(c) for c in json.loads((d / "cameras.json").read_text(encoding="utf-8"))["cameras"]]
    rgbs, gids, depths = [], [], []
    for k in range(len(cams)):
        rgbs.append(np.asarray(Image.open(d / f"view{k}.png").convert("RGB")))
        b = np.load(d / f"view{k}_buffers.npz")
        gids.append(b["gid"])
        depths.append(b["depth"])
    return cams, rgbs, gids, depths


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


def ask(rid: str, cfg: Cfg, metadata: bool = False) -> dict:
    from copilot.llm import generate
    d = rdir(rid)
    imgs = [(d / "view0.png").read_bytes(), (d / "view1.png").read_bytes() if (d / "view1.png").exists() else None,
            (d / "view0_ids.png").read_bytes()]
    imgs = [i for i in imgs if i is not None]
    prompt = PROMPT.format(metadata=METADATA_HINT if metadata else "")
    h = hashlib.sha256((cfg["copilot"]["model"] + prompt).encode())
    for i in imgs:
        h.update(hashlib.sha256(i).digest())
    key = h.hexdigest()[:20]
    tag = "meta" if metadata else "visual"
    cache = d / f"proposal_{tag}_{key}.json"
    if cache.exists():
        print(f"[ask] {rid}: cached {cache.name}")
        return json.loads(cache.read_text(encoding="utf-8"))
    parts: list[Any] = [prompt, "Image 1 (view A):", imgs[0]]
    if len(imgs) == 3:
        parts += ["Image 2 (view B):", imgs[1]]
    parts += [f"Image {len(imgs)} (view A, fragment IDs):", imgs[-1]]
    text, model = generate(parts, cfg["copilot"], json_mode=True, temperature=0.1, purpose=f"proposal_{rid}_{tag}",
                           max_tokens=1500)
    from copilot.agent import parse_json
    try:
        raw = parse_json(text)
    except Exception:  # noqa: BLE001
        raw = {}
    comps, bad = validate(raw)
    out = {"region": rid, "model": model, "mode": tag, "key": key, "raw_text": text, "components": comps,
           "rejected": bad, "n_images": len(imgs)}
    cache.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[ask] {rid}: {len(comps)} valid components, rejected {bad}")
    return out


def latest_proposal(rid: str, tag: str = "visual") -> dict:
    fs = sorted(rdir(rid).glob(f"proposal_{tag}_*.json"), key=lambda p: p.stat().st_mtime)
    if not fs:
        raise FileNotFoundError(f"no {tag} proposal for {rid}")
    out = json.loads(fs[-1].read_text(encoding="utf-8"))
    out["proposal_file"] = str(fs[-1].relative_to(ROOT))
    return out


# ---------- SAM ----------

_sam: dict = {}


def sam_model(name: str) -> tuple[Any, Any]:
    if name not in _sam:
        from transformers import Sam2Model, Sam2Processor
        _sam[name] = (Sam2Processor.from_pretrained(name), Sam2Model.from_pretrained(name).eval())
    return _sam[name]


def to_px(v: list[int], w: int, h: int) -> list[float]:
    return [v[0] / 1000 * (w - 1), v[1] / 1000 * (h - 1)] + ([v[2] / 1000 * (w - 1), v[3] / 1000 * (h - 1)]
                                                             if len(v) == 4 else [])


def sam_masks(rid: str, cfg: Cfg) -> dict:
    import torch
    prop = latest_proposal(rid)
    cams, rgbs, _, _ = load_views(rid)
    name = cfg["propose"]["sam_model"]
    proc, model = sam_model(name)
    d = rdir(rid)
    masks: dict[str, np.ndarray] = {}
    scores: dict[str, list] = {}
    for v, (cam, rgb) in enumerate(zip(cams, rgbs)):
        vis = Image.fromarray(rgb)
        boxes_draw = []
        for ci, c in enumerate(prop["components"]):
            if v >= len(c["box"]) or c["box"][v] is None:
                continue
            box = to_px(c["box"][v], W, H)
            raw_box = list(box)
            e = cfg["propose"]["box_expand"]                     # VLM boxes are coarse; SAM sets the boundary
            bw, bh = box[2] - box[0], box[3] - box[1]
            box = [max(0, box[0] - e * bw), max(0, box[1] - e * bh), min(W - 1, box[2] + e * bw), min(H - 1, box[3] + e * bh)]
            pts, lbl = [], []
            if c["point"][v]:
                pts.append(to_px(c["point"][v], W, H))
                lbl.append(1)
            for cj, o in enumerate(prop["components"]):          # negatives: other candidates' points
                if cj != ci and v < len(o["point"]) and o["point"][v]:
                    p = to_px(o["point"][v], W, H)
                    if not (box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]):
                        continue                                  # only those inside this box matter
                    pts.append(p)
                    lbl.append(0)
            kw: dict[str, Any] = {"images": vis, "input_boxes": [[box]], "return_tensors": "pt"}
            if pts:
                kw["input_points"] = [[pts]]
                kw["input_labels"] = [[lbl]]
            inp = proc(**kw)
            with torch.no_grad():
                out = model(**inp, multimask_output=True)
            m = proc.post_process_masks(out.pred_masks, inp["original_sizes"])[0][0].numpy()   # (3, H, W)
            s = out.iou_scores[0, 0].numpy()
            k = int(s.argmax())
            masks[f"c{ci}_v{v}"] = m[k].astype(bool)
            scores.setdefault(c["cid"], []).append({"view": v, "score": round(float(s[k]), 3),
                                                     "mask_px": int(m[k].sum()),
                                                     "box_px": int((box[2] - box[0]) * (box[3] - box[1]))})
            col = tuple(int(x) for x in np.random.default_rng(ci).integers(40, 255, 3))
            vis = mask_outline(vis, m[k].astype(bool), col)
            boxes_draw.append((raw_box, f"{c['cid']} {c['cls']} {c['confidence']:.2f} sam {s[k]:.2f}", col))
        draw_boxes(vis, boxes_draw, 2).save(d / f"sam_view{v}.png")
    np.savez_compressed(d / "sam_masks.npz", **masks)
    (d / "sam_scores.json").write_text(json.dumps({"model": name, "scores": scores}, indent=1), encoding="utf-8")
    print(f"[sam] {rid}: {len(masks)} masks; scores {json.dumps(scores)}")
    return scores


# ---------- association (visible support) ----------

def associate(sc: Scene, rid: str, cfg: Cfg) -> dict:
    pc = cfg["propose"]
    prop = latest_proposal(rid)
    cams, _, gids, depths = load_views(rid)
    d = rdir(rid)
    M = np.load(d / "sam_masks.npz")
    sc_json = json.loads((d / "sam_scores.json").read_text(encoding="utf-8"))
    nvox = len(sc.vox_xyz)
    arrays: dict[str, np.ndarray] = {}
    cands = []
    for ci, c in enumerate(prop["components"]):
        pos = np.zeros(nvox, np.float32)
        neg = np.zeros(nvox, np.float32)
        used_views = []
        for v, (cam, gid, dep) in enumerate(zip(cams, gids, depths)):
            k = f"c{ci}_v{v}"
            if k not in M.files:
                continue
            s = next((x["score"] for x in sc_json["scores"].get(c["cid"], []) if x["view"] == v), 0)
            if s < pc["min_sam_score"]:
                continue
            m = M[k]
            vis = gid >= 0
            # depth gate: drop background seen through gaps of the sparse splat
            dm = dep[m & vis]
            if len(dm) == 0:
                continue
            front = float(np.percentile(dm, 10))
            ys, xs = np.nonzero(m)
            size = max(np.ptp(xs), np.ptp(ys)) * front / cam.fx
            gate = front + max(pc["depth_gate_min"], pc["depth_gate_size"] * size)
            inside = m & vis & (dep <= gate)
            outside = ~m & vis
            gv_in = sc.g_vox[gid[inside]]
            gv_out = sc.g_vox[gid[outside]]
            np.add.at(pos, gv_in[gv_in >= 0], 1)
            np.add.at(neg, gv_out[gv_out >= 0], 1)
            used_views.append({"view": v, "sam": s, "mask_px": int(m.sum()), "kept_px": int(inside.sum()),
                               "gated_px": int((m & vis & (dep > gate)).sum()), "depth_front": round(front, 3),
                               "gate": round(gate, 3)})
        usable = bool(used_views) and c["cls"] in REPAIRABLE and c["confidence"] >= pc["min_confidence"]
        arrays[f"pos_{ci}"], arrays[f"neg_{ci}"] = pos, neg
        cands.append({**c, "usable": usable, "views_used": used_views,
                      "sam_scores": sc_json["scores"].get(c["cid"]),
                      "why_not": None if usable else ("no usable mask" if not used_views else
                                                      f"class {c['cls']} / confidence {c['confidence']}")})
    np.savez_compressed(PDIR / f"{rid}_assoc.npz", **arrays)
    meta = {"region": rid, "model": prop["model"], "proposal_file": prop.get("proposal_file"),
            "sam_model": sc_json["model"], "n_views": len(cams), "seg_dir": sc.seg_dir.name, "candidates": cands}
    (PDIR / f"{rid}_assoc.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[assoc] {rid}: " + ", ".join(f"{c['cid']} {c['cls']} usable={c['usable']}" for c in cands))
    return meta


if __name__ == "__main__":
    rid = sys.argv[1]
    step = sys.argv[2] if len(sys.argv) > 2 else "all"
    cfg = load_config()
    seg = cfg["propose"]["seg_dir"]
    if step in ("render", "all"):
        render_region(load_scene(cfg, seg), rid)
    if step in ("ask", "all"):
        ask(rid, cfg, metadata="--metadata" in sys.argv)
    if step in ("sam", "all"):
        sam_masks(rid, cfg)
    if step in ("assoc", "all"):
        associate(load_scene(cfg, seg), rid, cfg)
