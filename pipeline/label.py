"""Stage 2: name extracted objects with a vision LLM (Featherless Qwen3.5, zero-shot, fixed vocabulary).

For each object we render a 3-panel image from the full splat (object front | object side |
context with red box), send small batches to the VLM, and keep {label, confidence, reason}.
Answers are validated (JSON, current-batch IDs, allowed labels, finite confidence in [0,1]);
low-confidence / failed answers fall back to geometric rules. Cache key = hash of provider,
model, prompt, object description and the exact image bytes, so stale labels are never reused
and good labels survive re-runs.
Inputs : out/assets.json, out/segment_points.npz, the PLY
Outputs: out/assets_labeled.json, out/crops/OBJ-xxxx.png, out/label_contact_sheet.png, out/label_cache.json
Run    : python -m pipeline.label     (--rules-only: no requests; --batches N: paid batches this run)
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from typing import Any

import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

from pipeline.common import ROOT, load_config, out_path, repo_path
from pipeline.load_splat import load_splat

Cfg = dict[str, Any]


# ---------- data ----------

def load_inputs(cfg: Cfg) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """Assets + full-resolution splat points with an object label per point."""
    with open(out_path(cfg, "assets.json"), encoding="utf-8") as f:
        assets = json.load(f)
    seg = np.load(out_path(cfg, "segment_points.npz"))
    lc = cfg["load"]
    d = load_splat(str(repo_path(cfg["paths"]["ply"])), lc["min_opacity"], lc["max_scale"], lc["crop_floaters"])
    dist, j = cKDTree(seg["xyz"]).query(d["xyz"], k=1)
    lab = np.where(dist <= cfg["label"]["member_dist"], seg["labels"][j], -1)
    return assets, d["xyz"].astype(np.float64), d["rgb"], lab


# ---------- rendering ----------

def raster(pts: np.ndarray, rgb: np.ndarray, h_axis: int, v_axis: int, depth_axis: int,
           near_is_high: bool, lo: np.ndarray, hi: np.ndarray, px: int, splat_px: int,
           bg: tuple[int, int, int]) -> Image.Image:
    """Orthographic z-buffered point render of the window [lo, hi] (2D: h, v). Up = -y."""
    img = np.full((px, px, 3), bg, np.uint8)
    if len(pts) == 0:
        return Image.fromarray(img)
    span = max(hi[0] - lo[0], hi[1] - lo[1]) * 1.08 + 1e-6
    c = (lo + hi) / 2
    u = ((pts[:, h_axis] - c[0]) / span + 0.5) * (px - 1)
    v = ((pts[:, v_axis] - c[1]) / span + 0.5) * (px - 1)
    if v_axis != 1:                         # native up is -y: larger y = lower on screen
        v = (px - 1) - v
    depth = pts[:, depth_axis] * (-1 if near_is_high else 1)      # smaller = nearer
    offs = [(dx, dy) for dx in range(-splat_px, splat_px + 1) for dy in range(-splat_px, splat_px + 1)]
    U = np.concatenate([u + dx for dx, _ in offs]).astype(int)
    V = np.concatenate([v + dy for _, dy in offs]).astype(int)
    D = np.tile(depth, len(offs))
    C = np.tile((np.clip(rgb, 0, 1) * 255).astype(np.uint8), (len(offs), 1))
    ok = (U >= 0) & (U < px) & (V >= 0) & (V < px)
    U, V, D, C = U[ok], V[ok], D[ok], C[ok]
    order = np.argsort(D, kind="stable")
    flat = V[order] * px + U[order]
    _, first = np.unique(flat, return_index=True)
    sel = order[first]
    img[V[sel], U[sel]] = C[sel]
    return Image.fromarray(img)


def object_panels(a: dict, idx: int, xyz: np.ndarray, rgb: np.ndarray, lab: np.ndarray,
                  lc: Cfg) -> Image.Image:
    """[front (look -z) | side (look -x) | context front with red box]."""
    px, sp, bg = lc["panel_px"], lc["splat_px"], tuple(lc["background"])
    m = lab == idx
    P, Cc = xyz[m], rgb[m]
    lo, hi = np.array(a["bbox_min"]), np.array(a["bbox_max"])
    pad = np.maximum((hi - lo) * 0.05, 0.02)
    front = raster(P, Cc, 0, 1, 2, True, (lo - pad)[[0, 1]], (hi + pad)[[0, 1]], px, sp, bg)
    side = raster(P, Cc, 2, 1, 0, True, (lo - pad)[[2, 1]], (hi + pad)[[2, 1]], px, sp, bg)
    mg = lc["context_margin"]
    clo, chi = lo - mg, hi + mg
    dz = lc["context_depth"]               # only the object's depth slab, not the whole room behind it
    inwin = ((xyz[:, 0] >= clo[0]) & (xyz[:, 0] <= chi[0]) & (xyz[:, 1] >= clo[1]) & (xyz[:, 1] <= chi[1])
             & (xyz[:, 2] >= lo[2] - dz) & (xyz[:, 2] <= hi[2] + dz))
    ctx = raster(xyz[inwin], rgb[inwin], 0, 1, 2, True, clo[[0, 1]], chi[[0, 1]], px, max(sp - 1, 1), bg)
    span = max(chi[0] - clo[0], chi[1] - clo[1]) * 1.08
    cc = (clo[[0, 1]] + chi[[0, 1]]) / 2

    def to_px(x: float, y: float) -> tuple[float, float]:
        return ((x - cc[0]) / span + 0.5) * (px - 1), ((y - cc[1]) / span + 0.5) * (px - 1)
    ImageDraw.Draw(ctx).rectangle([to_px(lo[0], lo[1]), to_px(hi[0], hi[1])], outline=(230, 0, 0), width=3)
    out = Image.new("RGB", (px * 3 + 8, px), (255, 255, 255))
    for k, im in enumerate((front, side, ctx)):
        out.paste(im, (k * (px + 4), 0))
    return out


# ---------- vision LLM ----------

PROMPT = """You label objects extracted from a 3D Gaussian-splat scan of an industrial equipment room
(pipes, valves, pumps, ladders/cages, cabinets, doors). Each image has 3 panels: LEFT = the object alone,
front view; MIDDLE = the object alone, side view; RIGHT = the object in its surroundings (red box).
Renders come from a sparse noisy scan, so they look blurry. Real-world size and height are given.
Objects may be FRAGMENTS of larger equipment; label what the fragment is part of only if clearly visible.

Choose exactly one label per object from this list:
{vocab}
Use "other" if none fits; be honest with confidence (0-1). Return ONLY a JSON array:
[{{"id": "OBJ-0001", "label": "...", "confidence": 0.0, "reason": "max 12 words"}}]"""


def describe(a: dict, u2m: float) -> str:
    e = np.array(a["extent"]) * u2m
    f = a["features"]
    return (f"{a['id']}: size W{e[0]:.2f} x H{e[1]:.2f} x D{e[2]:.2f} m, bottom {f['height_bottom'] * u2m:.2f} m "
            f"above floor, linearity {f['linearity']:.2f}, planarity {f['planarity']:.2f}, "
            f"geometry hint: {a['hint'] or 'none'}{', overhead' if 'overhead' in a.get('tags', []) else ''}")


def png_bytes(im: Image.Image) -> bytes:
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def parse_list(text: str) -> list[dict]:
    t = text.strip()
    t = t[t.index("["): t.rindex("]") + 1] if "[" in t else t
    res = json.loads(t)
    if isinstance(res, dict):
        res = next((v for v in res.values() if isinstance(v, list)), [res])
    return [r for r in res if isinstance(r, dict)]


def cache_key(a: dict, png: bytes, cfg: Cfg) -> str:
    cc = cfg["copilot"]
    h = hashlib.sha256()
    for part in (cc["provider"], cc["model"], PROMPT, "|".join(cfg["label"]["vocab"]),
                 describe(a, cfg["scene"]["units_to_m"])):
        h.update(part.encode())
    h.update(hashlib.sha256(png).digest())
    return h.hexdigest()[:24]


def finite_conf(x: Any) -> float | None:
    try:
        c = float(x)
    except (TypeError, ValueError):
        return None
    return c if c == c and 0.0 <= c <= 1.0 else None


def validate_labels(res: list[dict], batch_ids: set[str], vocab: list[str]) -> tuple[list[dict], list[str]]:
    """Keep answers with an ID from THIS batch, an allowed label and a finite confidence in [0, 1]."""
    ok, bad = [], []
    seen: set[str] = set()
    for r in res:
        rid = str(r.get("id", "")).strip()
        conf = finite_conf(r.get("confidence"))
        why = ("unknown id" if rid not in batch_ids else "duplicate id" if rid in seen else
               "label not in vocab" if r.get("label") not in vocab else
               "bad confidence" if conf is None else "")
        if why:
            bad.append(f"{rid or '?'}: {why}")
            continue
        seen.add(rid)
        ok.append({"id": rid, "label": r["label"], "confidence": round(conf, 3),
                   "reason": str(r.get("reason", ""))[:120]})
    return ok, bad


def vlm_batch(batch: list[tuple[dict, bytes]], cfg: Cfg) -> tuple[list[dict], list[str], str]:
    from copilot.llm import generate
    lc = cfg["label"]
    parts: list[Any] = [PROMPT.format(vocab=", ".join(lc["vocab"]))]
    for a, png in batch:
        parts += [describe(a, cfg["scene"]["units_to_m"]), png]
    text, model = generate(parts, cfg["copilot"], json_mode=True, temperature=0.1, purpose="label",
                           max_tokens=60 * len(batch) + 200)
    out_path(cfg, "label_raw").mkdir(exist_ok=True)
    (out_path(cfg, "label_raw") / f"{batch[0][0]['id']}.txt").write_text(f"{model}\n{text}", encoding="utf-8")
    ok, bad = validate_labels(parse_list(text), {a["id"] for a, _ in batch}, lc["vocab"])
    return ok, bad, model


IDENTIFY = """This image shows one object extracted from a 3D Gaussian-splat scan of an industrial equipment room.
LEFT = the object alone (front), MIDDLE = side view, RIGHT = in its surroundings (red box). The render is blurry.
Facts: {facts}
In max 70 words of markdown: **what it most likely is**, what visual cues support that, and one alternative
if unsure. Do not invent asset tags or health data."""


def identify_object(node: dict, cfg: Cfg) -> str:
    """On-demand free-text identification for the viewer's click panel (cached per object; paid)."""
    from copilot.llm import LLMUnavailable, generate
    cache_p = out_path(cfg, "identify_cache.json")
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}
    crop = out_path(cfg, "crops") / f"{node['id']}.png"
    if not crop.exists():
        return "_No picture for this object; run `python -m pipeline.label` first._"
    k = hashlib.sha256(crop.read_bytes() + cfg["copilot"]["model"].encode()).hexdigest()[:16]
    if k in cache:
        return cache[k]
    try:
        text, model = generate([IDENTIFY.format(facts=describe(node, cfg["scene"]["units_to_m"])),
                                crop.read_bytes()], cfg["copilot"], json_mode=False, temperature=0.2,
                               purpose="viewer_identify", max_tokens=200)
    except LLMUnavailable as e:
        return f"_Vision model not called: {str(e)[:140]}._"
    except Exception as e:  # noqa: BLE001
        return f"_Vision model unavailable right now ({str(e)[:90]})._"
    md = f"{text.strip()}\n\n_{model}_"
    cache[k] = md
    cache_p.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return md


# ---------- rules fallback ----------

def label_by_rules(a: dict) -> str:
    if a["kind"] == "structure":
        return a["hint"]
    f, ext = a["features"], max(a["extent"])
    if a["hint"] == "pipe segment" or (f["linearity"] > 0.85 and ext > 0.4):
        return "pipe segment"
    if a["hint"] == "fitting/valve":
        return "pipe fitting/flange"
    if f["planarity"] > 0.5 and ext > 1.0:
        return "wall"
    return "other"


def decide(a: dict, g: dict | None, lc: Cfg) -> dict:
    rule = label_by_rules(a)
    if a["kind"] == "structure":
        return {"label": rule, "label_confidence": 1.0, "label_source": "plane_fit", "label_reason": "structural plane"}
    conf = finite_conf(g.get("confidence")) if g else None
    if g and g.get("label") in lc["vocab"] and conf is not None and conf >= lc["min_conf"]:
        return {"label": g["label"], "label_confidence": round(conf, 2),
                "label_source": g.get("source", "featherless_vision"), "label_model": g.get("model"),
                "label_reason": g.get("reason", "")}
    out = {"label": rule, "label_confidence": lc["rules_conf"], "label_source": "geometry_rules",
           "label_reason": "fallback: low/no vision confidence"}
    if g:
        out["label_vlm"] = {k: g.get(k) for k in ("label", "confidence", "reason", "model")}
    return out


def manual_decision(a: dict, lc: Cfg) -> dict | None:
    """Return a confident reviewer class label when the correction metadata resolves to the vocab."""
    manual = a.get("manual")
    if not manual or a.get("label_source") not in lc.get("manual_sources", []):
        return None
    cid = manual.get("component_id", a["id"])
    cls = manual.get("component_class")
    label = lc.get("manual_label_overrides", {}).get(cid) or lc.get("manual_class_vocab", {}).get(cls)
    if label not in lc["vocab"]:
        return None
    return {"label": label, "label_confidence": 1.0, "label_source": "manual_review",
            "label_reason": (f"reviewer boundary correction {cid} rev {manual.get('revision')} "
                             f"({manual.get('evidence_level')})")}


def vlm_todo(assets: list[dict], key: dict[str, str], cache: dict,
             only_ids: set[str] | None = None) -> list[dict]:
    """Uncached non-manual objects eligible for a paid vision request."""
    return [a for a in assets if a["kind"] == "object" and not a.get("manual")
            and key[a["id"]] not in cache and (only_ids is None or a["id"] in only_ids)]


# ---------- contact sheet ----------

def contact_sheet(assets: list[dict], crops: dict[str, Image.Image], path, lc: Cfg) -> None:
    objs = [a for a in assets if a["kind"] == "object"]
    t, cols = lc["thumb_px"], lc["sheet_cols"]
    rows = (len(objs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * t, rows * (t + 28)), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for k, a in enumerate(objs):
        im = crops[a["id"]]
        third = im.width // 3
        thumb = im.crop((0, 0, third, im.height)).resize((t, t))
        x, y = (k % cols) * t, (k // cols) * (t + 28)
        sheet.paste(thumb, (x, y))
        col = (0, 120, 0) if a["label_source"] in lc["vision_sources"] else (150, 90, 0)
        dr.text((x + 2, y + t + 1), a["id"].replace("OBJ-", "#"), fill=(60, 60, 60), font=font)
        dr.text((x + 2, y + t + 13), f"{a['label'][:20]} {a['label_confidence']:.2f}", fill=col, font=font)
    sheet.save(path)
    print(f"[plot] {path}")


# ---------- reuse of earlier vision labels ----------

def geo_sig(a: dict) -> str:
    """Identity of an object's geometry: point count + centre (mm). Same segmentation -> same signature."""
    return f"{a['n_points']}|{[round(float(v), 3) for v in a['center']]}"


def reusable_labels(cfg: Cfg, cache: dict) -> dict[str, dict]:
    """Earlier vision labels by geometry signature, for objects the current cache key misses (labels made by an
    earlier label.py or another provider). Sources: old cache entries that carry a 'sig', and label.reuse_from
    files (assets_labeled.json of a previous run, read before it is overwritten). Free, no requests."""
    lc = cfg["label"]
    out: dict[str, dict] = {}
    for v in cache.values():                                    # old cache format: {id: {..., "sig": ...}}
        if isinstance(v, dict) and v.get("sig") and v.get("label"):
            out.setdefault(v["sig"], {**v, "source": v.get("source") or "gemini_vision"})
    for rel in lc.get("reuse_from") or []:
        f = repo_path(rel)
        if not f.exists():
            continue
        for a in json.loads(f.read_text(encoding="utf-8")):
            if a.get("kind") == "object" and a.get("label_source") in lc["vision_sources"]:
                out[geo_sig(a)] = {"label": a["label"], "confidence": a.get("label_confidence"),
                                   "reason": a.get("label_reason", ""), "model": a.get("label_model"),
                                   "source": a["label_source"]}
    return out


# ---------- main ----------

def run(cfg: Cfg, rules_only: bool = False, max_batches: int | None = None,
        only_ids: set[str] | None = None) -> list[dict]:
    lc = cfg["label"]
    load_dotenv(ROOT / ".env")
    assets, xyz, rgb, lab = load_inputs(cfg)
    crop_dir = out_path(cfg, "crops")
    crop_dir.mkdir(exist_ok=True)
    crops: dict[str, Image.Image] = {}
    for i, a in enumerate(assets):
        if a["kind"] == "object":
            crops[a["id"]] = object_panels(a, i, xyz, rgb, lab, lc)
            crops[a["id"]].save(crop_dir / f"{a['id']}.png")
    print(f"[render] {len(crops)} crops -> {crop_dir}")

    cache_p = out_path(cfg, "label_cache.json")
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}
    reuse = reusable_labels(cfg, cache)
    pngs = {k: png_bytes(im.resize((lc["send_px"] * 3, lc["send_px"]))) for k, im in crops.items()}
    key = {a["id"]: cache_key(a, pngs[a["id"]], cfg) for a in assets if a["kind"] == "object"}
    todo = [a for a in vlm_todo(assets, key, cache, only_ids) if geo_sig(a) not in reuse]   # never pay twice
    if reuse:
        print(f"[reuse] {sum(geo_sig(a) in reuse for a in assets if a['kind'] == 'object')} objects keep an "
              f"earlier vision label (identical geometry)")
    # most informative first (equipment/valves/unnamed, then big pipes) so a budget cut-off hurts least
    todo.sort(key=lambda a: (a["hint"] == "pipe segment", -a["n_points"]))
    if not rules_only and todo:
        from copilot.llm import LLMUnavailable
        batches = [todo[k:k + lc["batch_size"]] for k in range(0, len(todo), lc["batch_size"])]
        if max_batches is not None:
            batches = batches[:max_batches]
        t0, n_ok = time.time(), 0
        for b in batches:                       # sequential, one worker, no retry
            try:
                res, bad, model = vlm_batch([(a, pngs[a["id"]]) for a in b], cfg)
            except LLMUnavailable as e:
                print(f"  [stop] {e}")
                break
            except Exception as e:  # noqa: BLE001 - recorded in the ledger; never retried
                print(f"  [warn] batch {b[0]['id']}..: {str(e)[:160]}")
                continue
            for r in res:
                cache[key[r["id"]]] = {**r, "model": model, "source": "featherless_vision",
                                       "at": time.strftime("%Y-%m-%d %H:%M")}
            n_ok += len(res)
            print(f"  batch {b[0]['id']}: {len(res)}/{len(b)} valid labels ({model})"
                  + (f"; rejected {bad}" if bad else ""))
            cache_p.write_text(json.dumps(cache, indent=1), encoding="utf-8")   # save as we go
        print(f"[vlm] {n_ok}/{sum(map(len, batches))} objects labelled in {time.time() - t0:.0f}s "
              f"({len(todo)} uncached in total)")

    for a in assets:
        if a.get("manual"):
            a["label_manual_class"] = a["manual"].get("component_class")
        manual = manual_decision(a, lc)
        if manual:
            a.update(manual)
            continue
        # per-object VLM label from the label cache; else the validated contextual proposal (repair stage)
        g = (cache.get(key.get(a["id"], "")) or (reuse.get(geo_sig(a)) if a["kind"] == "object" else None)
             or a.get("vlm_proposal"))
        a.update(decide(a, g, lc))
    with open(out_path(cfg, "assets_labeled.json"), "w", encoding="utf-8") as f:
        json.dump(assets, f, indent=1)

    objs = [a for a in assets if a["kind"] == "object"]
    hist: dict[str, int] = {}
    for a in objs:
        hist[a["label"]] = hist.get(a["label"], 0) + 1
    n_vlm = sum(a["label_source"] in lc["vision_sources"] for a in objs)
    n_manual = sum(a["label_source"] in lc.get("manual_sources", []) for a in objs)
    print(f"[labels] {n_vlm}/{len(objs)} from vision LLM (conf >= {lc['min_conf']}), "
          f"{n_manual} from manual review, rest geometry rules")
    print("[labels] " + ", ".join(f"{k}: {v}" for k, v in sorted(hist.items(), key=lambda kv: -kv[1])))
    contact_sheet(assets, crops, out_path(cfg, "label_contact_sheet.png"), lc)
    return assets


def export_vision_labels(cfg: Cfg, dest: str) -> int:
    """Write the vision labels of out/assets_labeled.json as a small shareable file (geometry signature + label),
    in the assets_labeled format that label.reuse_from reads. Committed, it gives every clone the same labels."""
    lc = cfg["label"]
    keep = ("id", "kind", "n_points", "center", "label", "label_confidence", "label_source", "label_reason",
            "label_model")
    rows = [{k: a[k] for k in keep if k in a} for a in json.loads(out_path(cfg, "assets_labeled.json").read_text(
        encoding="utf-8")) if a.get("kind") == "object" and a.get("label_source") in lc["vision_sources"]]
    p = repo_path(dest)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("[\n" + ",\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n]\n", encoding="utf-8")
    print(f"[export] {len(rows)} vision labels -> {p}")
    return len(rows)


if __name__ == "__main__":
    cfg = load_config()
    if "--export-labels" in sys.argv:                 # python -m pipeline.label --export-labels [path]
        i = sys.argv.index("--export-labels")
        export_vision_labels(cfg, sys.argv[i + 1] if len(sys.argv) > i + 1 else cfg["label"]["shared_labels"])
        sys.exit(0)
    mb = int(sys.argv[sys.argv.index("--batches") + 1]) if "--batches" in sys.argv else cfg["label"]["max_batches"]
    run(cfg, rules_only="--rules-only" in sys.argv, max_batches=mb)
