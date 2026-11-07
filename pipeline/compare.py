"""Before/after inspection images per review region: baseline | geometry-only | geometry+context,
each as an object-ID overlay of the same calibrated view, with reviewed reference boxes projected (white)
and the repaired objects' boxes (magenta). Output: out_review/compare/<region>_v<k>.png
Run: python -m pipeline.compare [RV-01 ...]"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline.common import ROOT, load_config
from pipeline.propose import load_views, regions
from pipeline.scene import load_scene
from pipeline.survey import short
from pipeline.views import Camera, View, id_overlay

EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def box_lines(cam: Camera, lo, hi) -> list[tuple[float, float, float, float]]:
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)
    u, v, z = cam.project(c)
    return [(u[a], v[a], u[b], v[b]) for a, b in EDGES if z[a] > 0.05 and z[b] > 0.05]


def panel(sc, gid: np.ndarray, rgb: np.ndarray, cam: Camera, title: str, refs: list[dict],
          highlight: list[dict]) -> Image.Image:
    gl = sc.g_label
    pix = np.where(gid >= 0, gl[np.maximum(gid, 0)], -1)
    names = {i: short(a["id"]) for i, a in enumerate(sc.assets)}
    objs = {i for i, a in enumerate(sc.assets) if a["kind"] == "object"}
    im = id_overlay(View(cam, rgb, np.zeros(gid.shape, np.float32), gid), pix, names, focus=objs, min_px=80)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for r in refs:
        for ln in box_lines(cam, *r["box"]):
            dr.line(ln, fill=(255, 255, 255), width=2)
    for a in highlight:
        for ln in box_lines(cam, a["bbox_min"], a["bbox_max"]):
            dr.line(ln, fill=(255, 0, 255), width=2)
    dr.rectangle([0, 0, im.width, 22], fill=(255, 255, 255))
    dr.text((6, 3), title, fill=(0, 0, 0), font=font)
    return im


def main(rids: list[str]) -> None:
    cfg = load_config()
    variants = [("out_baseline", "baseline"), ("out_geo", "geometry-only"), (cfg["paths"]["out_dir"], "geometry + context")]
    scenes = {d: load_scene(cfg, d) for d, _ in variants}
    out = ROOT / "out_review" / "compare"
    out.mkdir(parents=True, exist_ok=True)
    for rid in rids:
        reg = regions()[rid]
        cams, rgbs, gids, _ = load_views(rid)
        for k, (cam, rgb, gid) in enumerate(zip(cams, rgbs, gids)):
            ims = []
            for d, name in variants:
                sc = scenes[d]
                hl = [a for a in sc.assets if a.get("repair", {}).get("family") == "context"
                      and a["repair"].get("region") == rid]
                n = sum(a["kind"] == "object" for a in sc.assets)
                ims.append(panel(sc, gid, rgb, cam, f"{rid} view {k} - {name} ({n} objects)", reg["refs"], hl))
            W, H = ims[0].size
            sheet = Image.new("RGB", (W * len(ims) + 8 * (len(ims) - 1), H), (255, 255, 255))
            for i, im in enumerate(ims):
                sheet.paste(im, (i * (W + 8), 0))
            sheet.save(out / f"{rid}_v{k}.png")
            print(f"[compare] {out / f'{rid}_v{k}.png'}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(regions()))
