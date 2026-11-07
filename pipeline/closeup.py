"""Close-up calibrated renders for review: python -m pipeline.closeup name ex ey ez tx ty tz [fov] [seg_dir]
Writes out_review/closeups/<name>.png (RGB | object-ID overlay)."""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image

from pipeline.common import ROOT, load_config
from pipeline.scene import Scene, load_scene
from pipeline.survey import short
from pipeline.views import Camera, id_overlay, render


def closeup(sc: Scene, name: str, eye, target, fov: float = 55, W: int = 900, H: int = 680,
            out_sub: str = "closeups") -> Camera:
    cam = Camera.look_at(eye, target, W, H, fov, name)
    v = render(cam, sc.xyz, sc.rgb, sc.scales, sc.opacity)
    gl = sc.g_label
    pix = np.where(v.gid >= 0, gl[np.maximum(v.gid, 0)], -1)
    names = {i: short(a["id"]) for i, a in enumerate(sc.assets)}
    objs = {i for i, a in enumerate(sc.assets) if a["kind"] == "object"}
    ov = id_overlay(v, pix, names, focus=objs, min_px=120)
    im = Image.new("RGB", (W * 2 + 6, H), (255, 255, 255))
    im.paste(Image.fromarray(v.rgb), (0, 0))
    im.paste(ov, (W + 6, 0))
    out = ROOT / "out_review" / out_sub
    out.mkdir(parents=True, exist_ok=True)
    im.save(out / f"{name}.png")
    return cam


if __name__ == "__main__":
    a = sys.argv[1:]
    cfg = load_config()
    sc = load_scene(cfg, a[8] if len(a) > 8 else "out_baseline")
    closeup(sc, a[0], [float(x) for x in a[1:4]], [float(x) for x in a[4:7]], float(a[7]) if len(a) > 7 else 55)
