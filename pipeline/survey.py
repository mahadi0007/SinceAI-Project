"""Survey renders: calibrated views from walkable positions, RGB | baseline object-ID overlay.
Used to choose and document review regions. Output: out_review/survey/*.png
Run: python -m pipeline.survey [seg_dir]
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image

from pipeline.common import ROOT, load_config
from pipeline.scene import load_scene
from pipeline.views import Camera, id_overlay, render

EYE_Y = -0.15     # ~1.5 m above the floor (floor y = 1.174, up = -y)

SURVEY = [  # (name, eye xz, target xz, target y)
    ("c_to_left", (0.3, -0.8), (-4.5, -0.8), 0.6),
    ("c_to_right", (-0.5, -0.8), (4.0, -0.8), 0.6),
    ("c_to_front", (0.5, -1.5), (0.5, 3.3), 0.6),
    ("c_to_back", (0.5, 0.3), (0.5, -3.8), 0.6),
    ("lc_to_left", (-2.9, -1.5), (-4.6, -1.8), 0.7),
    ("lc_to_front", (-2.9, -2.5), (-3.0, 2.0), 0.6),
    ("top_to_front", (0.0, 2.4), (0.0, 3.6), 0.6),
    ("top_to_left", (-1.0, 2.4), (-4.6, 2.6), 0.6),
    ("top_to_right", (1.0, 2.5), (4.0, 2.5), 0.6),
    ("c_to_island", (-0.6, 0.6), (-2.0, -0.6), 0.6),
    ("r_to_right", (2.0, -1.2), (4.0, -2.0), 0.7),
    ("c_to_tanks", (0.5, 0.6), (1.5, 2.0), 0.4),
]


def short(asset_id: str) -> str:
    return asset_id.replace("OBJ-", "").lstrip("0") or "0"


def main(seg_dir: str = "out_baseline") -> None:
    cfg = load_config()
    sc = load_scene(cfg, seg_dir)
    out = ROOT / "out_review" / "survey"
    out.mkdir(parents=True, exist_ok=True)
    gl = sc.g_label
    names = {i: short(a["id"]) for i, a in enumerate(sc.assets)}
    objs = {i for i, a in enumerate(sc.assets) if a["kind"] == "object"}
    for name, e, t, ty in SURVEY:
        cam = Camera.look_at((e[0], EYE_Y, e[1]), (t[0], ty, t[1]), 900, 640, 70, name)
        v = render(cam, sc.xyz, sc.rgb, sc.scales, sc.opacity)
        pix = np.where(v.gid >= 0, gl[np.maximum(v.gid, 0)], -1)
        ov = id_overlay(v, pix, names, focus=objs, min_px=150)
        im = Image.new("RGB", (cam.W * 2 + 6, cam.H), (255, 255, 255))
        im.paste(Image.fromarray(v.rgb), (0, 0))
        im.paste(ov, (cam.W + 6, 0))
        im.save(out / f"{name}.png")
        print(f"[survey] {name}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out_baseline")
