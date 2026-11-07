"""Reviewer tool: turn a pixel box drawn on an EWA view into an approximate 3D reference box.
Pixels in the box are unprojected with the EWA MEDIAN depth; only the front surface band
[d_front, d_front + band] is kept (d_front = 5th percentile), then a 3-97 percentile AABB.
Run: python -m pipeline.review_refs RV-10 0 x0 y0 x1 y1 [band]"""
from __future__ import annotations

import sys

import numpy as np

from pipeline.propose import load_views


def px_box_to_3d(rid: str, view: int, box: list[float], band: float = 0.25) -> list[list[float]]:
    cams, _, bufs = load_views(rid, "ewa")
    cam, dep = cams[view], bufs[view]["depth"]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    z = dep[ys, xs].ravel()
    ok = np.isfinite(z)
    xs, ys, z = xs.ravel()[ok], ys.ravel()[ok], z[ok]
    front = np.percentile(z, 5)
    keep = z <= front + band
    xs, ys, z = xs[keep], ys[keep], z[keep]
    pc = np.stack([(xs - cam.cx) / cam.fx * z, (ys - cam.cy) / cam.fy * z, z], 1)
    P = pc @ cam.R + cam.C
    lo, hi = np.percentile(P, 3, 0), np.percentile(P, 97, 0)
    return [np.round(lo, 3).tolist(), np.round(hi, 3).tolist()]


if __name__ == "__main__":
    a = sys.argv[1:]
    print(px_box_to_3d(a[0], int(a[1]), [float(v) for v in a[2:6]], float(a[6]) if len(a) > 6 else 0.25))
