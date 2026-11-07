"""Calibrated perspective views of the splat with per-pixel visibility (Gaussian ID + depth buffers).

Every rendered view stores its exact camera (intrinsics K, world->camera R, centre C, size), an
RGB image, a depth buffer and an ID buffer naming the Gaussian that is VISIBLE at each pixel.
2D masks are associated with 3D support only through that ID buffer, so:
  * occluded Gaussians never receive mask support (no projection through walls/pipes in front);
  * a Gaussian whose centre projects outside a mask still counts where its footprint is visible.

Visibility approximation (documented, not a full splat renderer):
  * each Gaussian with opacity >= min_opacity (0.2) is drawn as an opaque disc whose screen radius is
    k_sigma (2.0) * sqrt(s1*s2) (two largest scales) * f / depth, clipped to [0.5, r_max] px;
  * nearest disc wins per pixel (z-buffer on the centre depth); Gaussians below min_opacity
    neither occlude nor receive support. Alpha blending and anisotropic footprints are ignored.
Native coordinates: up = -y. Camera frame: x right, y down, z forward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Cfg = dict[str, Any]


@dataclass
class Camera:
    C: np.ndarray            # centre (3,)
    R: np.ndarray            # world -> camera rotation (3,3), rows = x(right), y(down), z(forward)
    fx: float
    fy: float
    cx: float
    cy: float
    W: int
    H: int
    name: str = ""

    @staticmethod
    def look_at(eye: Any, target: Any, W: int = 768, H: int = 576, fov_deg: float = 60.0,
                name: str = "") -> "Camera":
        eye, target = np.asarray(eye, float), np.asarray(target, float)
        z = target - eye
        z /= np.linalg.norm(z)
        down = np.array([0.0, 1.0, 0.0])                      # native up is -y
        x = np.cross(down, z)
        if np.linalg.norm(x) < 1e-6:                          # looking straight up/down
            x = np.array([1.0, 0.0, 0.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        f = 0.5 * W / np.tan(np.radians(fov_deg) / 2)
        return Camera(eye, np.stack([x, y, z]), f, f, (W - 1) / 2, (H - 1) / 2, W, H, name)

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pc = (X - self.C) @ self.R.T
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * pc[:, 0] / z + self.cx
            v = self.fy * pc[:, 1] / z + self.cy
        return u, v, z

    def to_json(self) -> dict:
        return {"name": self.name, "C": self.C.round(5).tolist(), "R": self.R.round(6).tolist(),
                "K": [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], "W": self.W, "H": self.H}

    @staticmethod
    def from_json(d: dict) -> "Camera":
        K = d["K"]
        return Camera(np.array(d["C"]), np.array(d["R"]), K[0][0], K[1][1], K[0][2], K[1][2],
                      d["W"], d["H"], d.get("name", ""))


@dataclass
class View:
    cam: Camera
    rgb: np.ndarray          # (H, W, 3) uint8
    depth: np.ndarray        # (H, W) float32, inf = background
    gid: np.ndarray          # (H, W) int64, visible Gaussian index or -1
    meta: dict = field(default_factory=dict)


def render(cam: Camera, xyz: np.ndarray, rgb: np.ndarray, scales: np.ndarray, opacity: np.ndarray,
           min_opacity: float = 0.2, k_sigma: float = 2.0, r_max: int = 7, near: float = 0.05,
           far: float = 12.0, bg: tuple[int, int, int] = (235, 235, 235), chunk: int = 400_000) -> View:
    u, v, z = cam.project(xyz)
    s2 = np.sort(scales, 1)[:, 1:]
    r = np.clip(k_sigma * np.sqrt(s2[:, 0] * s2[:, 1]) * cam.fx / np.maximum(z, near), 0.5, r_max)
    ok = ((z > near) & (z < far) & (opacity >= min_opacity) & (u > -r_max) & (u < cam.W + r_max)
          & (v > -r_max) & (v < cam.H + r_max))
    idx = np.where(ok)[0]
    HW = cam.W * cam.H
    zbuf = np.full(HW, np.inf, np.float32)
    gbuf = np.full(HW, -1, np.int64)
    offs = {R: [(dx, dy) for dx in range(-R, R + 1) for dy in range(-R, R + 1) if dx * dx + dy * dy <= R * R + 0.5]
            for R in range(0, r_max + 1)}
    ri = np.round(r[idx]).astype(int)
    for R, off in offs.items():
        sel = idx[ri == R]
        for k in range(0, len(sel), max(1, chunk // len(off))):
            g = sel[k:k + max(1, chunk // len(off))]
            U = (np.round(u[g])[None, :] + np.array([o[0] for o in off])[:, None]).astype(np.int64).ravel()
            V = (np.round(v[g])[None, :] + np.array([o[1] for o in off])[:, None]).astype(np.int64).ravel()
            G = np.tile(g, len(off))
            D = np.tile(z[g], len(off)).astype(np.float32)
            m = (U >= 0) & (U < cam.W) & (V >= 0) & (V < cam.H)
            flat, G, D = V[m] * cam.W + U[m], G[m], D[m]
            order = np.lexsort((D, flat))
            flat, G, D = flat[order], G[order], D[order]
            first = np.ones(len(flat), bool)
            first[1:] = flat[1:] != flat[:-1]
            flat, G, D = flat[first], G[first], D[first]
            better = D < zbuf[flat]
            zbuf[flat[better]] = D[better]
            gbuf[flat[better]] = G[better]
    img = np.empty((HW, 3), np.uint8)
    img[:] = bg
    hit = gbuf >= 0
    img[hit] = (np.clip(rgb[gbuf[hit]], 0, 1) * 255).astype(np.uint8)
    return View(cam, img.reshape(cam.H, cam.W, 3), zbuf.reshape(cam.H, cam.W), gbuf.reshape(cam.H, cam.W))


# ---------- overlays ----------

def palette(n: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hues = (np.arange(n) * 0.618034) % 1.0
    out = []
    for h in hues:
        s, v = 0.85, rng.uniform(0.75, 1.0)
        i = int(h * 6) % 6
        f = h * 6 - int(h * 6)
        p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
        out.append([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i])
    return (np.array(out) * 255).astype(np.uint8)


def id_overlay(view: View, pix_obj: np.ndarray, names: dict[int, str], focus: set[int] | None = None,
               min_px: int = 60, alpha: float = 0.65) -> Image.Image:
    """Colour visible pixels by object id (pix_obj: (H,W) int, -1 none) and write each object's
    short name at the centroid of its visible pixels. Non-focus objects are greyed."""
    base = view.rgb.astype(np.float32)
    grey = base.mean(2, keepdims=True).repeat(3, 2) * 0.55 + 100
    out = grey.copy()
    ids = [i for i in np.unique(pix_obj) if i >= 0 and (focus is None or i in focus)]
    pal = palette(max(len(ids), 1))
    for k, i in enumerate(ids):
        m = pix_obj == i
        out[m] = (1 - alpha) * base[m] + alpha * pal[k]
    im = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for k, i in enumerate(ids):
        ys, xs = np.nonzero(pix_obj == i)
        if len(xs) < min_px:
            continue
        # label at the visible pixel closest to the centroid (stays on the object)
        j = np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)
        t = names.get(i, str(i))
        x, y = int(xs[j]), int(ys[j])
        bb = dr.textbbox((x, y), t, font=font, anchor="mm")
        dr.rectangle([bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1], fill=(255, 255, 255))
        dr.text((x, y), t, fill=tuple(int(c * 0.6) for c in pal[k]), font=font, anchor="mm")
    return im


def draw_boxes(im: Image.Image, boxes: list[tuple[list[float], str, tuple[int, int, int]]], width: int = 3) -> Image.Image:
    im = im.copy()
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for b, t, c in boxes:
        dr.rectangle(b, outline=c, width=width)
        if t:
            dr.text((b[0] + 3, b[1] + 2), t, fill=c, font=font)
    return im


def mask_outline(im: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.35) -> Image.Image:
    a = np.asarray(im).astype(np.float32)
    a[mask] = (1 - alpha) * a[mask] + alpha * np.array(color)
    edge = mask & ~(np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1))
    a[edge] = color
    return Image.fromarray(a.astype(np.uint8))
