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


# ---------- EWA Gaussian splatting (CPU, NumPy) ----------

@dataclass
class SplatView:
    """Alpha-composited Gaussian render with explicit per-Gaussian contribution weights.

    rgb      (H,W,3) uint8   composited colour over the background
    alpha    (H,W)   float32 accumulated opacity (1 - final transmittance)
    depth    (H,W)   float32 MEDIAN depth (where transmittance crosses 0.5), inf if never
    depth_exp(H,W)   float32 expected depth sum(w z)/sum(w) (diagnostic only)
    frag_*   sparse contributions: pixel index, Gaussian index, weight w = alpha_i * T_i (>= w_min)
    The weights are the visibility evidence: occluded Gaussians get T ~ 0, hence ~0 weight.
    """
    cam: Camera
    rgb: np.ndarray
    alpha: np.ndarray
    depth: np.ndarray
    depth_exp: np.ndarray
    frag_pix: np.ndarray
    frag_gid: np.ndarray
    frag_w: np.ndarray

    def top_id(self) -> np.ndarray:
        """Dominant contributor per pixel (-1 if none) - for overlays only, not for support."""
        HW = self.cam.W * self.cam.H
        out = np.full(HW, -1, np.int64)
        if len(self.frag_w):
            o = np.lexsort((-self.frag_w, self.frag_pix))
            p = self.frag_pix[o]
            first = np.r_[True, p[1:] != p[:-1]]
            out[p[first]] = self.frag_gid[o][first]
        return out.reshape(self.cam.H, self.cam.W)


def quat_cov(scales: np.ndarray, quats_wxyz: np.ndarray) -> np.ndarray:
    from pipeline.load_splat import covariances
    return covariances(scales, quats_wxyz)


def render_ewa(cam: Camera, xyz: np.ndarray, rgb: np.ndarray, scales: np.ndarray, quats: np.ndarray,
               opacity: np.ndarray, near: float = 0.05, far: float = 12.0, r_max: int = 64, tile: int = 64,
               dilation: float = 0.3, w_min: float = 0.004, t_min: float = 1e-4,
               bg: tuple[int, int, int] = (235, 235, 235)) -> SplatView:
    """EWA splatting (Zwicker et al.; same model as 3DGS): 3D covariance R S S^T R^T from quaternion +
    anisotropic scales, projected with the perspective Jacobian, +dilation px^2 low-pass, 3-sigma
    footprint, front-to-back alpha compositing sorted by camera depth per pixel. View-independent colour
    (SH degree 0). Global per-pixel depth sort (not per-tile), so no tile-boundary ordering artefacts."""
    W, H = cam.W, cam.H
    pc = (xyz - cam.C) @ cam.R.T
    z = pc[:, 2]
    keep = (z > near) & (z < far) & (opacity >= 1 / 255)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam.fx * pc[:, 0] / z + cam.cx
        v = cam.fy * pc[:, 1] / z + cam.cy
    keep &= (u > -r_max) & (u < W + r_max) & (v > -r_max) & (v < H + r_max)
    idx = np.where(keep)[0]
    P, zz = pc[idx], z[idx]
    S = cam.R @ quat_cov(scales[idx], quats[idx]) @ cam.R.T                       # camera-frame covariance
    J = np.zeros((len(idx), 2, 3))
    J[:, 0, 0] = cam.fx / zz
    J[:, 0, 2] = -cam.fx * P[:, 0] / zz ** 2
    J[:, 1, 1] = cam.fy / zz
    J[:, 1, 2] = -cam.fy * P[:, 1] / zz ** 2
    C2 = J @ S @ np.transpose(J, (0, 2, 1))
    a, b, c = C2[:, 0, 0] + dilation, C2[:, 0, 1], C2[:, 1, 1] + dilation
    det = a * c - b * b
    ok = det > 1e-12
    lam = 0.5 * (a + c) + np.sqrt(np.maximum(0.25 * (a - c) ** 2 + b * b, 0))
    r = np.minimum(np.ceil(3 * np.sqrt(lam)), r_max)
    idx, zz, a, b, c, det, r = idx[ok], zz[ok], a[ok], b[ok], c[ok], det[ok], r[ok]
    uu, vv = u[idx], v[idx]
    ca, cb, cc = c / det, -b / det, a / det                                         # conic (inverse cov)
    op = opacity[idx]
    col = np.clip(rgb[idx], 0, 1)
    x0 = np.clip(np.floor(uu - r), 0, W - 1).astype(np.int64)
    x1 = np.clip(np.ceil(uu + r), 0, W - 1).astype(np.int64)
    y0 = np.clip(np.floor(vv - r), 0, H - 1).astype(np.int64)
    y1 = np.clip(np.ceil(vv + r), 0, H - 1).astype(np.int64)
    fp, fg, fa, fz = [], [], [], []
    for ty in range(0, H, tile):                                  # tiles only bound memory; sort is global
        for tx in range(0, W, tile):
            sel = np.where((x1 >= tx) & (x0 < tx + tile) & (y1 >= ty) & (y0 < ty + tile))[0]
            if not len(sel):
                continue
            gx0, gx1 = np.maximum(x0[sel], tx), np.minimum(x1[sel], tx + tile - 1)
            gy0, gy1 = np.maximum(y0[sel], ty), np.minimum(y1[sel], ty + tile - 1)
            w_, h_ = gx1 - gx0 + 1, gy1 - gy0 + 1
            n = w_ * h_
            gi = np.repeat(sel, n)
            start = np.repeat(np.cumsum(n) - n, n)
            k = np.arange(n.sum()) - start
            px = np.repeat(gx0, n) + k % np.repeat(w_, n)
            py = np.repeat(gy0, n) + k // np.repeat(w_, n)
            dx, dy = px + 0.0 - uu[gi], py + 0.0 - vv[gi]
            power = -0.5 * (ca[gi] * dx * dx + cc[gi] * dy * dy) - cb[gi] * dx * dy
            al = np.minimum(0.99, op[gi] * np.exp(np.minimum(power, 0)))
            m = (al >= 1 / 255) & (power <= 0)
            fp.append(py[m] * W + px[m]); fg.append(gi[m]); fa.append(al[m].astype(np.float32)); fz.append(zz[gi[m]])
    pix = np.concatenate(fp) if fp else np.zeros(0, np.int64)
    g = np.concatenate(fg) if fg else np.zeros(0, np.int64)
    al = np.concatenate(fa).astype(np.float64) if fa else np.zeros(0)
    dz = np.concatenate(fz) if fz else np.zeros(0)
    o = np.lexsort((dz, pix))
    pix, g, al, dz = pix[o], g[o], al[o], dz[o]
    l1m = np.log1p(-al)
    cs = np.cumsum(l1m)
    seg_start = np.r_[True, pix[1:] != pix[:-1]]
    first_idx = np.maximum.accumulate(np.where(seg_start, np.arange(len(pix)), 0))
    before = cs - l1m - (cs[first_idx] - l1m[first_idx])                           # log T before this fragment
    T = np.exp(before)
    w = al * T
    live = T > t_min
    HW = W * H
    wsum = np.bincount(pix[live], weights=w[live], minlength=HW)
    img = np.empty((HW, 3))
    for ch in range(3):
        img[:, ch] = np.bincount(pix[live], weights=(w * col[g, ch])[live], minlength=HW)
    img += (1 - wsum)[:, None] * (np.array(bg) / 255.0)
    dexp = np.bincount(pix[live], weights=(w * dz)[live], minlength=HW) / np.maximum(wsum, 1e-9)
    dexp[wsum < 1e-6] = np.inf
    after = T * (1 - al)
    cross = live & (T >= 0.5) & (after < 0.5)                                       # median depth fragment
    dmed = np.full(HW, np.inf)
    dmed[pix[cross]] = dz[cross]
    keepw = live & (w >= w_min)
    return SplatView(cam, (np.clip(img, 0, 1) * 255).astype(np.uint8).reshape(H, W, 3),
                     wsum.reshape(H, W).astype(np.float32), dmed.reshape(H, W).astype(np.float32),
                     dexp.reshape(H, W).astype(np.float32), pix[keepw].astype(np.int32),
                     idx[g[keepw]].astype(np.int64), w[keepw].astype(np.float32))
