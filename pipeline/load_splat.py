"""Load the Valmet 'Industrial Boiler Room.ply' Gaussian splat safely.

Findings (6 Nov 2026):
- Header says 1,506,136 Gaussians, but only the first ~502,483 rows are valid.
  After that the file contains random bytes (NaN, +-3e38 positions). They must be
  dropped or Open3D / DBSCAN / viewers will break.
- Up axis is -Y (floor plane at y ~ +1.075, ceiling at y ~ -0.889).
  In Viser use: server.scene.set_up_direction("-y") and keep native coordinates.
- Quaternion fields are stored as rot_1, rot_2, rot_3, rot_0 (rot_0 = w).
"""
from __future__ import annotations

import numpy as np

SH_C0 = 0.28209479177387814


def _read_header(path: str):
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError("No end_header found")
            header += line
    lines = header.split(b"\n")
    n = int([l for l in lines if l.startswith(b"element vertex")][0].split()[-1])
    names = [l.split()[-1].decode() for l in lines if l.startswith(b"property")]
    return len(header), n, names


def _valid_prefix(xyz: np.ndarray, bound: float = 100.0, run: int = 50) -> int:
    """Index of the first row of a run of `run` invalid rows (corruption start)."""
    ok = np.isfinite(xyz).all(1) & (np.abs(xyz) < bound).all(1)
    bad = (~ok).astype(np.int32)
    if bad.sum() == 0:
        return len(xyz)
    window = np.convolve(bad, np.ones(run, dtype=np.int32), "valid")
    hits = np.where(window >= int(run * 0.8))[0]
    return int(hits[0]) if len(hits) else len(xyz)


def load_splat(path: str, min_opacity: float = 0.1, max_scale: float = 0.3,
               crop_floaters: bool = True) -> dict:
    """Return clean arrays in native file coordinates.

    keys: xyz (N,3), rgb (N,3 in 0..1), opacity (N,), scales (N,3, linear),
          quats_wxyz (N,4, normalised), row_ids (N,) source PLY rows, n_file, n_valid_prefix
    """
    offset, n, names = _read_header(path)
    raw = np.memmap(path, dtype=np.float32, mode="r", offset=offset,
                    shape=(n, len(names)))
    col = {k: i for i, k in enumerate(names)}
    def get(k):
        with np.errstate(all="ignore"):
            return np.asarray(raw[:, col[k]], dtype=np.float64)

    xyz = np.stack([get("x"), get("y"), get("z")], 1)
    cut = _valid_prefix(xyz)
    sl = slice(0, cut)

    xyz = xyz[sl]
    logit = np.clip(get("opacity")[sl], -30, 30)
    opacity = 1.0 / (1.0 + np.exp(-logit))
    log_s = np.clip(np.stack([get(f"scale_{i}")[sl] for i in range(3)], 1), -20, 5)
    scales = np.exp(log_s)
    q = np.stack([get("rot_0")[sl], get("rot_1")[sl], get("rot_2")[sl], get("rot_3")[sl]], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    rgb = np.clip(0.5 + SH_C0 * np.stack([get(f"f_dc_{i}")[sl] for i in range(3)], 1), 0, 1)

    keep = (np.isfinite(xyz).all(1) & np.isfinite(opacity) & np.isfinite(scales).all(1)
            & (opacity > min_opacity) & (scales.max(1) < max_scale))
    if crop_floaters:  # robust box around the scene; drops capture-side floaters
        lo, hi = np.percentile(xyz[keep], 1, 0), np.percentile(xyz[keep], 99, 0)
        pad = 0.05 * (hi - lo)
        keep &= ((xyz > lo - pad) & (xyz < hi + pad)).all(1)

    return {
        "xyz": xyz[keep].astype(np.float32),
        "rgb": rgb[keep].astype(np.float32),
        "opacity": opacity[keep].astype(np.float32),
        "scales": scales[keep].astype(np.float32),
        "quats_wxyz": q[keep].astype(np.float32),
        "row_ids": np.where(keep)[0].astype(np.int64),   # original PLY row of every kept Gaussian
        "n_file": n,
        "n_valid_prefix": cut,
    }


def covariances(scales: np.ndarray, quats_wxyz: np.ndarray) -> np.ndarray:
    """(N,3,3) covariance matrices, e.g. for viser add_gaussian_splats."""
    w, x, y, z = quats_wxyz.T
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], 1).reshape(-1, 3, 3)
    S = R * scales[:, None, :]
    return S @ np.transpose(S, (0, 2, 1))


if __name__ == "__main__":
    import sys
    d = load_splat(sys.argv[1])
    print(f"file rows: {d['n_file']:,}  valid prefix: {d['n_valid_prefix']:,}  "
          f"kept after filters: {len(d['xyz']):,}")
    print("bounds min", d["xyz"].min(0).round(2), "max", d["xyz"].max(0).round(2))
