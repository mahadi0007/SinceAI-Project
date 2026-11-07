"""Rendering comparison sheets, same calibrated camera: rows = clean RGB | depth (disc z-buffer vs EWA median)
| visibility (disc coverage vs EWA accumulated alpha) | fragment overlay; columns = disc | EWA.
Output: out_review/render_compare/<rid>_v<k>.png   Run: python -m pipeline.render_compare [RV-01 ...]"""
from __future__ import annotations

import sys

from PIL import Image, ImageDraw, ImageFont

from pipeline.common import ROOT
from pipeline.propose import rdir, regions, render_meta


def sheet(rid: str, k: int) -> Image.Image:
    rows = [("clean RGB (SAM/VLM input)", "view{k}.png"), ("depth", "view{k}_depth.png"),
            ("visibility / alpha", "view{k}_alpha.png"), ("fragment overlay", "view{k}_ids.png")]
    ims = [[Image.open(rdir(rid, r) / f.format(k=k)).convert("RGB") for r in ("disc", "ewa")] for _, f in rows]
    w, h = ims[0][0].size
    pad, head = 6, 24
    out = Image.new("RGB", (2 * w + pad, len(rows) * (h + head)), (255, 255, 255))
    dr = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for i, ((title, _), pair) in enumerate(zip(rows, ims)):
        y = i * (h + head)
        for j, (im, r) in enumerate(zip(pair, ("disc (baseline)", "EWA Gaussians"))):
            dr.text((j * (w + pad) + 4, y + 4), f"{rid} v{k} - {r} - {title}", fill=(0, 0, 0), font=font)
            out.paste(im, (j * (w + pad), y + head))
    return out


def main(rids: list[str]) -> None:
    out = ROOT / "out_review" / "render_compare"
    out.mkdir(parents=True, exist_ok=True)
    for rid in rids:
        n = len(render_meta(rid, "ewa")["cameras"])
        for k in range(n):
            sheet(rid, k).save(out / f"{rid}_v{k}.png")
            print(f"[render_compare] {out / f'{rid}_v{k}.png'}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(regions()))
