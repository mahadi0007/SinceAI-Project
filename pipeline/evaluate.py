"""Compare segmentation versions on the reviewed reference components (review/regions.yaml).

Per reference component (voxel points inside its reviewed box, structure planes excluded):
  completeness  = share of the reference's points held by the single best object
  n_pieces      = objects holding >= 10% of the reference's points (fragmentation)
  leakage       = share of that best object's points lying OUTSIDE the reference box (+0.05 margin)
                  (swallowed background / neighbours; under-segmentation)
  iou           = |best object AND reference| / |best object OR reference| (points) - completeness and
                  swallowing in one number
  merged_with   = other references for which the same object also holds >= 20% (accidental merge)
Fewer boxes alone is never counted as success.
Run: python -m pipeline.evaluate out_baseline out_geo out
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pipeline.common import ROOT


def refs_hash() -> str:
    import hashlib
    reg = yaml.safe_load((ROOT / "review" / "regions.yaml").read_text(encoding="utf-8"))["regions"]
    return hashlib.sha256(json.dumps({g["id"]: g["refs"] for g in reg}, sort_keys=True).encode()).hexdigest()[:16]


def load_refs(check_frozen: bool = True) -> list[dict]:
    """Frozen v2 references; refuses to evaluate if they changed since freezing (review/refs_v2.sha256)."""
    if check_frozen:
        want = (ROOT / "review" / "refs_v2.sha256").read_text().strip()
        if refs_hash() != want:
            raise SystemExit(f"reference boxes changed since freezing ({refs_hash()} != {want}); refusing to evaluate")
    reg = yaml.safe_load((ROOT / "review" / "regions.yaml").read_text(encoding="utf-8"))["regions"]
    return [{**r, "region": g["id"], "holdout": bool(g.get("holdout"))} for g in reg for r in g["refs"]]


def in_box(xyz: np.ndarray, box: Any, margin: float = 0.0) -> np.ndarray:
    lo, hi = np.array(box[0]) - margin, np.array(box[1]) + margin
    return ((xyz >= lo) & (xyz <= hi)).all(1)


def evaluate(seg_dir: str | Path) -> dict:
    d = np.load(ROOT / seg_dir / "segment_points.npz")
    assets = json.loads((ROOT / seg_dir / "assets.json").read_text(encoding="utf-8"))
    xyz, lab = d["xyz"].astype(np.float64), d["labels"]
    kind = np.array([a["kind"] == "object" for a in assets])
    refs = load_refs()
    best_of: dict[str, int] = {}
    rows = []
    for r in refs:
        m = in_box(xyz, r["box"]) & (lab != -2)
        m &= ~((lab >= 0) & ~kind[np.maximum(lab, 0)])            # drop structure planes
        L = lab[m]
        n = int(m.sum())
        obj = L[L >= 0]
        ids, cnt = np.unique(obj, return_counts=True)
        if n == 0 or len(ids) == 0:
            rows.append({"ref": r["id"], "cls": r["cls"], "n": n, "completeness": 0.0, "n_pieces": 0})
            continue
        b = int(ids[cnt.argmax()])
        best_of[r["id"]] = b
        ob = lab == b
        leak = float((ob & ~in_box(xyz, r["box"], 0.05)).sum() / ob.sum())
        inter = float(cnt.max())
        iou = inter / (ob.sum() + n - inter)                           # points: best object vs reference
        rows.append({"ref": r["id"], "cls": r["cls"], "region": r["region"], "holdout": r["holdout"], "n": n,
                     "completeness": round(float(cnt.max() / n), 3),
                     "unassigned": round(float((L < 0).sum() / n), 3),
                     "iou": round(float(iou), 3),
                     "n_pieces": int((cnt >= 0.10 * n).sum()), "best_object": assets[b]["id"],
                     "best_object_points": int(ob.sum()), "leakage": round(leak, 3)})
    for row in rows:
        b = best_of.get(row["ref"])
        if b is None:
            continue
        others = []
        for r in refs:
            if r["id"] == row["ref"]:
                continue
            m = in_box(xyz, r["box"]) & (lab >= 0)
            if m.sum() and (lab[m] == b).sum() >= 0.2 * m.sum():
                others.append(r["id"])
        row["merged_with"] = others
    n_obj = int(kind.sum())
    ok = [r for r in rows if r.get("n")]
    tune = [r for r in ok if not r.get("holdout")]
    hold = [r for r in ok if r.get("holdout")]
    summ = {"seg_dir": str(seg_dir), "objects": n_obj, "refs_hash": refs_hash(),
            "tuning_iou": round(float(np.mean([r.get("iou", 0) for r in tune])), 3) if tune else None,
            "holdout_iou": round(float(np.mean([r.get("iou", 0) for r in hold])), 3) if hold else None,
            "mean_iou": round(float(np.mean([r.get("iou", 0) for r in ok])), 3),
            "mean_completeness": round(float(np.mean([r["completeness"] for r in ok])), 3),
            "mean_leakage": round(float(np.mean([r.get("leakage", 0) for r in ok])), 3),
            "fragmented_refs(n_pieces>1)": sum(r["n_pieces"] > 1 for r in ok),
            "accidental_merges": sum(bool(r.get("merged_with")) for r in ok),
            "rows": rows}
    return summ


def table(results: list[dict]) -> str:
    out = []
    for res in results:
        out.append(f"\n== {res['seg_dir']}: {res['objects']} objects | IoU tuning {res['tuning_iou']} "
                   f"hold-out {res['holdout_iou']} | mean IoU {res['mean_iou']} "
                   f"| mean completeness {res['mean_completeness']} "
                   f"| mean leakage {res['mean_leakage']} | fragmented refs {res['fragmented_refs(n_pieces>1)']} "
                   f"| accidental merges {res['accidental_merges']}")
        for r in res["rows"]:
            out.append(f"  {'H' if r.get('holdout') else ' '} {r['ref']} {r['cls']:9s} n={r['n']:5d} IoU={r.get('iou', 0):.2f} compl={r['completeness']:.2f} "
                       f"pieces={r['n_pieces']} leak={r.get('leakage', 0):.2f} best={r.get('best_object')}"
                       f"({r.get('best_object_points')}) merged_with={r.get('merged_with')}")
    return "\n".join(out)


if __name__ == "__main__":
    dirs = sys.argv[1:] or ["out_baseline"]
    res = [evaluate(d) for d in dirs]
    print(table(res))
    (ROOT / "out_review").mkdir(exist_ok=True)
    (ROOT / "out_review" / "evaluation.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
