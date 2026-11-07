"""Regression tests (synthetic data, no network): provider limits, validation, label-source gates,
pipe continuity, visibility/occlusion, mask-supported split/merge, source membership.
Run: .venv/Scripts/python.exe -m pytest -q"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pipeline.common import is_vision_label


# ---------- provider: budget, offline, no retry ----------

def test_budget_and_offline(tmp_path, monkeypatch):
    import copilot.llm as llm
    monkeypatch.setattr(llm, "ROOT", tmp_path)
    monkeypatch.delenv("VALMET_OFFLINE", raising=False)
    cc = {"model": "m", "api_key_env": "NOPE_KEY", "base_url": "http://127.0.0.1:9", "temperature": 0,
          "budget": {"max_paid_requests": 0, "ledger": "ledger.jsonl"}}
    monkeypatch.setenv("NOPE_KEY", "x")
    with pytest.raises(llm.BudgetExceeded):
        llm.generate("hi", cc)
    assert llm.usage_summary(cc)["paid_requests"] == 0
    monkeypatch.setenv("VALMET_OFFLINE", "1")
    with pytest.raises(llm.LLMUnavailable):
        llm.generate("hi", {**cc, "budget": {"max_paid_requests": 5, "ledger": "ledger.jsonl"}})


def test_failed_request_is_counted_and_not_retried(tmp_path, monkeypatch):
    import copilot.llm as llm
    monkeypatch.setattr(llm, "ROOT", tmp_path)
    monkeypatch.delenv("VALMET_OFFLINE", raising=False)
    monkeypatch.setenv("NOPE_KEY", "x")
    cc = {"model": "m", "api_key_env": "NOPE_KEY", "base_url": "http://127.0.0.1:9/v1", "temperature": 0,
          "timeout_s": 2, "budget": {"max_paid_requests": 1, "ledger": "ledger.jsonl"}}
    with pytest.raises(Exception):
        llm.generate("hi", cc)                       # connection refused; must be logged once
    rows = llm.ledger(cc)
    assert len(rows) == 1 and rows[0]["status"] == "error"
    with pytest.raises(llm.BudgetExceeded):          # the failed attempt consumed the budget
        llm.generate("hi", cc)


# ---------- validation ----------

def test_label_validation():
    from pipeline.label import validate_labels
    vocab = ["pump", "pipe segment", "other"]
    res = [{"id": "OBJ-0001", "label": "pump", "confidence": 0.9},
           {"id": "OBJ-0001", "label": "pump", "confidence": 0.8},       # duplicate
           {"id": "OBJ-0099", "label": "pump", "confidence": 0.9},       # not in batch
           {"id": "OBJ-0002", "label": "spaceship", "confidence": 0.9},  # not in vocab
           {"id": "OBJ-0003", "label": "pump", "confidence": float("nan")},
           {"id": "OBJ-0004", "label": "pump", "confidence": 1.5}]
    ok, bad = validate_labels(res, {"OBJ-0001", "OBJ-0002", "OBJ-0003", "OBJ-0004"}, vocab)
    assert [r["id"] for r in ok] == ["OBJ-0001"] and len(bad) == 5


def test_proposal_validation():
    from pipeline.propose import validate
    ok, bad = validate({"components": [
        {"cid": "C1", "class": "pump", "confidence": 0.8, "box_a": [10, 10, 200, 200], "point_a": [50, 50]},
        {"cid": "C2", "class": "control_valve", "confidence": 0.8, "box_a": [10, 10, 200, 200]},
        {"cid": "C3", "class": "valve", "confidence": 0.8, "box_a": [300, 10, 200, 200]},     # x1 < x0
        {"cid": "C4", "class": "valve", "confidence": 2, "box_a": [10, 10, 20, 20]}]})
    assert [c["cid"] for c in ok] == ["C1"] and len(bad) == 3


def test_vision_label_gate_is_provider_neutral():
    cfg = {"label": {"vision_sources": ["featherless_vision", "gemini_vision"]}}
    assert is_vision_label({"label_source": "featherless_vision", "label": "pump", "label_confidence": 0.9}, cfg, ["pump"], 0.8)
    assert is_vision_label({"label_source": "gemini_vision", "label": "pump", "label_confidence": 0.9}, cfg, ["pump"], 0.8)
    assert not is_vision_label({"label_source": "geometry_rules", "label": "pump", "label_confidence": 0.9}, cfg, ["pump"])
    assert not is_vision_label({"label_source": "featherless_vision", "label": "pump", "label_confidence": float("nan")}, cfg)
    assert not is_vision_label({"label_source": "featherless_vision", "label": "pump", "label_confidence": 0.5}, cfg, ["pump"], 0.8)


# ---------- synthetic scene helpers ----------

def cylinder(p0, p1, r, n=300, rng=None):
    rng = rng or np.random.default_rng(0)
    p0, p1 = np.array(p0, float), np.array(p1, float)
    ax = (p1 - p0) / np.linalg.norm(p1 - p0)
    u = np.cross(ax, [1, 0, 0] if abs(ax[0]) < 0.9 else [0, 1, 0])
    u /= np.linalg.norm(u)
    v = np.cross(ax, u)
    t = rng.uniform(0, 1, n)
    a = rng.uniform(0, np.pi, n)                       # half surface: partial scan
    return p0 + np.outer(t, p1 - p0) + r * (np.outer(np.cos(a), u) + np.outer(np.sin(a), v))


def make_base(parts):
    """parts: list of (points, hint, axis). One leaf per part, one baseline object per leaf."""
    xyz = np.concatenate([p for p, _, _ in parts])
    frag = np.concatenate([np.full(len(p), i) for i, (p, _, _) in enumerate(parts)])
    assets = [{"id": f"OBJ-{i:04d}", "kind": "object", "fragments": [i], "tags": [], "hint": h}
              for i, (_, h, _) in enumerate(parts)]
    return {"xyz": xyz.astype(np.float32), "frag": frag.astype(np.int32), "labels": frag.astype(np.int32),
            "frag_hint": np.array([h for _, h, _ in parts]), "frag_axis": np.array([a for _, _, a in parts], np.int8),
            "rgb": np.full((len(xyz), 3), 0.5, np.float32), "opacity": np.ones(len(xyz), np.float32),
            "normals": np.zeros((len(xyz), 3), np.float32), "src_rows": np.arange(len(xyz)),
            "src_vox": np.arange(len(xyz)), "src_dist": np.zeros(len(xyz), np.float32), "assets": assets}


GC = {"min_points": 30, "max_angle_deg": 8, "offset_frac": 0.9, "offset_abs": 0.015, "max_thickness_ratio": 1.6,
      "max_gap": 0.15, "barrier_points": 12, "barrier_max_diag": 0.6, "run_fit_frac": 1.3}


def groups_of(base):
    from pipeline.repair import geo_group, leaves_of
    groups, _ = geo_group(base, leaves_of(base), GC)
    return [sorted(g) for g in groups]


def test_collinear_gap_joined_parallel_kept_apart():
    base = make_base([(cylinder([0, 0, 0], [0, 0.5, 0], 0.05), "pipe segment", 1),
                      (cylinder([0, 0.58, 0], [0, 1.0, 0], 0.05), "pipe segment", 1),      # same pipe after gap
                      (cylinder([0.14, 0, 0], [0.14, 1.0, 0], 0.05), "pipe segment", 1)])  # parallel neighbour
    g = groups_of(base)
    assert [0, 1] in g and [2] in g


def test_valve_barrier_and_bend_not_crossed():
    rng = np.random.default_rng(1)
    valve = rng.normal([0, 0.55, 0], 0.03, (80, 3))                                     # compact blob in the gap
    base = make_base([(cylinder([0, 0, 0], [0, 0.5, 0], 0.05), "pipe segment", 1),
                      (cylinder([0, 0.62, 0], [0, 1.0, 0], 0.05), "pipe segment", 1),
                      (valve, "fitting/valve", 3),
                      (cylinder([0.06, 1.05, 0], [0.6, 1.05, 0], 0.05), "pipe segment", 0)])  # elbow to x
    g = groups_of(base)
    assert [0] in g and [1] in g and [3] in g


def test_thickness_mismatch_not_joined():
    base = make_base([(cylinder([0, 0, 0], [0, 0.5, 0], 0.05), "pipe segment", 1),
                      (cylinder([0, 0.55, 0], [0, 1.0, 0], 0.15, 600), "pipe segment", 1)])
    assert [0] in groups_of(base) and [1] in groups_of(base)


# ---------- visibility ----------

def test_render_occlusion_and_footprint():
    from pipeline.views import Camera, render
    rng = np.random.default_rng(0)
    wall = np.c_[rng.uniform(-0.3, 0.3, 4000), rng.uniform(-0.3, 0.3, 4000), np.full(4000, 1.0)]
    behind = np.c_[rng.uniform(-0.1, 0.1, 500), rng.uniform(-0.1, 0.1, 500), np.full(500, 2.0)]
    xyz = np.r_[wall, behind]
    cam = Camera.look_at([0, 0, 0], [0, 0, 1], 64, 64, 60)
    v = render(cam, xyz, np.full((len(xyz), 3), 0.5), np.full((len(xyz), 3), 0.02), np.ones(len(xyz)))
    ids = v.gid[v.gid >= 0]
    assert len(ids) and (ids < len(wall)).all()                  # nothing behind the wall is visible
    one = render(cam, np.array([[0, 0, 1.0]]), np.full((1, 3), 0.5), np.full((1, 3), 0.05), np.ones(1))
    assert (one.gid == 0).sum() > 1                               # footprint covers more than its centre pixel


# ---------- context: mask-supported split / merge ----------

CTX = {"min_in_px": 5, "member_frac": 0.6, "min_split_voxels": 5, "min_candidate_voxels": 10,
       "cc_eps": 0.05, "cc_min_frac": 0.2, "max_extent": 1.2, "pipe_protect_min_len": 0.4, "disabled": [],
       "seen_eps": 0.02, "min_coverage": 0.5, "spatial_frac": 0.8, "spatial_dist": 0.05, "claim_margin": 0.2,
       "provisional_frac": 0.1, "corroborate_overlap": 0.3}


def write_assoc(d, pos, neg, cls="pump", region="RV-T"):
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{region}_assoc.npz", pos_0=pos, neg_0=neg)
    (d / f"{region}_assoc.json").write_text(json.dumps({"region": region, "model": "m", "candidates": [
        {"cid": "C1", "cls": cls, "confidence": 0.8, "usable": True}]}), encoding="utf-8")


def test_mixed_cluster_split_and_member_merge(tmp_path):
    from pipeline.repair import context_edits, leaves_of
    rng = np.random.default_rng(2)
    pump_part = rng.normal([0, 0, 0], 0.03, (60, 3))
    wall_part = rng.normal([0.6, 0, 0], 0.05, (200, 3))
    fragment = rng.normal([0.05, 0, 0], 0.02, (40, 3))
    long_pipe = cylinder([-0.2, -0.5, 0.05], [-0.2, 0.5, 0.05], 0.03, 200)
    base = make_base([(np.r_[pump_part, wall_part], "", 3), (fragment, "", 3), (long_pipe, "pipe segment", 1)])
    n = len(base["xyz"])
    pos, neg = np.zeros(n, np.float32), np.zeros(n, np.float32)
    pos[:60] = 3                    # pump half of the mixed cluster: inside the mask
    neg[60:260] = 3                 # wall half: visible outside the mask
    pos[260:300] = 3                # separate pump fragment: inside
    pos[300:] = 1                   # pipe run passing through the mask
    write_assoc(tmp_path, pos, neg)
    leaves = leaves_of(base)
    groups = [[0], [1], [2]]
    vg, log, meta = context_edits(base, groups, leaves, {"repair": {"context": CTX}}, tmp_path)
    new = [k for k in meta][0]
    moved = np.where(vg == new)[0]
    assert set(moved) == set(range(60)) | set(range(260, 300))          # pump part + fragment only
    assert (vg[60:260] == 0).all() and (vg[300:] == 2).all()             # wall stays, pipe protected
    e = [x for x in log if x["op"] == "context_repair"][0]
    assert e["members"][0]["group"] == 1 and e["splits"][0]["group"] == 0


def test_disabled_repair_is_reverted(tmp_path):
    from pipeline.repair import context_edits, leaves_of
    rng = np.random.default_rng(3)
    base = make_base([(rng.normal(0, 0.03, (50, 3)), "", 3), (rng.normal([0.05, 0, 0], 0.03, (50, 3)), "", 3)])
    write_assoc(tmp_path, np.full(100, 3, np.float32), np.zeros(100, np.float32))
    ctx = {**CTX, "disabled": ["RV-T/C1"]}
    vg, log, meta = context_edits(base, [[0], [1]], leaves_of(base), {"repair": {"context": ctx}}, tmp_path)
    assert not meta and (vg[:50] == 0).all() and (vg[50:] == 1).all() and log[0]["op"] == "disabled"


def test_build_preserves_points_and_provenance(tmp_path, monkeypatch):
    import pipeline.repair as rp
    monkeypatch.setattr(rp, "ROOT", tmp_path)
    rng = np.random.default_rng(4)
    base = make_base([(rng.normal(0, 0.03, (50, 3)), "", 3), (rng.normal([0.05, 0, 0], 0.03, (50, 3)), "", 3)])
    vg = np.full(100, 2, np.int64)                                     # both leaves -> candidate label 2
    cfg = {"repair": {"min_object_points": 5}, "scene": {"floor_y": 1.0}, "segment": {"pipe_linearity": 0.9}}
    rp.build(base, vg, [[0], [1]], {2: {"cid": "C1", "region": "RV-T", "cls": "pump", "confidence": 0.8,
                                         "model": "m", "assembly": None}}, cfg, "o", [], "test")
    d = np.load(tmp_path / "o" / "segment_points.npz")
    a = json.loads((tmp_path / "o" / "assets.json").read_text())
    assert (d["labels_baseline"] == base["labels"]).all() and (d["frag"] == base["frag"]).all()
    assert len(a) == 1 and a[0]["fragments"] == [0, 1] and a[0]["baseline_ids"] == ["OBJ-0000", "OBJ-0001"]
    assert a[0]["vlm_proposal"]["label"] == "pump" and (d["labels"] == 0).all()
