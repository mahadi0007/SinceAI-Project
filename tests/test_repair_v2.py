"""v2 regressions: clean SAM inputs + embedding reuse, low-coverage whole-group claims, competing and
corroborating candidates (order independence), stale artifact rejection, EWA occlusion weights."""
from __future__ import annotations

import json

import numpy as np

from tests.test_pipeline import CTX, make_base, write_assoc


def test_sam_gets_clean_image_and_reuses_embedding(monkeypatch):
    """SAM must see the untouched render for EVERY prompt (no earlier masks painted on it), and the image
    embedding is computed once per view."""
    import torch
    import pipeline.propose as pp
    seen, calls = [], {"emb": 0}

    class Proc:
        def __call__(self, images=None, **kw):
            seen.append(np.asarray(images).copy())
            out = {"pixel_values": torch.zeros(1, 3, 4, 4), "original_sizes": torch.tensor([[8, 8]])}
            out.update({k: torch.tensor(v, dtype=torch.float32) for k, v in kw.items() if k.startswith("input_")})
            return out

        def post_process_masks(self, m, sizes):
            return [m[0]]

    class Out:
        pred_masks = torch.ones(1, 1, 3, 8, 8)
        iou_scores = torch.tensor([[[0.1, 0.9, 0.2]]])

    class Model:
        def get_image_embeddings(self, pv):
            calls["emb"] += 1
            return [torch.zeros(1)]

        def __call__(self, image_embeddings=None, **kw):
            assert "pixel_values" not in kw and image_embeddings is not None
            return Out()

    monkeypatch.setattr(pp, "sam_model", lambda name: (Proc(), Model()))
    img = np.random.default_rng(0).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    orig = img.copy()
    prompts = [{"box": [0, 0, 4, 4], "pts": [], "lbl": []}, {"box": [2, 2, 7, 7], "pts": [[3, 3]], "lbl": [1]}]
    res = pp.run_sam(img, prompts, "x")
    assert calls["emb"] == 1 and len(res) == 2 and res[0][1] == np.float32(0.9)
    assert all(np.array_equal(s, orig) for s in seen)
    assert np.array_equal(img, orig)


def test_low_coverage_group_is_not_claimed_whole(tmp_path):
    from pipeline.repair import context_edits, leaves_of
    base = make_base([(np.random.default_rng(5).normal(0, 0.03, (100, 3)), "", 3)])
    pos, neg = np.zeros(100, np.float32), np.zeros(100, np.float32)
    pos[:15] = 5                         # only 15% observed, all inside the mask (support 100%)
    write_assoc(tmp_path, pos, neg)
    vg, log, meta = context_edits(base, [[0]], leaves_of(base), {"repair": {"context": CTX}}, tmp_path)
    new = list(meta)[0]
    assert set(np.where(vg == new)[0]) == set(range(15))         # only the observed, supported subset
    e = [x for x in log if x["op"] == "context_repair"][0]
    assert e["splits"][0]["why_not_whole"] == "low coverage" and e["uncertain_unseen"] == 85


def test_competing_claims_are_order_independent_and_ambiguous(tmp_path):
    from pipeline.repair import context_edits, leaves_of
    base = make_base([(np.random.default_rng(6).normal(0, 0.03, (60, 3)), "", 3)])
    pos = np.full(60, 5, np.float32)
    results = []
    for names in (("RV-A", "RV-B"), ("RV-B", "RV-A")):
        d = tmp_path / "-".join(names)
        write_assoc(d, pos, np.zeros(60, np.float32), cls="pump", region=names[0])
        write_assoc(d, pos, np.full(60, 0.5, np.float32), cls="valve", region=names[1])   # 1.0 vs 0.91 support
        vg, log, _ = context_edits(base, [[0]], leaves_of(base), {"repair": {"context": CTX}}, d)
        results.append((vg.copy(), sorted(e["op"] for e in log)))
    assert np.array_equal(results[0][0], results[1][0]) and results[0][1] == results[1][1]
    assert (results[0][0] == 0).all()                            # inside the margin: nobody takes the points


def test_clear_winner_takes_contested_points(tmp_path):
    from pipeline.repair import context_edits, leaves_of
    base = make_base([(np.random.default_rng(9).normal(0, 0.03, (60, 3)), "", 3)])
    write_assoc(tmp_path, np.full(60, 5, np.float32), np.zeros(60, np.float32), cls="pump", region="RV-A")
    write_assoc(tmp_path, np.full(60, 5, np.float32), np.full(60, 5, np.float32), cls="valve", region="RV-B")
    vg, log, meta = context_edits(base, [[0]], leaves_of(base), {"repair": {"context": CTX}}, tmp_path)
    reps = [e for e in log if e["op"] == "context_repair"]
    assert len(reps) == 1 and reps[0]["region"] == "RV-A" and (vg == list(meta)[0]).mean() > 0.9


def test_same_class_claims_are_corroborated_not_competing(tmp_path):
    from pipeline.repair import context_edits, leaves_of
    base = make_base([(np.random.default_rng(7).normal(0, 0.03, (60, 3)), "", 3)])
    write_assoc(tmp_path, np.full(60, 5, np.float32), np.zeros(60, np.float32), cls="valve", region="RV-A")
    write_assoc(tmp_path, np.full(60, 5, np.float32), np.full(60, 0.5, np.float32), cls="control valve",
                region="RV-B")
    vg, log, meta = context_edits(base, [[0]], leaves_of(base), {"repair": {"context": CTX}}, tmp_path)
    reps = [e for e in log if e["op"] == "context_repair"]
    assert len(reps) == 1 and reps[0]["corroborated_by"] == ["RV-B/C1"] and (vg == list(meta)[0]).all()


def test_stale_artifacts_are_rejected(tmp_path, monkeypatch):
    import pipeline.propose as pp
    import pipeline.scene as ps
    monkeypatch.setattr(pp, "PDIR", tmp_path / "prop")
    monkeypatch.setattr(ps, "ROOT", tmp_path)
    views = [{"eye": [0, 0, 0], "target": [0, 0, 1], "fov": 60}]
    monkeypatch.setattr(pp, "regions", lambda: {"RV-T": {"views": views}})
    seg = tmp_path / "seg"
    seg.mkdir()
    np.savez(seg / "segment_points.npz", labels=np.zeros(5, np.int32), label_ids=np.array(["OBJ-0000"]))
    ck = pp.cams_key(pp.region_cameras("RV-T"))
    d = pp.rdir("RV-T", "ewa")
    (d / "render_meta.json").write_text(json.dumps({"render_key": "R1", "cams_key": ck, "cameras": []}))
    (d / "sam_meta.json").write_text(json.dumps({"sam_key": "S1", "proposal_key": "P1"}))
    info = {"region": "RV-T", "renderer": "ewa", "render_key": "R1", "cams_key": ck, "sam_key": "S1",
            "proposal_key": "P1", "seg_rev": ps.seg_revision(seg)}
    assert pp.check_fresh(info, seg) is None
    assert "render" in pp.check_fresh({**info, "render_key": "R0"}, seg)
    assert "SAM" in pp.check_fresh({**info, "sam_key": "S0"}, seg)
    assert "segmentation" in pp.check_fresh({**info, "seg_rev": "old"}, seg)
    monkeypatch.setattr(pp, "regions", lambda: {"RV-T": {"views": [{**views[0], "fov": 50}]}})
    assert "cameras" in pp.check_fresh(info, seg)


def test_stale_association_skipped_by_repair(tmp_path):
    from pipeline.repair import context_edits, leaves_of
    base = make_base([(np.random.default_rng(10).normal(0, 0.03, (60, 3)), "", 3)])
    write_assoc(tmp_path, np.full(60, 5, np.float32), np.zeros(60, np.float32))
    vg, log, meta = context_edits(base, [[0]], leaves_of(base), {"repair": {"context": CTX}}, tmp_path,
                                  freshness=lambda info: "render changed since association")
    assert not meta and log[0]["op"] == "reject_stale" and (vg == 0).all()


def test_ewa_occlusion_weights():
    """Gaussians behind an opaque surface get ~zero contribution weight; front ones carry the pixels."""
    from pipeline.views import Camera, render_ewa
    rng = np.random.default_rng(8)
    front = np.c_[rng.uniform(-0.3, 0.3, 3000), rng.uniform(-0.3, 0.3, 3000), np.full(3000, 1.0)]
    back = np.c_[rng.uniform(-0.1, 0.1, 300), rng.uniform(-0.1, 0.1, 300), np.full(300, 2.0)]
    xyz = np.r_[front, back]
    q = np.tile([1.0, 0, 0, 0], (len(xyz), 1))
    v = render_ewa(Camera.look_at([0, 0, 0], [0, 0, 1], 48, 48, 60), xyz, np.full((len(xyz), 3), 0.5),
                   np.full((len(xyz), 3), 0.02), q, np.full(len(xyz), 0.95))
    wb = v.frag_w[v.frag_gid >= 3000].sum()
    wf = v.frag_w[v.frag_gid < 3000].sum()
    assert wf > 100 * max(wb, 1e-9) and v.alpha[20:28, 20:28].min() > 0.9


def test_ewa_anisotropy_follows_quaternion():
    """An elongated Gaussian rotated 90 deg about the view axis must turn its footprint by 90 deg."""
    from pipeline.views import Camera, render_ewa
    cam = Camera.look_at([0, 0, 0], [0, 0, 1], 64, 64, 60)
    s = np.array([[0.2, 0.01, 0.01]])
    ext = []
    for q in ([1.0, 0, 0, 0], [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]):
        v = render_ewa(cam, np.array([[0, 0, 1.0]]), np.full((1, 3), 0.5), s, np.array([q]), np.array([0.99]))
        ys, xs = np.nonzero(v.alpha > 0.3)
        ext.append((np.ptp(xs), np.ptp(ys)))
    assert ext[0][0] > 3 * ext[0][1] and ext[1][1] > 3 * ext[1][0]
