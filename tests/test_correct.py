"""Guided boundary correction: prompt coordinates/modes, clean SAM input + cached embedding, unknown-class
corrections, stale artifacts, low-coverage support, second-view disagreement, stale inference responses,
persistence, apply and undo (also after restart)."""
from __future__ import annotations

import hashlib
import json
import types

import numpy as np
import pytest
from PIL import Image

from pipeline import correct as cr

RID = "RV-T"
VIEWS = [{"eye": [0, 0, 0], "target": [0, 0, 2], "fov": 60}, {"eye": [0.8, 0, 0], "target": [0, 0, 2], "fov": 60}]
RC = {"seen_eps": 0.02, "min_in_px": 5, "member_frac": 0.6, "min_coverage": 0.5, "spatial_frac": 0.8,
      "spatial_dist": 0.05, "min_split_voxels": 10, "min_candidate_voxels": 25, "cc_eps": 0.05, "cc_min_frac": 0.2,
      "pipe_protect_min_len": 0.4, "max_extent": 1.2}
PC = {"sam_model": "fake-sam", "depth_gate_min": 0.15, "depth_gate_size": 1.2}


def grid(x0, x1, y0, y1, z, step=0.01):
    xs, ys = np.meshgrid(np.arange(x0, x1 + 1e-9, step), np.arange(y0, y1 + 1e-9, step))
    return np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, z)]


def make_points():
    """A = compact equipment (label 0), B = big cluster with a visible front and an UNSEEN back (label 1),
    W = wall structure behind (label 2)."""
    A = grid(-0.1, 0.1, -0.1, 0.1, 2.0)
    Bf = grid(0.15, 0.6, -0.1, 0.1, 2.0)
    Bb = np.r_[grid(0.15, 0.6, -0.1, 0.1, 2.05), grid(0.15, 0.6, -0.1, 0.1, 2.4)]   # hidden behind Bf (both views)
    Wl = grid(-1.0, 1.0, -0.6, 0.6, 2.8, 0.03)
    xyz = np.r_[A, Bf, Bb, Wl]
    lab = np.r_[np.zeros(len(A)), np.ones(len(Bf) + len(Bb)), np.full(len(Wl), 2)].astype(np.int32)
    hidden = np.r_[np.zeros(len(A) + len(Bf), bool), np.ones(len(Bb), bool), np.zeros(len(Wl), bool)]
    return xyz, lab, hidden, len(A), len(Bf)


def buffers(cam, xyz, hidden):
    """Minimal renderer: each visible point owns its pixel (nearest wins), weight 1, depth = z."""
    u, v, z = cam.project(xyz)
    ok = ~hidden & (z > 0) & (u >= 0) & (u < cam.W) & (v >= 0) & (v < cam.H)
    i = np.where(ok)[0]
    pix = np.round(v[i]).astype(np.int64) * cam.W + np.round(u[i]).astype(np.int64)
    o = np.lexsort((z[i], pix))
    pix, i = pix[o], i[o]
    first = np.r_[True, pix[1:] != pix[:-1]]
    pix, i = pix[first], i[first]
    depth = np.full(cam.W * cam.H, np.inf, np.float32)
    depth[pix] = z[i]
    return {"depth": depth.reshape(cam.H, cam.W), "depth_exp": depth.reshape(cam.H, cam.W),
            "alpha": np.isfinite(depth).reshape(cam.H, cam.W).astype(np.float32), "frag_pix": pix.astype(np.int32),
            "frag_gid": i.astype(np.int64), "frag_w": np.ones(len(i), np.float32),
            "top_id": np.full((cam.H, cam.W), -1, np.int64)}


def mask_of(cam, P, r=4):
    u, v, z = cam.project(P)
    m = np.zeros((cam.H, cam.W), bool)
    for x, y in zip(np.round(u).astype(int), np.round(v).astype(int)):
        m[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True
    return m


@pytest.fixture
def world(tmp_path, monkeypatch):
    import pipeline.propose as pp
    from pipeline.views import Camera
    monkeypatch.setattr(pp, "PDIR", tmp_path / "prop")
    monkeypatch.setattr(pp, "regions", lambda: {RID: {"name": "test region", "views": VIEWS}})
    xyz, lab, hidden, nA, nBf = make_points()
    cams = pp.region_cameras(RID)
    d = pp.rdir(RID, "ewa")
    for k, cam in enumerate(cams):
        b = buffers(cam, xyz, hidden)
        np.savez_compressed(d / f"view{k}_buffers.npz", **b)
        img = np.where(np.isfinite(b["depth"])[..., None], 90, 235).astype(np.uint8).repeat(3, 2)
        Image.fromarray(img).save(d / f"view{k}.png")
    (d / "render_meta.json").write_text(json.dumps({"render_key": "R1", "cams_key": pp.cams_key(cams),
                                                     "cameras": [c.to_json() for c in cams]}), encoding="utf-8")
    base = tmp_path / "base"
    base.mkdir()
    ids = ["OBJ-0000", "OBJ-0001", "OBJ-0002"]
    np.savez_compressed(base / "segment_points.npz", xyz=xyz.astype(np.float32), rgb=np.full((len(xyz), 3), .5, np.float32),
                        opacity=np.ones(len(xyz), np.float32), labels=lab, label_ids=np.array(ids),
                        frag=lab.copy(), src_rows=np.arange(len(xyz)), src_vox=np.arange(len(xyz)),
                        src_dist=np.zeros(len(xyz), np.float32))
    assets = [{"id": ids[0], "kind": "object", "hint": None, "tags": [], "n_points": int((lab == 0).sum()),
               "extent": [0.2, 0.2, 0.0], "label": "pump", "label_source": "featherless_vision"},
              {"id": ids[1], "kind": "object", "hint": None, "tags": [], "n_points": int((lab == 1).sum()),
               "extent": [0.45, 0.2, 0.4]},
              {"id": ids[2], "kind": "structure", "hint": "wall", "tags": [], "n_points": int((lab == 2).sum()),
               "extent": [2, 1.2, 0]}]
    (base / "assets.json").write_text(json.dumps(assets), encoding="utf-8")
    cfg = {"repair": {"context": RC}, "propose": PC, "scene": {"floor_y": 1.0}, "segment": {"pipe_linearity": 0.99},
           "correct": {"base_seg": str(base), "out_seg": str(tmp_path / "out_manual"),
                       "store": str(tmp_path / "store"), "renderer": "ewa"}}
    sc = types.SimpleNamespace(xyz=xyz, g_vox=np.arange(len(xyz)), vox_xyz=xyz, vox_label=lab, assets=assets,
                               seg_dir=base)

    def engine():
        from pipeline.scene import seg_revision
        e = cr.Engine(cfg)
        e._scene, e.seg_rev = sc, seg_revision(base)
        e.is_structure = np.array([a["kind"] == "structure" for a in assets])
        e.protected = e.is_structure.copy()
        return e

    calls = {"emb": 0, "seen": []}

    def fake_emb(image, name, cache=True):
        calls["emb"] += 1
        calls["seen"].append(np.array(image, copy=True))
        return "emb", {"original_sizes": [[image.shape[0], image.shape[1]]]}, Image.fromarray(np.array(image))

    def fake_predict(emb, base_, img, prompt, name):
        calls.setdefault("prompts", []).append(prompt)
        W, H = img.size
        m = np.zeros((H, W), bool)
        if prompt.get("box"):
            x0, y0, x1, y1 = (int(round(t)) for t in prompt["box"])
            m[y0:y1 + 1, x0:x1 + 1] = True
        for (x, y), l in zip(prompt["pts"], prompt["lbl"]):
            yy, xx = np.ogrid[:H, :W]
            disc = (xx - x) ** 2 + (yy - y) ** 2 <= 15 ** 2
            m = m | disc if l == 1 else m & ~disc
        return m, 0.9

    monkeypatch.setattr(pp, "sam_embedding", fake_emb)
    monkeypatch.setattr(pp, "sam_predict", fake_predict)
    return types.SimpleNamespace(cfg=cfg, cams=cams, xyz=xyz, lab=lab, nA=nA, nBf=nBf, engine=engine, calls=calls,
                                 tmp=tmp_path, base=base, pdir=d)


def service(w, sink=None):
    from api.annotate import Service
    return Service(w.cfg, sink=sink, store=cr.Store(w.cfg), engine=w.engine())


def box_px(cam, P, pad=6):
    u, v, _ = cam.project(P)
    return [float(u.min() - pad), float(v.min() - pad), float(u.max() + pad), float(v.max() + pad)]


# ---------- prompts ----------

def test_prompt_coordinates_are_validated_in_original_pixels():
    p = cr.clean_prompt([300.4, 200, 100, 50.2], [[10, 20]], [[700, 500]], 768, 576)
    assert p["box"] == [100.0, 50.2, 300.4, 200.0]                       # corners ordered
    assert cr.clean_prompt([-20, -5, 900, 700], None, None, 768, 576)["box"] == [0.0, 0.0, 767.0, 575.0]
    with pytest.raises(ValueError):
        cr.clean_prompt(None, [[800, 10]], [], 768, 576)                  # displayed-size coords beyond the image
    with pytest.raises(ValueError):
        cr.clean_prompt(None, [], [[10, 10]], 768, 576)                   # exclude-only is not a prompt
    with pytest.raises(ValueError):
        cr.clean_prompt([10, 10, 11, 11], None, None, 768, 576)           # degenerate box


def test_display_scaling_maps_to_original_pixels():
    """Same arithmetic as the page's toImage(): (client - rect.left) * natural / displayed."""
    def to_image(cx, cy, left, top, disp_w, disp_h, nat_w=768, nat_h=576):
        return (cx - left) * nat_w / disp_w, (cy - top) * nat_h / disp_h
    x, y = to_image(10 + 192, 40 + 144, 10, 40, 384, 288)               # half-size display, centre click
    assert (x, y) == (384.0, 288.0)
    x, y = to_image(10 + 1536, 40 + 1152, 10, 40, 1536, 1152)           # double size, bottom-right corner
    p = cr.clean_prompt(None, [[min(x, 767), min(y, 575)]], [], 768, 576)
    assert p["pos"] == [[767.0, 575.0]]
    html = (cr.ROOT / "api" / "annotate.html").read_text(encoding="utf-8")
    assert "img.naturalWidth / r.width" in html and "img.naturalHeight / r.height" in html


def test_prompt_modes_reach_sam(world):
    e = world.engine()
    cam = world.cams[0]
    for pr, lbl in ((cr.clean_prompt([300, 200, 460, 380], None, None, 768, 576), []),
                    (cr.clean_prompt(None, [[383, 287]], None, 768, 576), [1]),
                    (cr.clean_prompt([300, 200, 460, 380], [[383, 287], [390, 290]], [[440, 300]], 768, 576), [1, 1, 0])):
        e.mask(f"{RID}/v0", pr)
        sent = world.calls["prompts"][-1]
        assert sent["lbl"] == lbl and (sent["box"] is None) == (pr["box"] is None)
    m, _ = e.mask(f"{RID}/v0", cr.clean_prompt([300, 200, 460, 380], None, [[383, 287]], 768, 576))
    assert m.shape == (cam.H, cam.W) and not m[287, 383] and m[210, 310]        # exclude carved out


def test_sam_input_is_clean_and_embedding_cached(monkeypatch):
    import torch
    import pipeline.propose as pp
    pp._emb_cache.clear()
    seen, calls = [], {"emb": 0}

    class Proc:
        def __call__(self, images=None, **kw):
            seen.append(np.asarray(images).copy())
            out = {"pixel_values": torch.zeros(1, 3, 4, 4), "original_sizes": torch.tensor([[6, 9]])}
            out.update({k: torch.tensor(v, dtype=torch.float32) for k, v in kw.items() if k.startswith("input_")})
            return out

        def post_process_masks(self, m, sizes):
            return [torch.ones(1, 3, 6, 9)]

    class Model:
        def get_image_embeddings(self, pv):
            calls["emb"] += 1
            return [torch.zeros(1)]

        def __call__(self, image_embeddings=None, **kw):
            assert image_embeddings is not None and "pixel_values" not in kw
            return types.SimpleNamespace(pred_masks=torch.ones(1, 1, 3, 6, 9),
                                         iou_scores=torch.tensor([[[0.2, 0.8, 0.1]]]))

    monkeypatch.setattr(pp, "sam_model", lambda name: (Proc(), Model()))
    img = np.random.default_rng(7).integers(0, 255, (6, 9, 3), dtype=np.uint8)
    img.setflags(write=False)
    orig = img.copy()
    for pr in ({"box": [1, 1, 5, 4], "pts": [], "lbl": []}, {"box": None, "pts": [[2, 2]], "lbl": [1]},
               {"box": [0, 0, 8, 5], "pts": [[2, 2], [6, 3]], "lbl": [1, 0]}):
        emb, base, pil = pp.sam_embedding(img, "cache-test")
        m, s = pp.sam_predict(emb, base, pil, pr, "cache-test")
        assert m.shape == (6, 9) and abs(s - 0.8) < 1e-6
    assert calls["emb"] == 1                                  # one encoder pass for three prompts
    assert all(np.array_equal(x, orig) for x in seen) and np.array_equal(img, orig)
    pp._emb_cache.clear()


# ---------- support ----------

def test_low_coverage_patch_does_not_absorb_cluster(world):
    """B: only its front half is visible; the mask covers part of that front -> only visible supported voxels move,
    the unseen back half stays with B (uncertain near the selection), B is never claimed whole."""
    e = world.engine()
    cam = world.cams[0]
    patch = world.xyz[world.nA:world.nA + world.nBf]
    patch = patch[patch[:, 0] < 0.3]
    sel = e.support({f"{RID}/v0": mask_of(cam, patch, 2)})
    s = sel["selected"]
    assert len(s) and (world.lab[s] == 1).all() and (world.xyz[s, 2] < 2.2).all()
    assert len(s) < world.nBf                                     # not the whole cluster
    assert sel["stats"]["inferred_unseen"] == 0
    assert any(x["object"] == "OBJ-0001" for x in sel["splits"]) and not sel["members"]
    assert any("only" in w and "observed" in w for w in sel["warnings"])   # low coverage reported
    assert sel["stats"]["uncertain_left"] > 0                                  # occluded neighbours kept, flagged
    assert (world.xyz[sel["uncertain"], 2] > 2.0).all() and not np.isin(sel["uncertain"], s).any()


def test_structure_never_moved_and_whole_object_inferred_flagged(world):
    e = world.engine()
    cam = world.cams[0]
    m = np.zeros((cam.H, cam.W), bool)
    u, v, _ = cam.project(world.xyz[:world.nA])
    m[int(v.min()) - 10:int(v.max()) + 10, int(u.min()) - 10:int(u.max()) + 10] = True   # also covers wall behind
    W = np.where(world.lab == 2)[0]
    wu, wv, _ = cam.project(world.xyz[W])
    assert m[np.round(wv).astype(int), np.round(wu).astype(int)].any()                 # wall really in the mask
    sel = e.support({f"{RID}/v0": m})
    assert (world.lab[sel["selected"]] != 2).all()
    assert set(sel["selected"]) == set(range(world.nA))              # A complete, nothing else


def test_second_view_disagreement_is_explicit(world):
    e = world.engine()
    A = world.xyz[:world.nA]
    m0 = mask_of(world.cams[0], A)
    sel1 = e.support({f"{RID}/v0": m0})
    assert sel1["stats"]["evidence_level"] == "single-view-provisional"
    agree = e.support({f"{RID}/v0": m0, f"{RID}/v1": mask_of(world.cams[1], A)})
    assert agree["stats"]["evidence_level"] == "two-view" and agree["stats"]["disputed"] == 0
    assert agree["stats"]["multi_view_agree"] == len(agree["selected"]) == world.nA
    half = e.support({f"{RID}/v0": m0, f"{RID}/v1": mask_of(world.cams[1], A[A[:, 0] < 0], 1)})
    d = half["disputed"]
    assert len(d) > 0.3 * world.nA and (world.xyz[d, 0] >= -0.01).all()      # right half disputed ...
    assert not np.isin(d, half["selected"]).any()                             # ... and excluded
    assert half["stats"]["evidence_level"] == "views-disagree"
    assert any("disputed" in w for w in half["warnings"])
    g = e.guide(f"{RID}/v1", half)
    assert g["disputed"]["n_visible"] > 0 and g["selected"]["n_visible"] > 0


# ---------- full flow ----------

def test_unknown_class_flow_apply_undo_and_restart(world):
    import pipeline.scene as ps
    svc = service(world)
    key0, key1 = f"{RID}/v0", f"{RID}/v1"
    A = world.xyz[:world.nA]
    c = svc.create(RID)                                            # class optional -> unknown
    cid = c["id"]
    assert c["component_class"] == "unknown" and c["asset"]["status"] == "unassigned" and c["assembly_id"] is None
    r = svc.mask(cid, key0, box_px(world.cams[0], A), [], [], 1)
    assert not r["stale"] and r["W"] == 768
    p = svc.preview3d(cid)
    assert p["stats"]["selected"] == world.nA and p["stats"]["evidence_level"] == "single-view-provisional"
    assert p["next_views"] and p["next_views"][0]["key"] == key1
    with pytest.raises(ValueError, match="provisional"):
        svc.accept(cid, p["preview_hash"])                         # single view needs explicit acceptance
    svc.mask(cid, key0, box_px(world.cams[0], A), [], [], 2)       # prompts changed after the preview
    with pytest.raises(ValueError, match="out of date"):
        svc.accept(cid, p["preview_hash"], True)
    p = svc.preview3d(cid)
    rec = svc.accept(cid, p["preview_hash"], accept_provisional=True)
    assert rec["boundary_status"] == "accepted-provisional"
    assert svc.store.status(cid)["state"] == "accepted, not applied"
    base_sha = hashlib.sha256((world.base / "segment_points.npz").read_bytes()).hexdigest()
    man = svc.apply()
    out = cr.ROOT / world.cfg["correct"]["out_seg"]
    d = np.load(out / "segment_points.npz")
    assets = json.loads((out / "assets.json").read_text())
    new = next(a for a in assets if a["id"] == cid)
    assert new["label"] == "unknown" and new["manual"]["component_class"] == "unknown"
    assert new["manual"]["asset"]["status"] == "unassigned"
    assert new["manual"]["membership"] == [{"from": "OBJ-0000", "voxels": world.nA, "whole": True}]
    assert man["absorbed_objects"] == {"OBJ-0000": [cid]}
    assert (d["labels_before_manual"] == world.lab).all() and (d["src_rows"] == np.arange(len(world.xyz))).all()
    assert (d["labels"] >= 0).sum() == (world.lab >= 0).sum()                  # no point lost
    assert hashlib.sha256((world.base / "segment_points.npz").read_bytes()).hexdigest() == base_sha
    assert cr.downstream_stale(out, "scene_graph.json")                        # must be regenerated before use
    assert svc.store.status(cid)["state"] == "applied"
    # second, wider revision (includes part of B) then undo -> the first revision is active again
    svc.mask(cid, key0, box_px(world.cams[0], np.r_[A, world.xyz[world.nA:world.nA + 40]]), [], [], 3)
    p = svc.preview3d(cid)
    svc.accept(cid, p["preview_hash"], accept_provisional=True)
    svc.apply()
    assert (np.load(out / "segment_points.npz")["labels"] >= 0).sum() == (world.lab >= 0).sum()
    # restart: everything comes back from disk
    svc2 = service(world)
    assert svc2.store.status(cid)["accepted_rev"] == 2 and svc2.store.status(cid)["state"] == "applied"
    res = svc2.undo(cid)
    assert res == {**res, "undone": 2, "active": 1, "reapplied": True}
    ids = np.load(out / "segment_points.npz")["label_ids"]
    lab = np.load(out / "segment_points.npz")["labels"]
    assert (np.asarray(ids)[lab[:world.nA]] == cid).all() and (np.asarray(ids)[lab[world.nA:world.nA + world.nBf]] == "OBJ-0001").all()
    svc3 = service(world)
    svc3.undo(cid)                                                 # back to the original assignment
    lab = np.load(out / "segment_points.npz")["labels"]
    ids = np.asarray(np.load(out / "segment_points.npz")["label_ids"])
    assert list(ids[lab]) == list(np.array(["OBJ-0000", "OBJ-0001", "OBJ-0002"])[world.lab])
    with pytest.raises(ValueError):
        svc3.undo(cid)
    ev = [e["op"] for e in json.loads((cr.ROOT / world.cfg["correct"]["store"] / "index.json").read_text())["events"]]
    assert ev.count("accept") == 2 and ev.count("undo") == 2 and ev.count("apply") == 4
    assert ps.seg_revision(out) == ps.seg_revision(world.base)


def test_two_view_acceptance_and_reject(world):
    svc = service(world)
    A = world.xyz[:world.nA]
    cid = svc.create(RID, "valve", assembly_id="ASM-1", asset_tag="CV-003")["id"]
    c = svc.component(cid)
    assert c["asset"] == {**c["asset"], "tag": "CV-003", "status": "unverified"} and c["assembly_id"] == "ASM-1"
    svc.mask(cid, f"{RID}/v0", box_px(world.cams[0], A), [], [], 1)
    svc.mask(cid, f"{RID}/v1", box_px(world.cams[1], A), [], [], 1)
    p = svc.preview3d(cid)
    assert p["stats"]["evidence_level"] == "two-view"
    rec = svc.accept(cid, p["preview_hash"])                        # no provisional tick needed
    assert rec["boundary_status"] == "accepted-two-view"
    cid2 = svc.create(RID)["id"]
    svc.mask(cid2, f"{RID}/v0", box_px(world.cams[0], A), [], [], 1)
    p2 = svc.preview3d(cid2)
    assert p2["stats"]["overlaps"] == {cid: world.nA}                 # overlap with an accepted component shown
    svc.reject(cid2)
    st = svc.store.status(cid2)
    assert st["state"] == "rejected" and not st["draft_views"]


def test_stale_artifacts_block_accept_and_apply(world):
    svc = service(world)
    A = world.xyz[:world.nA]
    cid = svc.create(RID)["id"]
    svc.mask(cid, f"{RID}/v0", box_px(world.cams[0], A), [], [], 1)
    p = svc.preview3d(cid)
    svc.accept(cid, p["preview_hash"], True)
    png = world.pdir / "view0.png"
    orig = png.read_bytes()
    im = np.asarray(Image.open(png)).copy()
    im[0, 0] = 0
    Image.fromarray(im).save(png)                                  # render changed after acceptance
    with pytest.raises(ValueError, match="clean image changed"):
        svc.apply()
    assert not (cr.ROOT / world.cfg["correct"]["out_seg"]).exists()   # nothing written
    png.write_bytes(orig)
    svc.apply()
    s = dict(np.load(world.base / "segment_points.npz"))
    s["labels"] = s["labels"].copy()
    s["labels"][-1] = 0
    np.savez_compressed(world.base / "segment_points.npz", **s)    # base segmentation regenerated
    with pytest.raises(ValueError, match="base segmentation"):
        svc.apply()


def test_stale_inference_response_is_discarded(world, monkeypatch):
    svc = service(world)
    A = world.xyz[:world.nA]
    cid = svc.create(RID)["id"]
    key = f"{RID}/v0"
    real = svc.eng.mask

    def slow_mask(k, pr):                       # a newer request arrives while this one is "running"
        svc.latest_seq[(cid, k)] = 99
        return real(k, pr)

    monkeypatch.setattr(svc.eng, "mask", slow_mask)
    r = svc.mask(cid, key, box_px(world.cams[0], A), [], [], 5)
    assert r["stale"] and key not in svc.store.comp(cid)["draft"]["views"]       # never saved
    monkeypatch.setattr(svc.eng, "mask", real)
    assert svc.mask(cid, key, box_px(world.cams[0], A), [], [], 3)["stale"]      # older than the latest seq
    r = svc.mask(cid, key, box_px(world.cams[0], A), [[383, 287]], [], 100)
    assert not r["stale"] and svc.store.comp(cid)["draft"]["views"][key]["pos"] == [[383.0, 287.0]]


def test_http_api_roundtrip(world):
    from fastapi.testclient import TestClient
    from api.annotate import create_app
    svc = service(world)
    cl = TestClient(create_app(svc))
    assert "Boundary Correction" in cl.get("/").text
    assert [v["key"] for v in cl.get("/api/views").json()] == [f"{RID}/v0", f"{RID}/v1"]
    assert cl.get(f"/api/image?view={RID}/v0").headers["content-type"] == "image/png"
    cid = cl.post("/api/component", json={"region": RID}).json()["id"]
    bad = cl.post("/api/mask", json={"cid": cid, "view": f"{RID}/v0", "pos": [[1000, 10]], "seq": 1})
    assert bad.status_code == 400 and "outside" in bad.json()["detail"]
    ok = cl.post("/api/mask", json={"cid": cid, "view": f"{RID}/v0", "box": box_px(world.cams[0], world.xyz[:world.nA]),
                                    "seq": 2}).json()
    assert ok["overlay"].startswith("data:image/png;base64,")
    p = cl.post("/api/preview3d", json={"cid": cid}).json()
    assert cl.post("/api/accept", json={"cid": cid, "preview_hash": p["preview_hash"]}).status_code == 400
    assert cl.post("/api/accept", json={"cid": cid, "preview_hash": p["preview_hash"],
                                        "accept_provisional": True}).status_code == 200
    assert cl.post("/api/apply", json={}).json()["corrections"][0]["cid"] == cid
    assert cl.post("/api/undo", json={"cid": cid}).json()["reapplied"] is True
