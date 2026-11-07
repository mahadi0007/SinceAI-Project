"""BIM in the normal viewer: coordinates, versioning, associations, shared edits, navigation invalidation.

Pure tests always run; the real-scene tests use out/ (scene.ifc, removable.json from the same segmentation) and
are skipped when those artifacts are missing. The viewer test runs on a temporary copy of out/, so the real
scene_edits.json is never touched.
"""
from __future__ import annotations

import json
import os
import shutil
import socket

import numpy as np
import pytest
from viser.transforms import SO3

from pipeline.bim_provenance import ArtifactMismatch, check_bim, compare, scene_revision
from pipeline.common import ROOT, load_config
from pipeline.export_ifc import from_ifc, rot_from_ifc, rot_to_ifc, to_ifc
from viewer import scene_edits as se

RNG = np.random.default_rng(0)
U2M = 1.15


def yaw(deg: float) -> np.ndarray:
    """Rotation about the scene's vertical axis (y)."""
    return SO3.from_y_radians(np.radians(deg)).wxyz


# ---------- coordinates ----------

def test_point_round_trip_and_axes():
    p = RNG.normal(size=(50, 3))
    assert np.allclose(from_ifc(to_ifc(p, U2M), U2M), p)
    up_scene = np.array([0.0, -1.0, 0.0])                  # scene up = -y  ->  IFC up = +z
    assert np.allclose(to_ifc(up_scene), [0, 0, 1])
    assert np.allclose(to_ifc([1.0, 0, 0], U2M), [U2M, 0, 0])


def test_rotation_conversion_matches_point_mapping():
    for q in (yaw(37), SO3.from_x_radians(0.3).wxyz, SO3.from_z_radians(-1.1).wxyz):
        R = SO3(q).as_matrix()
        p = RNG.normal(size=(20, 3))
        assert np.allclose(to_ifc(p @ R.T, U2M), to_ifc(p, U2M) @ rot_to_ifc(R).T)
        assert np.allclose(rot_from_ifc(rot_to_ifc(R)), R)
    Ri = rot_to_ifc(SO3(yaw(30)).as_matrix())            # a yaw in the scene stays a yaw (about IFC +z)
    assert np.allclose(Ri @ [0, 0, 1], [0, 0, 1])


def test_bim_pose_round_trip_and_same_rigid_motion():
    pivot, centre = np.array([1.2, 0.7, -1.8]), np.array([1.4, -2.1, -0.8])
    pos, q = pivot + [0.5, 0.0, -0.3], yaw(75)
    pos_i, q_i = se.to_bim_pose(pos, q, pivot, centre, U2M)
    back_p, back_q = se.from_bim_pose(pos_i, q_i, centre, pivot, U2M)
    assert np.allclose(back_p, pos) and np.allclose(np.abs(np.dot(back_q, q)), 1)
    p = RNG.normal(size=(30, 3)) + pivot                     # bim_view: R_i (x - c) + pos_i in IFC
    Ri = SO3(q_i).as_matrix()
    assert np.allclose(to_ifc(se.apply(p, pos, q, pivot), U2M), (to_ifc(p, U2M) - centre) @ Ri.T + pos_i)


# ---------- edit formats ----------

PIVOTS = {"OBJ-0029": [1.222, 0.742, -1.85], "OBJ-0034": [1.242, 0.76, 0.265]}
CENTRES = {k: to_ifc(np.array(v) + 0.01, U2M) for k, v in PIVOTS.items()}


def test_viewer_edits_adapter_keeps_copies_as_copies():
    legacy = {"OBJ-0029": {"oid": "OBJ-0029", "position": [1.5, 0.742, -1.85], "wxyz": list(yaw(20)), "removed": False},
              "OBJ-0029_copy1": {"oid": "OBJ-0029", "position": [2, 0.742, -1], "wxyz": [1, 0, 0, 0], "removed": False},
              "OBJ-9999": {"oid": "OBJ-9999", "position": [0, 0, 0], "wxyz": [1, 0, 0, 0], "removed": True}}
    inst, skipped = se.from_viewer_edits(legacy, PIVOTS)
    assert skipped == ["OBJ-9999"]
    assert inst["OBJ-0029_copy1"]["copy"] and inst["OBJ-0029_copy1"]["source"] == "OBJ-0029"
    assert not inst["OBJ-0029"]["copy"] and inst["OBJ-0034"] == se.instance("OBJ-0034", PIVOTS["OBJ-0034"])


def test_bim_edits_adapter_round_trip_and_unsupported_kept():
    inst = se.originals(PIVOTS)
    inst["OBJ-0029"].update(position=[1.6, 0.742, -1.5], wxyz=list(yaw(-40)))
    inst["OBJ-0034"]["removed"] = True
    inst["OBJ-0029_copy1"] = se.instance("OBJ-0029", PIVOTS["OBJ-0029"], True, [0.2, 0.742, 0.4], yaw(90))
    bim = se.to_bim_edits(inst, PIVOTS, CENTRES, U2M, {"groups": {"OBJ-0029": ["OBJ-0030"]}})
    assert bim["removed"] == ["OBJ-0034"] and bim["groups"] == {"OBJ-0029": ["OBJ-0030"]}
    back, bim_only = se.from_bim_edits(bim, PIVOTS, CENTRES, U2M)
    assert bim_only == {"groups": {"OBJ-0029": ["OBJ-0030"]}}
    for k in inst:
        assert back[k]["copy"] == inst[k]["copy"] and back[k]["removed"] == inst[k]["removed"]
        assert np.allclose(back[k]["position"], inst[k]["position"], atol=1e-3)
        assert np.allclose(np.abs(np.dot(back[k]["wxyz"], inst[k]["wxyz"])), 1, atol=1e-4)


def test_layout_hash_ignores_untouched_and_undo_restores():
    st = se.new_state({"segmentation": "x"}, PIVOTS)
    base = se.layout_hash(st["instances"], PIVOTS)
    se.push_history(st, "move")
    st["instances"]["OBJ-0029"]["position"] = [9, 9, 9]
    assert se.layout_hash(st["instances"], PIVOTS) != base
    assert se.undo(st) == "move" and se.layout_hash(st["instances"], PIVOTS) == base
    assert se.undo(st) is None


def test_revision_compare_reports_each_mismatch():
    want = {"ply_sha256": "a", "segmentation": "s1", "n_objects": 3, "units_to_m": 1.15, "up": "-y"}
    assert compare(dict(want), want, "x") == []
    errs = compare({**want, "segmentation": "s0", "n_objects": 2}, want, "scene.ifc")
    assert len(errs) == 2 and all(e.startswith("scene.ifc: ") for e in errs)
    assert "no scene revision" in compare(None, want, "removable.json")[0]


# ---------- real scene (out/) ----------

OUT = ROOT / "out"
NEED = ["scene.ifc", "bim_manifest.json", "removable.json", "removable_rows.npz", "scene_graph.json"]
real = pytest.mark.skipif(not all((OUT / n).exists() for n in NEED), reason="out/ BIM artifacts not generated")


@pytest.fixture(scope="module")
def cfg():
    os.environ.pop("VALMET_OUT", None)
    return load_config()


@real
def test_ifc_built_from_current_scene_and_mismatch_rejected(cfg, tmp_path, monkeypatch):
    rev = scene_revision(cfg)
    m = check_bim(cfg, rev)
    assert m["revision"] == rev
    for n in ("scene.ifc", "bim_manifest.json", "deviation.npz", "segment_manifest.json", "scene_graph.json",
              "segment_points.npz"):
        shutil.copy2(OUT / n, tmp_path / n)
    monkeypatch.setenv("VALMET_OUT", str(tmp_path))
    bad = json.loads((tmp_path / "bim_manifest.json").read_text())
    bad["revision"]["segmentation"] = "b9ccba976377"                   # e.g. an out_geo-era model
    (tmp_path / "bim_manifest.json").write_text(json.dumps(bad))
    with pytest.raises(ArtifactMismatch, match="segmentation"):
        check_bim(cfg, scene_revision(cfg))
    (tmp_path / "bim_manifest.json").write_text((OUT / "bim_manifest.json").read_text())
    with open(tmp_path / "scene.ifc", "a") as f:                       # file edited after export
        f.write("\n")
    with pytest.raises(ArtifactMismatch, match="changed"):
        check_bim(cfg, scene_revision(cfg))


@real
def test_ifc_elements_associate_with_scene_objects(cfg):
    from viewer.bim_layer import BimLayer
    sg = json.loads((OUT / "scene_graph.json").read_text(encoding="utf-8"))
    nodes = [n for n in sg["objects"] if n["kind"] == "object"]
    layer = BimLayer(cfg, nodes, scene_revision(cfg))
    assert set(layer.mesh) == {n["id"] for n in nodes}
    assert layer.centre_offset < 0.1
    by_id = {n["id"]: n for n in nodes}
    for n in nodes:
        if n["asset"] and n["asset"]["asset_class"] == "PROCESS_PUMP":
            assert layer.mesh[n["id"]]["cls"] == "IfcPump", n["id"]
    p3 = next(n for n in nodes if n["asset"] and n["asset"]["tag"] == "P-003")
    m = layer.mesh[p3["id"]]
    lo, hi = np.array(p3["bbox_min"]), np.array(p3["bbox_max"])
    assert np.all(m["lo"] > lo - 0.15) and np.all(m["hi"] < hi + 0.15)    # catalog shape inside the scan box
    assert by_id[p3["id"]]["asset"]["tag"] == "P-003"


@real
def test_removable_ids_are_current_segmentation(cfg):
    r = json.loads((OUT / "removable.json").read_text(encoding="utf-8"))
    assert r["revision"] == scene_revision(cfg)
    assert all(v["iou"] >= 0.9 for v in r["include_mapping"].values())
    assert set(r["removable"]) == {v["id"] for v in r["include_mapping"].values()}


@real
def test_layout_navigation_moves_obstacles(cfg):
    from pipeline.layout_nav import build, edited_points
    from pipeline.tour import load_inputs
    seg = np.load(OUT / "segment_points.npz")
    sg = json.loads((OUT / "scene_graph.json").read_text(encoding="utf-8"))
    idx = {o["id"]: k for k, o in enumerate(sg["objects"])}
    oid = "OBJ-0029"
    pivot = sg["objects"][idx[oid]]["center"]
    st = se.new_state(scene_revision(cfg), {oid: pivot})
    st["instances"][oid]["removed"] = True
    xyz, lab = edited_points(seg["xyz"].astype(float), seg["labels"], {oid: [idx[oid]]}, st["instances"], st["pivots"])
    assert not (lab == idx[oid]).any() and len(xyz) == len(seg["xyz"]) - int((seg["labels"] == idx[oid]).sum())
    os.environ["VALMET_OUT"] = str(OUT)
    try:
        res = build(cfg, st, {oid: [idx[oid]]}, se.layout_hash(st["instances"], st["pivots"]))
    finally:
        os.environ.pop("VALMET_OUT", None)
        for n in ("layout_nav.npz", "layout_nav.json"):                  # derived files of this test only
            (OUT / n).unlink(missing_ok=True)
    _, base, _, _ = load_inputs(cfg)
    assert res["report"]["obstacle_cells"] < res["report"]["obstacle_cells_before"]
    assert res["bounds"][oid]["removed"] and res["inputs"][1]["walkable"].sum() >= base["walkable"].sum()


# ---------- the integrated viewer on a copy of out/ ----------

COPY = ["segment_manifest.json", "segment_points.npz", "scene_graph.json", "path.json", "tour.json", "navgrid.npz",
        "assets.json", "assets_labeled.json", "scene.ifc", "deviation.npz", "bim_manifest.json", "removable.json",
        "removable_rows.npz"]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    if not all((OUT / n).exists() for n in NEED):
        pytest.skip("out/ BIM artifacts not generated")
    d = tmp_path_factory.mktemp("out_copy")
    for n in COPY:
        shutil.copy2(OUT / n, d / n)
    for p in OUT.glob("diagnosis_*.json"):
        shutil.copy2(p, d / p.name)
    for p in OUT.glob("evidence_*.json"):
        shutil.copy2(p, d / p.name)
    os.environ["VALMET_OUT"] = str(d)
    os.environ["VALMET_OFFLINE"] = "1"
    c = load_config()
    c["copilot"]["offline"] = True
    c["viewer"]["voice"]["enabled"] = False
    c["viewer"]["port"] = free_port()
    c["correct"]["store"] = str(d / "corrections")
    from viewer.app import Sim
    s = Sim(c)
    import time
    for _ in range(240):
        if s.bim is not None or "unavailable" in s.bim_note:
            break
        time.sleep(0.25)
    assert s.bim is not None, s.bim_note
    s.out = d
    yield s
    s.server.stop()
    if s.annot is not None and hasattr(s.annot, "shutdown"):
        s.annot.shutdown()
    os.environ.pop("VALMET_OUT", None)


def ray_to(sim, n, cands=((0, -0.4, 2.5), (2.5, -0.4, 0), (-2.5, -0.4, 0), (0, -0.4, -2.5), (0, -2.5, 0.3))):
    """A ray from free space that hits object n's scan points first."""
    c = np.array(n["center"])
    for off in cands:
        o = c + np.array(off, float)
        if sim._pick_scan(o, (c - o) / np.linalg.norm(c - o))[1] is n:
            return o, (c - o) / np.linalg.norm(c - o)
    pytest.skip(f"no unobstructed view of {n['id']}")


def test_display_modes(sim):
    for mode, splats, meshes, wire in (("BIM", False, True, False), ("Overlay", True, True, True),
                                       ("Scan", True, False, False)):
        sim.set_display(mode)
        assert sim.h_splat["room"].visible == splats
        assert all(h.visible == meshes for h in sim.h_bim.values()) and len(sim.h_bim) > 900
        assert all(h.wireframe == wire for h in sim.h_bim.values())
        it = sim.inst["OBJ-0029"]
        assert it["splat"].visible == splats and it["bim"].visible == meshes


def test_real_equipment_select_move_restore(sim):
    p3 = sim.find_node("P-003")
    assert p3["id"] == "OBJ-0029" and p3["id"] in sim.movable
    # same object from scan points and from its IFC element
    o, d = ray_to(sim, p3)
    sim.set_display("Scan")
    assert sim.pick(o, d) is p3 and sim._picked_inst == "OBJ-0029"
    sim.set_display("BIM")
    assert sim.pick(o, d) is p3 and sim._picked_inst == "OBJ-0029"
    sim.select(p3, inst=sim._picked_inst)
    assert sim.selected is p3 and "IfcPump" in sim.md_sel.content and "P-003" in sim.dd_equip.value
    # health outline + light tint on the BIM element, class colour underneath
    sim.highlight_stop({"object_id": p3["id"], "kind": "goto"})
    col = tuple(sim.h_stop_box.colors[0, 0])
    assert sim.h_stop_box.visible and sim.h_stop_box.thickness == 6.0
    healthy = p3["asset"]["condition_state"] == "healthy"
    assert (col[1] > col[0]) if healthy else (col[0] > col[1])
    assert tuple(sim.inst["OBJ-0029"]["bim"].color) != tuple(sim.bim.mesh["OBJ-0029"]["color"])
    # move + turn: scan piece, BIM mesh, bounds, outline and picking follow one transform
    it = sim.inst["OBJ-0029"]
    c = sim.movable["OBJ-0029"]["c"]
    new_pos, q = c + np.array([0.6, 0.0, 0.4]), yaw(90)
    sim.move_instance("OBJ-0029", new_pos, q)
    assert np.allclose(it["h"].position, new_pos)
    assert it["splat"].name == "/movable/OBJ-0029/splat" and it["bim"].name == "/movable/OBJ-0029/bim"
    lo, hi = sim.inst_bounds("OBJ-0029")
    pts = se.apply(sim.movable["OBJ-0029"]["pick"] + c, new_pos, q, c)
    assert np.allclose(lo, pts.min(0)) and np.allclose(hi, pts.max(0))
    bim_world = se.apply(sim.movable["OBJ-0029"]["bim"][0] + c, new_pos, q, c)     # mesh shares the frame
    assert np.all(bim_world.min(0) > lo - 0.2) and np.all(bim_world.max(0) < hi + 0.2)
    seg = sim.h_stop_box.points.reshape(-1, 3)
    assert np.allclose(seg.min(0), lo - sim.vc["stop_outline_pad"], atol=1e-5)
    o2 = o + (new_pos - c)
    t_scan = sim._pick_scan(o2, d)
    t_bim = sim._pick_bim(o2, d)
    assert t_scan[2] == "OBJ-0029" or t_bim[2] == "OBJ-0029"           # found where it is now
    # navigation: stale until Apply layout; nothing walks a route built for the old layout
    assert sim.nav_stale() and not sim.h_path.visible
    sim.go_to(sim.find_node("M-005"))
    assert not sim.playing and "Apply layout" in sim.md_walk.content
    assert not sim.handle_command({"cmd": "tour"})["ok"]
    sim.apply_layout()
    assert sim.nav_stale() is None and sim.nav_inputs is not None and sim.h_path.visible
    assert np.allclose(p3["bbox_min"], lo, atol=1e-4)                       # node bounds updated
    ed = json.loads((sim.out / "scene_edits.json").read_text())
    assert ed["applied_layout"]["hash"] == sim.layout_hash() and ed["revision"] == sim.revision
    assert (sim.out / "layout_nav.npz").exists() and (sim.out / "navgrid.npz").stat().st_size > 0
    # restore (with undo history) -> original navigation again
    sim.clear_stop_highlight()
    sim.edit_action("restore OBJ-0029", lambda: sim.restore("OBJ-0029"))
    assert sim.layout_hash() == sim.base_layout and sim.nav_stale() is None and sim.nav_inputs is None
    assert np.allclose(p3["bbox_min"], sim._bounds0["OBJ-0029"]["bbox_min"])
    assert sim.pick(o, d) is p3
    sim.undo_edit()                                                       # undo the restore: moved again
    assert np.allclose(it["h"].position, new_pos)
    sim.undo_edit()                                                       # undo the move
    assert sim.layout_hash() == sim.base_layout and sim.nav_stale() is None


def test_chat_edits_share_undo_and_navigation(sim):
    from copilot.chat import parse_quick, run_quick
    node = sim.find_node("P-003")
    oid = node["id"]
    sim.select(node, inst=oid)
    pivot = np.array(sim.inst[oid]["h"].position)
    for text in ("move it 20 cm up", "turn it 30 degrees right", f"remove {oid}"):
        answer = run_quick(parse_quick(text), sim.chat_act)
        assert not answer.startswith("⚠️"), answer
        assert sim.nav_stale() and not sim.h_path.visible
        if text.startswith("move"):
            assert np.allclose(sim.inst[oid]["h"].position, pivot + [0, -.2 / sim.cfg["scene"]["units_to_m"], 0])
        elif text.startswith("remove"):
            assert sim.inst[oid]["removed"] and not sim.inst[oid]["h"].visible
        sim.undo_edit()
        assert sim.nav_stale() is None and not sim.inst[oid]["removed"]
        sim.select(node, inst=oid)
    sim.edit_command({"cmd": "move", "target": oid, "right_m": .2})
    moved = sim.inst[oid]["h"].position
    assert sim.edit_command({"cmd": "restore", "target": oid})["ok"]
    assert sim.nav_stale() is None and np.allclose(sim.inst[oid]["h"].position, pivot)
    sim.undo_edit()
    assert np.allclose(sim.inst[oid]["h"].position, moved) and sim.nav_stale()
    sim.undo_edit()
    assert sim.nav_stale() is None


def test_copy_has_no_identity_and_persists(sim):
    sim.set_display("Overlay")
    p3 = sim.find_node("P-003")
    sim.select(p3, inst="OBJ-0029")
    name = sim.duplicate("OBJ-0029")
    it = sim.inst[name]
    assert it["copy"] and it["bim"] is not None and it["bim"].wireframe
    n = sim.selected
    assert n["asset"] is None and n["copy_of"] == "OBJ-0029" and n["id"] == name
    assert "No asset record" in sim.md_sel.content and "copy" in sim.md_sel.content
    assert sim.find_node("P-003") is p3                                     # identity stays with the original
    sim.highlight_stop({"object_id": name, "kind": "goto"})                 # a copy is never shown as healthy/faulty
    assert tuple(sim.h_stop_box.colors[0, 0]) == (170, 180, 190)
    sim.clear_stop_highlight()
    ed = json.loads((sim.out / "scene_edits.json").read_text())
    assert ed["instances"][name]["copy"] and ed["instances"][name]["source"] == "OBJ-0029"
    st, _ = se.load(sim.cfg, sim.revision, {k: m["c"] for k, m in sim.movable.items()})
    assert st["instances"][name] == ed["instances"][name]                    # reload gives the same layout
    assert sim.nav_stale()
    # edited IFC: explicit, separate file; the copy has no asset pset; scene.ifc unchanged
    before = (sim.out / "scene.ifc").read_bytes()
    sim.export_edited_ifc()
    assert "scene_edited.ifc" in sim.md_ifc.content and (sim.out / "scene.ifc").read_bytes() == before
    import ifcopenshell
    import ifcopenshell.util.element as ue
    f = ifcopenshell.open(str(sim.out / "scene_edited.ifc"))
    copies = [e for e in f.by_type("IfcElement") if ue.get_psets(e).get("Pset_ScanExtraction", {}).get("CopyOf")]
    assert len(copies) == 1 and "Pset_ValmetAsset" not in ue.get_psets(copies[0])
    from viewer.bim_view import tessellate
    from pipeline.export_ifc import from_ifc as fi
    m = next(e for e in tessellate(f) if e["el"] == copies[0])
    v = fi(m["verts"], sim.cfg["scene"]["units_to_m"])
    want = se.apply(sim.bim.mesh["OBJ-0029"]["verts"], it["h"].position, it["h"].wxyz, sim.movable["OBJ-0029"]["c"])
    assert np.allclose(np.sort(v, 0), np.sort(want, 0), atol=2e-3)
    sim.undo_edit()
    assert name not in sim.inst and sim.nav_stale() is None


def test_failed_rebuild_in_edit_callback_keeps_walking_blocked(sim, monkeypatch):
    c = sim.movable["OBJ-0049"]["c"]
    sim.edit_action("move OBJ-0049", lambda: setattr(sim.inst["OBJ-0049"]["h"], "position", tuple(c + [0.3, 0, 0])))
    sim.apply_layout()
    applied = sim.layout_hash()
    assert sim.nav_stale() is None and sim.edits["applied_layout"]["hash"] == applied
    sim.undo_edit()                                                   # back to the unedited scene: no rebuild needed
    assert sim.nav_stale() is None and sim.nav_inputs is None

    def broken(_layout):
        raise ValueError("layout changed while navigation was being rebuilt")
    monkeypatch.setattr(sim, "_build_nav", broken)
    sim.edit_action("move OBJ-0049", lambda: setattr(sim.inst["OBJ-0049"]["h"], "position", tuple(c + [0.3, 0, 0])))
    assert sim.layout_hash() == applied                               # the applied layout again -> rebuild fails
    assert sim.nav_stale() and not sim.playing and not sim.h_path.visible
    assert "rebuild failed" in sim.md_layout.content
    sim.go_to(sim.find_node("M-005"))
    assert not sim.playing                                            # never falls back to the unedited routes
    monkeypatch.undo()
    sim.apply_layout()                                                # retry succeeds once the inputs are fine
    assert sim.nav_stale() is None and "rebuild failed" not in sim.md_layout.content


def test_viewer_starts_when_saved_layout_cannot_be_rebuilt(sim):
    """A saved applied_layout with missing out/ navigation files must not stop the viewer from starting."""
    from viewer.app import Sim
    assert sim.edits["applied_layout"] and sim.layout_hash() == sim.edits["applied_layout"]["hash"]
    (sim.out / "navgrid.npz").unlink()
    cfg2 = {**sim.cfg, "viewer": {**sim.cfg["viewer"], "port": free_port()}}
    s2 = Sim(cfg2)
    try:
        assert s2.layout_hash() == sim.layout_hash()
        assert s2.nav_stale() and not s2.h_path.visible and "rebuild failed" in s2.nav_error
        s2.go_to(s2.find_node("M-005"))
        assert not s2.playing
    finally:
        s2.server.stop()


def test_mismatched_edits_rejected(sim, tmp_path):
    bad = se.new_state({**sim.revision, "segmentation": "b9ccba976377"}, {"OBJ-0029": [0, 0, 0]})
    (sim.out / "scene_edits.json").write_text(json.dumps(bad))
    with pytest.raises(ArtifactMismatch, match="segmentation"):
        se.load(sim.cfg, sim.revision, {k: m["c"] for k, m in sim.movable.items()})
