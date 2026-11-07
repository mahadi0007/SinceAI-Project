"""label.py: earlier vision labels are reused for objects with identical geometry (no paid relabelling)."""
import json

from pipeline import label as L


def obj(i, n, c, **kw):
    return {"id": f"OBJ-{i:04d}", "kind": "object", "n_points": n, "center": c, **kw}


def cfg(tmp_path, files=()):
    return {"label": {"vision_sources": ["featherless_vision", "gemini_vision"], "reuse_from": [str(f) for f in files]}}


def test_geo_sig_matches_old_cache_format():
    assert L.geo_sig(obj(3, 5002, [0.92801, -0.18, -3.2141])) == "5002|[0.928, -0.18, -3.214]"


def test_reuse_from_old_cache_entries(tmp_path):
    cache = {"OBJ-0003": {"label": "pump", "confidence": 0.8, "model": "gemini-3.5-flash", "sig": "10|[1.0, 2.0, 3.0]"},
             "a91f3c": {"label": "motor", "confidence": 0.9}}                    # new-style entry: no sig
    r = L.reusable_labels(cfg(tmp_path), cache)
    assert list(r) == ["10|[1.0, 2.0, 3.0]"]
    assert r["10|[1.0, 2.0, 3.0]"]["source"] == "gemini_vision"


def test_reuse_from_labelled_file_only_vision(tmp_path):
    f = tmp_path / "assets_labeled.json"
    f.write_text(json.dumps([
        obj(1, 10, [1, 2, 3], label="tank/vessel", label_confidence=0.9, label_source="gemini_vision"),
        obj(2, 20, [1, 2, 3], label="pipe segment", label_confidence=0.3, label_source="geometry_rules"),
        {"id": "S-1", "kind": "structure", "label": "wall", "label_source": "plane_fit"}]), encoding="utf-8")
    r = L.reusable_labels(cfg(tmp_path, [f]), {})
    assert set(r) == {L.geo_sig(obj(1, 10, [1, 2, 3]))}                        # rules / structures not reused
    assert r[L.geo_sig(obj(1, 10, [1, 2, 3]))]["label"] == "tank/vessel"


def test_changed_geometry_is_not_reused(tmp_path):
    cache = {"OBJ-0001": {"label": "pump", "confidence": 0.8, "sig": L.geo_sig(obj(1, 10, [1, 2, 3]))}}
    r = L.reusable_labels(cfg(tmp_path), cache)
    assert L.geo_sig(obj(1, 11, [1, 2, 3])) not in r                            # one point more: new object
    assert L.geo_sig(obj(1, 10, [1, 2, 3.01])) not in r                         # moved 1 cm: new object


def test_reused_label_passes_decide():
    lc = {"vocab": ["pump", "pipe segment"], "min_conf": 0.45, "rules_conf": 0.3}
    a = obj(1, 10, [1, 2, 3], extent=[0.3, 0.3, 0.3], hint="unlabelled",
            features={"linearity": 0.1, "planarity": 0.1})
    g = {"label": "pump", "confidence": 0.8, "reason": "r", "model": "gemini-3.5-flash", "source": "gemini_vision"}
    d = L.decide(a, g, lc)
    assert d["label"] == "pump" and d["label_source"] == "gemini_vision"
