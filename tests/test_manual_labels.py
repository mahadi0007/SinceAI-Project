"""Manual correction labels remain authoritative class evidence without becoming asset identity."""
from __future__ import annotations

from pipeline.common import is_vision_label
from pipeline.label import decide, manual_decision, vlm_todo


LC = {
    "vocab": ["pump", "control valve", "manual valve/handwheel", "pipe fitting/flange", "other"],
    "min_conf": 0.45,
    "rules_conf": 0.4,
    "manual_sources": ["manual_review"],
    "manual_class_vocab": {"pump": "pump"},
    "manual_label_overrides": {"MC-0006": "control valve"},
}


def asset(cid: str, cls: str, hint: str = "fitting/valve") -> dict:
    return {"id": cid, "kind": "object", "hint": hint, "extent": [0.2, 0.2, 0.2],
            "features": {"linearity": 0.1, "planarity": 0.1}, "label_source": "manual_review",
            "manual": {"component_id": cid, "revision": 2, "component_class": cls,
                       "evidence_level": "two-view"}}


def apply_label(a: dict) -> dict:
    a["label_manual_class"] = a["manual"]["component_class"]
    a.update(manual_decision(a, LC) or decide(a, None, LC))
    return a


def test_manual_pump_is_confident_reviewer_label():
    a = apply_label(asset("MC-0004", "pump"))
    assert (a["label"], a["label_confidence"], a["label_source"]) == ("pump", 1.0, "manual_review")


def test_manual_valve_override_wins():
    assert apply_label(asset("MC-0006", "valve"))["label"] == "control valve"


def test_unresolved_manual_classes_fall_back_and_keep_reviewer_class():
    for cls in ("valve", "unknown"):
        a = apply_label(asset("MC-0099", cls))
        assert a["label"] == "pipe fitting/flange"
        assert a["label_source"] == "geometry_rules"
        assert a["label_manual_class"] == cls


def test_manual_source_participates_in_downstream_class_matching():
    cfg = {"label": {"vision_sources": ["featherless_vision"], "manual_sources": ["manual_review"]}}
    pump = {"label_source": "manual_review", "label": "pump", "label_confidence": 1.0}
    unknown = {"label_source": "manual_review", "label": "unknown", "label_confidence": 1.0}
    assert is_vision_label(pump, cfg, ["pump"], 0.8)
    assert not is_vision_label(unknown, cfg, ["pump"], 0.8)


def test_manual_objects_are_never_paid_todo():
    manual = asset("MC-0004", "pump")
    ordinary = {"id": "OBJ-0001", "kind": "object"}
    todo = vlm_todo([manual, ordinary], {"MC-0004": "m", "OBJ-0001": "o"}, {})
    assert [a["id"] for a in todo] == ["OBJ-0001"]
