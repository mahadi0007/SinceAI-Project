"""Shared field-service tools (service layer) used by the MCP server and, later, the viewer chat/API.

Plain functions over our own data: Excel builder pack (via copilot.evidence), scene graph and
navigation outputs (out/). Every function returns JSON-serialisable dicts and never raises for bad
input: problems come back as {"ok": False, "message": ...}.

Cost rule: nothing here spends paid LLM requests unless explicitly asked (diagnose(allow_paid=True)).
Viewer control works through a small command file (out/viewer_cmd.jsonl) that the running viewer polls;
results come back in out/viewer_cmd_results.jsonl.
"""
from __future__ import annotations

import json
import time
import uuid
from functools import lru_cache
from typing import Any

import numpy as np

from copilot.evidence import build_evidence, load_pack
from pipeline.common import load_config, out_path, repo_path

Cfg = dict[str, Any]
URGENCY = {"attention_required": 0, "degraded": 1, "watch": 2, "healthy": 3}


# ---------- data access ----------

@lru_cache(maxsize=1)
def cfg() -> Cfg:
    return load_config()


def scene() -> dict:
    with open(out_path(cfg(), "scene_graph.json"), encoding="utf-8") as f:
        return json.load(f)


def excel() -> dict:
    return load_pack(str(repo_path(cfg()["paths"]["xlsx"])))


def resolve(key: str) -> dict | None:
    """Scene node for an asset tag (P-002), asset id (ASSET-014) or object id (OBJ-0269); case-insensitive."""
    k = str(key).strip().upper()
    for n in scene()["objects"]:
        a = n.get("asset")
        if n["id"].upper() == k or (a and k in (a["tag"].upper(), a["asset_id"].upper())):
            return n
    return None


def u2m(v: float) -> float:
    return round(float(v) * cfg()["scene"]["units_to_m"], 2)


def summary(n: dict) -> dict:
    a = n.get("asset")
    base = {"object_id": n["id"], "label": n["label"], "label_confidence": n.get("label_confidence")}
    if not a:
        return {**base, "tag": None, "asset_record": False}
    return {**base, "asset_record": True, "tag": a["tag"], "asset_id": a["asset_id"], "asset_class": a["asset_class"],
            "condition_state": a["condition_state"], "health_score": a["health_score"],
            "issue_code": a["issue_code"], "urgency": a["urgency"],
            "mapping_confidence": a.get("mapping_confidence")}


# ---------- read-only tools ----------

def search_assets(condition: str | None = None, asset_class: str | None = None, issue: str | None = None,
                  text: str | None = None, limit: int = 30) -> dict:
    """Mapped Excel assets filtered by condition / class / issue / free text, most urgent first."""
    out = []
    for n in scene()["objects"]:
        a = n.get("asset")
        if not a:
            continue
        hay = " ".join(str(v) for v in (a["tag"], a["asset_class"], a["condition_state"], a["issue_code"],
                                       a.get("issue_summary", ""), n["label"])).lower()
        if condition and a["condition_state"].lower() != condition.lower():
            continue
        if asset_class and asset_class.lower() not in a["asset_class"].lower():
            continue
        if issue and issue.lower() not in a["issue_code"].lower():
            continue
        if text and not all(w in hay for w in text.lower().split()):
            continue
        out.append(summary(n))
    out.sort(key=lambda s: (URGENCY.get(s["condition_state"], 9), s["health_score"]))
    return {"ok": True, "count": len(out), "assets": out[:limit]}


def label_matches(label: str, query: str) -> bool:
    """'tank' matches 'tank/vessel', 'stairs' matches 'ladder/stairs', 'cabinet' matches 'electrical cabinet'."""
    q = query.strip().lower().rstrip("s") if len(query.strip()) > 3 else query.strip().lower()
    parts = [p.strip() for p in label.lower().replace("/", " / ").split("/")] + [label.lower()]
    return any(q and (q in p or p in query.lower()) for p in parts if p)


def objects_by_label(query: str) -> list[dict]:
    """All detected scene objects whose vision label matches, most confident first."""
    hits = [n for n in scene()["objects"] if n["kind"] == "object" and label_matches(n["label"], query)]
    return sorted(hits, key=lambda n: -(n.get("label_confidence") or 0))


def find_objects(label: str | None = None, text: str | None = None, limit: int = 20) -> dict:
    """Search ALL detected 3D objects (also those without an Excel record) by vision label or reason text."""
    hits = objects_by_label(label) if label else [n for n in scene()["objects"] if n["kind"] == "object"]
    if text:
        words = text.lower().split()
        hits = [n for n in hits if all(w in f"{n['label']} {n.get('label_reason', '')}".lower() for w in words)]
    labels = sorted({n["label"] for n in scene()["objects"] if n["kind"] == "object"})
    if not hits:
        return {"ok": False, "message": f"No detected object matches label={label!r} text={text!r}.",
                "known_labels": labels}
    return {"ok": True, "count": len(hits),
            "objects": [{**summary(n), "reason": n.get("label_reason"),
                         "center_m": [u2m(v) for v in n["center"]], "size_m": [u2m(v) for v in n["extent"]]}
                        for n in hits[:limit]],
            "note": "Objects with asset_record=false are detected in 3D but have no Excel data; "
                    "use their object_id with walk_in_viewer / show_in_viewer / route_to."}


def get_asset(key: str) -> dict:
    """Identity, condition and 3D mapping of one asset or scene object."""
    n = resolve(key)
    if n is None:
        return {"ok": False, "message": f"Unknown asset/object '{key}'."}
    a = n.get("asset")
    res = {"ok": True, **summary(n), "size_m": [u2m(v) for v in n["extent"]],
           "height_above_floor_m": u2m(n.get("features", {}).get("height_bottom", 0.0))}
    if a:
        res.update({k: a.get(k) for k in ("area_name", "criticality", "issue_summary", "recommended_next_check",
                                          "knowledge_doc_id", "mapping_method", "mapping_evidence")})
    else:
        res["message"] = "No asset record: this object is not in the Excel pack."
    return res


def _pack(key: str) -> tuple[dict | None, str]:
    n = resolve(key)
    if n is None or not n.get("asset"):
        return None, (f"Unknown asset '{key}'." if n is None else f"{n['id']} has no asset record (no Excel data).")
    return build_evidence(n["asset"]["tag"], cfg()), ""


def telemetry_trend(key: str, metric: str | None = None) -> dict:
    """3-day trends (first vs last 24 h) with healthy same-class peers; evidence IDs TEL-<tag>-<metric>."""
    pack, msg = _pack(key)
    if pack is None:
        return {"ok": False, "message": msg}
    tel = [t for t in pack["telemetry"] if not metric or metric.lower() in t["metric"].lower()]
    return {"ok": True, "tag": pack["asset"]["tag"], "window": tel[0]["window"] if tel else None,
            "trends": [{k: t[k] for k in ("id", "metric", "first_day_mean", "last_day_mean", "change_pct",
                                         "slope_per_day", "healthy_peer_mean")} for t in tel]}


def recent_events(key: str, limit: int = 10) -> dict:
    """Alarm/event counts by code and the most recent issue-related and important events (EVT-ids)."""
    pack, msg = _pack(key)
    if pack is None:
        return {"ok": False, "message": msg}
    ev = pack["events"]
    return {"ok": True, "tag": pack["asset"]["tag"], "total": ev["total"], "counts_by_code": ev["counts_by_code"],
            "issue_code_events": ev["issue_code_events"][:limit],
            "other_important_events": ev["other_important_events"][:limit]}


def maintenance_history(key: str) -> dict:
    """Recent work orders (WO-ids), similar historical cases (CASE-ids) and data-quality issues (DQ-ids)."""
    pack, msg = _pack(key)
    if pack is None:
        return {"ok": False, "message": msg}
    return {"ok": True, "tag": pack["asset"]["tag"], "work_orders": pack["work_orders"],
            "similar_cases": pack["cases"], "data_quality": pack["data_quality"],
            "documents": [{"doc_id": d["doc_id"], "title": d["title"]} for d in pack["documents"]]}


def nearby(key: str, radius_m: float = 1.5) -> dict:
    """Objects within radius_m (x-z, to their boxes) of an asset/object: mapped assets with status +
    counts of other objects by label."""
    n = resolve(key)
    if n is None:
        return {"ok": False, "message": f"Unknown asset/object '{key}'."}
    r = radius_m / cfg()["scene"]["units_to_m"]
    lo, hi = np.array(n["bbox_min"])[[0, 2]], np.array(n["bbox_max"])[[0, 2]]
    mapped, others = [], {}
    for m in scene()["objects"]:
        if m["kind"] != "object" or m["id"] == n["id"]:
            continue
        mlo, mhi = np.array(m["bbox_min"])[[0, 2]], np.array(m["bbox_max"])[[0, 2]]
        gap = np.maximum(np.maximum(mlo - hi, lo - mhi), 0)
        if float(np.linalg.norm(gap)) <= r:
            if m.get("asset"):
                mapped.append({**summary(m), "distance_m": u2m(float(np.linalg.norm(gap)))})
            else:
                others[m["label"]] = others.get(m["label"], 0) + 1
    mapped.sort(key=lambda s: s["distance_m"])
    return {"ok": True, "center": key, "radius_m": radius_m, "assets": mapped,
            "other_objects_by_label": dict(sorted(others.items(), key=lambda kv: -kv[1]))}


def route_to(key: str) -> dict:
    """Collision-free walking route from the door to an asset/object (length, clearance)."""
    from pipeline.tour import WalkPlanner
    n = resolve(key)
    if n is None:
        return {"ok": False, "message": f"Unknown asset/object '{key}'."}
    with open(out_path(cfg(), "path.json"), encoding="utf-8") as f:
        start = json.load(f)["start"]
    plan = WalkPlanner(cfg()).to_object(start[0], start[2], n)
    if not plan["ok"]:
        return {"ok": False, "message": plan["message"]}
    return {"ok": True, "to": key, "from": "door", "length_m": u2m(plan["length"]),
            "min_clearance_m": u2m(plan["min_clearance"]), "asset_record": bool(n.get("asset"))}


def plan_tour(stops: list[str] | None = None) -> dict:
    """Inspection tour door -> stops -> faulty target. stops=None/[] = automatic from Excel status."""
    from pipeline.tour import plan_tour as _plan
    t = _plan(cfg(), list(stops) if stops else [])
    return {"ok": True, "mode": t["mode"], "length_m": u2m(t["length"]),
            "min_clearance_m": u2m(t["min_clearance"]),
            "route": [{"tag": s["tag"], "kind": s["kind"], "condition_state": s["condition_state"],
                       "health_score": s["health_score"], "issue_code": s["issue_code"]} for s in t["stops"]],
            "skipped": t["skipped"]}


def draft_work_order(key: str) -> dict:
    """Draft (never submitted) work order from Excel status and evidence; no LLM, no cost."""
    pack, msg = _pack(key)
    if pack is None:
        return {"ok": False, "message": msg}
    a, s = pack["asset"], pack["current_state"]
    moved = sorted((t for t in pack["telemetry"] if t["change_pct"] and abs(t["change_pct"]) >= 5),
                   key=lambda t: -abs(t["change_pct"]))
    ev = pack["events"]["issue_code_events"][:3]
    return {"ok": True, "status": "DRAFT - requires review; follow site procedures, LOTO and permits",
            "title": f"Verify {s['issue_code']} on {a['tag']}", "asset": f"{a['tag']} ({a['asset_id']}, {a['asset_class']})",
            "priority": s["urgency"], "recommended_role": "field_technician",
            "description": f"{s['issue_summary']}. Next check: {s['recommended_next_check']}.",
            "evidence_ids": [t["id"] for t in moved[:3]] + [e["event_id"] for e in ev] + [s["knowledge_doc_id"],
                                                                                        "DOC-FIELD-SAFETY"]}


def diagnose(key: str, question: str | None = None, allow_paid: bool = False) -> dict:
    """Grounded diagnosis with validated citations. Default allow_paid=False: cached answer for the same
    evidence, else rule template (free). allow_paid=True may spend 1 request from the team budget."""
    from copilot.agent import diagnose as _diag, to_markdown
    n = resolve(key)
    if n is None or not n.get("asset"):
        return {"ok": False, "message": f"'{key}' is not an asset with Excel data."}
    d = _diag(n["asset"]["tag"], question, cfg(), allow_paid=allow_paid)
    cc = d["citation_check"]
    return {"ok": True, "tag": n["asset"]["tag"], "source": d.get("meta", {}).get("source"),
            "confidence": d.get("confidence"), "citations_valid": len(cc["cited_ids"]),
            "unknown_ids_removed": cc["unknown_ids_removed"], "markdown": to_markdown(d)}


def budget_status() -> dict:
    """Paid LLM requests used vs the team's hard cap (ledger)."""
    from copilot.llm import usage_summary
    return {"ok": True, **usage_summary(cfg()["copilot"])}


# ---------- viewer control (running viewer polls out/viewer_cmd.jsonl) ----------

def viewer_running(max_age_s: float = 5.0) -> bool:
    p = out_path(cfg(), "viewer_heartbeat")
    return p.exists() and time.time() - p.stat().st_mtime < max_age_s


def viewer_command(cmd: str, wait_s: float = 4.0, **args: Any) -> dict:
    """Send a command to the running viewer and wait briefly for its result."""
    if not viewer_running():
        return {"ok": False, "message": "The viewer is not running (start it with: python -m viewer.app)."}
    cid = uuid.uuid4().hex[:8]
    with open(out_path(cfg(), "viewer_cmd.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": cid, "cmd": cmd, "at": time.time(), **args}) + "\n")
    res_p = out_path(cfg(), "viewer_cmd_results.jsonl")
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if res_p.exists():
            for line in res_p.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("id") == cid:
                    return r
        time.sleep(0.2)
    return {"ok": True, "message": "Command sent; the viewer has not confirmed yet."}


def show_in_viewer(key: str) -> dict:
    """Select an asset/object in the viewer and fly the camera to it."""
    return viewer_command("show", target=key)


def walk_in_viewer(key: str) -> dict:
    """Make the worker in the viewer walk to an asset/object (Go to equipment mode)."""
    return viewer_command("walk", target=key)


def tour_in_viewer(stops: list[str] | None = None) -> dict:
    """Start an inspection tour in the viewer (stops=None/[] = automatic from Excel), ends at the faulty target."""
    return viewer_command("tour", stops=list(stops) if stops else [], wait_s=15.0)
