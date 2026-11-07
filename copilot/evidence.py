"""Evidence pack for one asset: everything the copilot may cite, each item with a stable ID.

IDs: TEL-<tag>-<metric> (telemetry trend), EVT-xxxxx (Alarms_Events), WO-xxxxx
(Maintenance_History), CASE-xxx (Historical_Service_Cases), DOC-... (Knowledge_Docs),
DQ-xxxx (Data_Quality). Output: out/evidence_<tag>.json
Run: python -m copilot.evidence [TAG]
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from pipeline.common import load_config, out_path, repo_path

Cfg = dict[str, Any]


@lru_cache(maxsize=1)
def load_pack(xlsx: str) -> dict[str, pd.DataFrame]:
    return pd.read_excel(xlsx, sheet_name=None)


def _py(v: Any) -> Any:
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    return [{c: _py(r[c]) for c in cols if c in df.columns} for _, r in df.iterrows()]


def resolve_asset(xl: dict[str, pd.DataFrame], key: str) -> pd.Series:
    am = xl["Asset_Master"]
    hit = am[(am["tag"] == key) | (am["asset_id"] == key)]
    if hit.empty:
        raise KeyError(f"unknown asset {key}")
    return hit.iloc[0]


# ---------- telemetry ----------

def metric_trend(series: pd.Series, per_day: int) -> dict[str, float]:
    y = series.to_numpy(dtype=float)
    t = np.arange(len(y)) / per_day
    slope = float(np.polyfit(t, y, 1)[0]) if len(y) > 2 else 0.0
    first, last = float(np.mean(y[:per_day])), float(np.mean(y[-per_day:]))
    return {"first_day_mean": round(first, 3), "last_day_mean": round(last, 3),
            "change_pct": round(100 * (last - first) / first, 1) if first else None,
            "slope_per_day": round(slope, 3), "min": round(float(y.min()), 3),
            "max": round(float(y.max()), 3), "n_samples": int(len(y))}


def telemetry_trends(xl: dict[str, pd.DataFrame], asset: pd.Series, ec: Cfg) -> list[dict]:
    tel = xl["Telemetry_Sample_3d"].sort_values("timestamp")
    mine = tel[tel["asset_id"] == asset["asset_id"]]
    cs = xl["Current_State"]
    healthy_ids = cs[(cs["asset_class"] == asset["asset_class"]) &
                     (cs["condition_state"] == "healthy")]["asset_id"]
    peers = tel[tel["asset_id"].isin(healthy_ids)]
    per_day = ec["samples_per_day"]
    out = []
    for m in ec["metrics"]:
        s = mine[m].dropna()
        if s.empty:
            continue
        tr = metric_trend(s, per_day)
        base = peers.groupby("asset_id")[m].apply(lambda v: v.dropna().tail(per_day).mean()).dropna()
        tr["healthy_peer_mean"] = round(float(base.mean()), 3) if len(base) else None
        tr["healthy_peers"] = sorted(cs.set_index("asset_id").loc[base.index, "tag"].tolist())
        tr.update({"id": f"TEL-{asset['tag']}-{m}", "metric": m,
                   "comparison": "mean of first 24 h vs mean of last 24 h of a 3-day, 30-min window; "
                                 "healthy_peer_mean = last 24 h of healthy same-class assets",
                   "window": f"{_py(mine['timestamp'].iloc[0])} .. {_py(mine['timestamp'].iloc[-1])}"})
        out.append(tr)
    return out


# ---------- events / work orders / cases / docs / data quality ----------

def recent_events(xl: dict[str, pd.DataFrame], asset: pd.Series, issue: str, ec: Cfg) -> dict:
    ev = xl["Alarms_Events"]
    mine = ev[ev["asset_id"] == asset["asset_id"]].sort_values("timestamp", ascending=False)
    issue_ev = mine[mine["event_code"] == issue].head(ec["n_issue_events"])
    other = mine[(mine["event_code"] != issue) &
                 mine["severity"].isin(ec["important_severities"])].head(ec["n_other_events"])
    cols = ["event_id", "timestamp", "event_type", "severity", "event_code", "status", "source"]
    return {"total": int(len(mine)),
            "counts_by_code": {k: int(v) for k, v in mine["event_code"].value_counts().items()},
            "issue_code_events": records(issue_ev, cols),
            "other_important_events": records(other, cols)}


def related_work_orders(xl: dict[str, pd.DataFrame], asset: pd.Series, ec: Cfg) -> list[dict]:
    mh = xl["Maintenance_History"]
    mine = mh[mh["asset_id"] == asset["asset_id"]].sort_values("opened_at", ascending=False)
    return records(mine.head(ec["n_work_orders"]),
                   ["work_order_id", "opened_at", "action", "maintenance_type", "priority",
                    "result", "duration_h", "downtime_h", "role"])


def similar_cases(xl: dict[str, pd.DataFrame], asset: pd.Series, issue: str, ec: Cfg) -> list[dict]:
    hc = xl["Historical_Service_Cases"]
    same = hc[hc["asset_class"] == asset["asset_class"]]
    exact = same[same["historical_issue_code"] == issue]
    pick = pd.concat([exact, same.drop(exact.index)]).head(ec["n_cases"])
    rows = records(pick, ["case_id", "historical_issue_code", "historical_interpretation",
                          "recommended_next_check", "safe_evidence_workflow", "severity"])
    for r in rows:
        r["match"] = "class+issue" if r["historical_issue_code"] == issue else "class"
    return rows


def knowledge_docs(xl: dict[str, pd.DataFrame], asset: pd.Series, ec: Cfg) -> list[dict]:
    kd = xl["Knowledge_Docs"]
    pick = kd[(kd["scope"] == asset["asset_class"]) | kd["doc_id"].isin(ec["always_docs"])]
    return records(pick, ["doc_id", "title", "scope", "content"])


def data_quality(xl: dict[str, pd.DataFrame], asset: pd.Series, ec: Cfg) -> dict:
    dq = xl["Data_Quality"]
    mine = dq[dq["asset_id"] == asset["asset_id"]]
    penalty = min(sum(ec["dq_penalty"].get(s, 0.0) for s in mine["severity"]), ec["dq_penalty_max"])
    return {"issues": records(mine, ["issue_id", "issue_type", "severity", "source_sheet",
                                     "recommended_handling"]),
            "confidence_cap": round(1.0 - penalty, 2)}


def scene_mapping(cfg: Cfg, tag: str) -> dict | None:
    p = out_path(cfg, "scene_graph.json")
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        sg = json.load(f)
    for n in sg["objects"]:
        if n["asset"] and n["asset"]["tag"] == tag:
            return {"id": f"MAP-{tag}", "object_id": n["id"], "label": n["label"], "center": n["center"],
                    "mapping_method": n["asset"]["mapping_method"],
                    "mapping_confidence": n["asset"]["mapping_confidence"]}
    return None


# ---------- pack ----------

def allowed_ids(pack: dict) -> list[str]:
    ids = [t["id"] for t in pack["telemetry"]]
    ids += [e["event_id"] for k in ("issue_code_events", "other_important_events") for e in pack["events"][k]]
    ids += [w["work_order_id"] for w in pack["work_orders"]]
    ids += [c["case_id"] for c in pack["cases"]] + [d["doc_id"] for d in pack["documents"]]
    ids += [q["issue_id"] for q in pack["data_quality"]["issues"]]
    ids += [pack["asset"]["asset_id"], pack["asset"]["tag"]]
    if pack["scene"]:
        ids.append(pack["scene"]["id"])
    return sorted(set(ids))


def build_evidence(key: str, cfg: Cfg | None = None) -> dict:
    cfg = cfg or load_config()
    ec = cfg["evidence"]
    xl = load_pack(str(repo_path(cfg["paths"]["xlsx"])))
    a = resolve_asset(xl, key)
    st = xl["Current_State"].set_index("asset_id").loc[a["asset_id"]]
    asset = {k: _py(a[k]) for k in ("asset_id", "tag", "asset_class", "area_name", "process_context",
                                    "criticality", "last_service_date", "service_interval_days")}
    state = {k: _py(st[k]) for k in ("health_score", "condition_state", "issue_code", "issue_summary",
                                     "urgency", "recommended_next_check", "model_confidence",
                                     "last_updated", "knowledge_doc_id")}
    pack = {"asset": asset, "current_state": state, "scene": scene_mapping(cfg, asset["tag"]),
            "telemetry": telemetry_trends(xl, a, ec),
            "events": recent_events(xl, a, state["issue_code"], ec),
            "work_orders": related_work_orders(xl, a, ec),
            "cases": similar_cases(xl, a, state["issue_code"], ec),
            "documents": knowledge_docs(xl, a, ec),
            "data_quality": data_quality(xl, a, ec)}
    pack["allowed_ids"] = allowed_ids(pack)
    return pack


def status_line(key: str, cfg: Cfg | None = None) -> str:
    """One-line template status for assets passed while walking (no LLM)."""
    cfg = cfg or load_config()
    xl = load_pack(str(repo_path(cfg["paths"]["xlsx"])))
    a = resolve_asset(xl, key)
    s = xl["Current_State"].set_index("asset_id").loc[a["asset_id"]]
    return (f"{a['tag']} ({a['asset_class']}): {s['condition_state']}, health {s['health_score']}, "
            f"{s['issue_code']} — next: {s['recommended_next_check']} [{s['knowledge_doc_id']}]")


def save(pack: dict, cfg: Cfg) -> str:
    p = out_path(cfg, f"evidence_{pack['asset']['tag']}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(pack, f, indent=1, default=str)
    return str(p)


if __name__ == "__main__":
    cfg = load_config()
    tag = sys.argv[1] if len(sys.argv) > 1 else cfg["mapping"]["target_tag"]
    pk = build_evidence(tag, cfg)
    print(f"[evidence] {tag}: {len(pk['telemetry'])} metrics, {pk['events']['total']} events "
          f"({len(pk['events']['issue_code_events'])} issue events cited), {len(pk['work_orders'])} WOs, "
          f"{len(pk['cases'])} cases, {len(pk['documents'])} docs, {len(pk['data_quality']['issues'])} DQ "
          f"issues (confidence cap {pk['data_quality']['confidence_cap']})")
    for t in pk["telemetry"]:
        print(f"  {t['id']:38s} {t['first_day_mean']:>9} -> {t['last_day_mean']:>9} "
              f"({t['change_pct']:+}%)  healthy peers {t['healthy_peer_mean']}")
    print(f"  event codes: {pk['events']['counts_by_code']}")
    print(f"[out] {save(pk, cfg)}")
    print(status_line(tag, cfg))
