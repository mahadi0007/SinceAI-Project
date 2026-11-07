"""Readable asset card from the Excel evidence pack (copilot.evidence.build_evidence), for the viewer panel.

No LLM: every line is a field or a simple computation (rounding, % change, dates) from the pack.
Sections: header (status, location), readings vs healthy peers, alarms & events, maintenance,
what to check (recommended check + similar past cases + safety), reference (docs, data quality, 3D mapping).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

# metric -> (display name, unit, decimals)
METRICS: dict[str, tuple[str, str, int]] = {
    "vibration_mm_s": ("Vibration", "mm/s", 2),
    "bearing_temp_c": ("Bearing temperature", "°C", 1),
    "motor_current_a": ("Motor current", "A", 1),
    "speed_rpm": ("Speed", "rpm", 0),
    "process_pressure_bar": ("Process pressure", "bar", 2),
    "process_flow_pct": ("Process flow", "%", 1),
    "pneumatic_supply_bar": ("Pneumatic supply", "bar", 2),
    "position_setpoint_pct": ("Valve setpoint", "%", 1),
    "position_actual_pct": ("Valve position", "%", 1),
    "load_for_opening_pct": ("Load to open", "%", 1),
    "stiction_index": ("Stiction index", "", 2),
    "signal_quality_pct": ("Signal quality", "%", 1),
    "sensor_bias": ("Sensor bias", "", 2),
}
STATUS = {"attention_required": ("🔴", "Attention required"), "degraded": ("🟠", "Degraded"),
          "watch": ("🟡", "Watch"), "healthy": ("🟢", "Healthy")}
URGENCY = {"high": "🔺 High", "medium": "🔸 Medium", "low": "▫️ Low"}
FLAT_PCT = 3.0              # |change| below this reads as steady
OFF_PEER_PCT = 10.0         # "now" this far from healthy peers gets a warning mark


def _words(code: str) -> str:
    """'CAVITATION_RISK' / 'service_completed' -> 'Cavitation risk' / 'Service completed'."""
    t = str(code).replace("_", " ").strip().lower()
    return t[:1].upper() + t[1:]


def _dt(v: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(v)[:19])
    except ValueError:
        return None


def _day(v: Any) -> str:
    d = _dt(v)
    return f"{d.day} {d:%b %Y}" if d else "—"


def _stamp(v: Any) -> str:
    d = _dt(v)
    return f"{d.day} {d:%b %H:%M}" if d else "—"


def _num(x: Any, dec: int) -> str:
    return "—" if x is None else f"{float(x):,.{dec}f}"


def _reading_rows(telemetry: list[dict]) -> list[str]:
    rows = []
    for t in sorted(telemetry, key=lambda t: -abs(t.get("change_pct") or 0)):
        name, unit, dec = METRICS.get(t["metric"], (_words(t["metric"]), "", 2))
        u = f" {unit}" if unit and unit != "%" else unit
        ch = t.get("change_pct") or 0.0
        arrow = "≈" if abs(ch) < FLAT_PCT else ("▲" if ch > 0 else "▼")
        peer, now = t.get("healthy_peer_mean"), t.get("last_day_mean")
        off = (peer not in (None, 0) and now is not None and abs(now - peer) / abs(peer) * 100 >= OFF_PEER_PCT)
        rows.append(f"| {'⚠️ ' if off else ''}{name} | {_num(t.get('first_day_mean'), dec)} → "
                    f"**{_num(now, dec)}**{u} | {arrow} {ch:+.0f}% | {_num(peer, dec)}{u} |")
    return rows


def asset_card(pack: dict, today: date | None = None, compact: bool = False) -> str:
    """Summary card. compact=True: top readings, one-line events/maintenance, no reference (the viewer
    shows the full tables from asset_details() in collapsible sections below it)."""
    a, s, ev = pack["asset"], pack["current_state"], pack["events"]
    today = today or date.today()
    dot, state = STATUS.get(s["condition_state"], ("⚪", _words(s["condition_state"])))
    md = [f"### {dot} {a['tag']} · {_words(a['asset_class'])}",
          f"{a.get('area_name', '')} · {_words(a.get('process_context', ''))} · criticality "
          f"{a.get('criticality', '—')}",
          "",
          "| Health | Status | Urgency |", "|---|---|---|",
          f"| **{s['health_score']}**/100 | {state} | {URGENCY.get(s['urgency'], _words(s['urgency']))} |",
          "",
          f"**{_words(s['issue_code'])}**: {s['issue_summary']}"]

    if pack.get("telemetry"):
        md += ["", "#### 📈 Readings · last 3 days", "| Reading | Then → now | Trend | Healthy peers |",
               "|---|---|---|---|", *_reading_rows(pack["telemetry"])[:3 if compact else None],
               "", "_⚠️ = now more than 10% away from healthy equipment of the same type._"]

    counts = sorted(ev.get("counts_by_code", {}).items(), key=lambda kv: -kv[1])
    if counts:
        md += ["", f"#### 🔔 Alarms & events · {ev.get('total', sum(c for _, c in counts))} in the log",
               " · ".join(f"**{c}** {_words(k).lower()}" for k, c in counts)]
        recent = sorted(ev.get("issue_code_events", []) + ev.get("other_important_events", []),
                        key=lambda e: str(e["timestamp"]), reverse=True)[:3]
        if recent and not compact:
            md += ["", "| When | Event | Severity | Status |", "|---|---|---|---|"]
            md += [f"| {_stamp(e['timestamp'])} | {_words(e['event_code'])} ({_words(e['event_type']).lower()}) | "
                   f"{_words(e['severity'])} | {_words(e['status'])} |" for e in recent]

    md += ["", "#### 🔧 Maintenance"]
    last = _dt(a.get("last_service_date"))
    if last and a.get("service_interval_days"):
        due = (last + timedelta(days=int(a["service_interval_days"]))).date()
        days = (due - today).days
        when = f"due in {days} days" if days >= 0 else f"**overdue by {-days} days**"
        md.append(f"Last service {_day(last)} · every {a['service_interval_days']} days · next {_day(due)} ({when})")
    wos = sorted(pack.get("work_orders", []), key=lambda w: str(w["opened_at"]), reverse=True)[:3]
    if wos and not compact:
        md += ["", "| Opened | Work | Result | Downtime |", "|---|---|---|---|"]
        md += [f"| {_day(w['opened_at'])} | {_words(w['action'])} ({w['maintenance_type']}) | "
               f"{_words(w['result'])} | {w.get('downtime_h', 0):g} h |" for w in wos]

    md += ["", "#### ✅ What to check", f"1. **{_words(s['recommended_next_check'])}** (recommended for this issue)"]
    seen = {s["recommended_next_check"].lower()}
    for c in pack.get("cases", []):
        chk = c["recommended_next_check"]
        if chk.lower() not in seen:
            seen.add(chk.lower())
            md.append(f"{len(seen)}. {_words(chk)} (past case: {c['historical_interpretation'].lower()})")
    md.append("\n🦺 _Follow site procedures, LOTO and permits before any physical work._")

    ref = [f"📄 {d['title']}" for d in pack.get("documents", []) if d.get("scope") == a["asset_class"]]
    dq = pack.get("data_quality", {}).get("issues", [])
    if dq:
        ref.append("⚠️ Data quality: " + ", ".join(f"{_words(q['issue_type']).lower()} ({q['severity']})" for q in dq))
    sc = pack.get("scene")
    if sc:
        ref.append(f"📍 3D: {sc['object_id']} · {_words(sc['mapping_method']).lower()} · "
                   f"mapping confidence {float(sc['mapping_confidence']):.2f}")
    if ref and not compact:
        md += ["", "#### 📚 Reference", "  \n".join(ref)]
    if compact:
        md += ["", "_Full tables below: asset details, all readings, every event, maintenance history, "
                   "past cases, documents, data quality, 3D mapping._"]
    return "\n".join(md)


# ---------- full detail tables (collapsible sections under the summary card) ----------

DETAIL_SECTIONS = ("Asset details", "Readings · last 3 days", "Alarms & events", "Maintenance history",
                   "Similar past cases", "Documents", "Data quality", "3D object & mapping")


def _rows(df: Any, tag: str) -> Any:
    return df[df["tag"] == tag] if "tag" in df.columns else df.iloc[0:0]


def _table(header: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_None recorded._"
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join("—" if v is None or str(v) in ("nan", "NaT", "") else str(v) for v in r) + " |"
            for r in rows]
    return "\n".join(out)


def asset_details(pack: dict, xl: dict, node: dict | None = None, u2m: float = 1.0,
                  today: date | None = None) -> dict[str, str]:
    """Every record about one asset, as markdown tables keyed by DETAIL_SECTIONS.
    xl = copilot.evidence.load_pack(...) (all Excel sheets); node = the scene-graph object (3D info)."""
    a, s, tag = pack["asset"], pack["current_state"], pack["asset"]["tag"]
    today = today or date.today()
    am = _rows(xl["Asset_Master"], tag)
    am = am.iloc[0].to_dict() if len(am) else {}
    out: dict[str, str] = {}

    installed = _dt(am.get("installation_date"))
    age = f" ({(today - installed.date()).days / 365.25:.1f} years)" if installed else ""
    last = _dt(a.get("last_service_date"))
    due = (last + timedelta(days=int(a["service_interval_days"]))).date() if last and a.get("service_interval_days") else None
    out["Asset details"] = _table(["Field", "Value"], [
        ["Tag / asset ID", f"**{tag}** / {a['asset_id']}"],
        ["Type", _words(a["asset_class"])],
        ["Area", f"{a.get('area_name')} ({am.get('area_id', '—')})"],
        ["Process", _words(a.get("process_context", ""))],
        ["Criticality", a.get("criticality")],
        ["Installed", _day(installed) + age],
        ["Last service", _day(last)],
        ["Service interval", f"{a.get('service_interval_days')} days → next {_day(due) if due else '—'}"
                             + (f" ({(due - today).days:+d} days)" if due else "")],
        ["Connectivity", am.get("connectivity_profile")],
        ["QR / NFC", f"{am.get('qr_id', '—')} / {am.get('nfc_id', '—')}"],
        ["Health · status", f"{s['health_score']}/100 · {_words(s['condition_state'])}"],
        ["Issue", f"{_words(s['issue_code'])}: {s['issue_summary']}"],
        ["Urgency", _words(s["urgency"])],
        ["Recommended check", _words(s["recommended_next_check"])],
        ["Model confidence", f"{float(s['model_confidence']):.2f}"],
        ["State last updated", (lambda d: f"{d.day} {d:%b %Y %H:%M}" if d else "—")(_dt(s.get("last_updated")))],
        ["Data classification", am.get("data_classification")],
    ])

    rd = []
    for t in sorted(pack.get("telemetry", []), key=lambda t: -abs(t.get("change_pct") or 0)):
        name, unit, dec = METRICS.get(t["metric"], (_words(t["metric"]), "", 2))
        rd.append([f"{name}{f' ({unit})' if unit else ''}",
                   f"{_num(t.get('first_day_mean'), dec)} → **{_num(t.get('last_day_mean'), dec)}**",
                   f"{_num(t.get('min'), dec)}–{_num(t.get('max'), dec)}",
                   f"{(t.get('change_pct') or 0):+.1f}% ({float(t.get('slope_per_day') or 0):+.{dec}f}/day)",
                   _num(t.get("healthy_peer_mean"), dec)])
    window = pack["telemetry"][0].get("window", "") if pack.get("telemetry") else ""
    out["Readings · last 3 days"] = (_table(["Reading", "Day 1 → day 3 avg", "Min–max", "Change", "Healthy peers"], rd)
                                     + (f"\n\n_{len(rd)} signals · 30-min samples · {window.replace('T', ' ')}_"
                                        if window else ""))

    ev = _rows(xl["Alarms_Events"], tag).sort_values("timestamp", ascending=False)
    counts = ev.groupby("event_code").size().sort_values(ascending=False)
    out["Alarms & events"] = (
        _table(["Event", "Count"], [[_words(k), int(v)] for k, v in counts.items()]) + "\n\n" +
        ("_Reported by: " + " · ".join(f"{_words(k).lower()} ×{int(v)}" for k, v in
                                       ev.groupby("source").size().sort_values(ascending=False).items()) + "_\n\n"
         if len(ev) else "") +
        _table(["When", "Event", "Type", "Severity · status"],
               [[f"{_stamp(r.timestamp)}", _words(r.event_code), _words(r.event_type),
                 f"{_words(r.severity)} · {str(r.status).lower()}"] for r in ev.itertuples()]))

    wo = _rows(xl["Maintenance_History"], tag).sort_values("opened_at", ascending=False)
    out["Maintenance history"] = _table(
        ["Opened", "Work (type, role)", "Priority", "Result", "Work / downtime"],
        [[_day(r.opened_at), f"{_words(r.action)} ({str(r.maintenance_type).lower()}, {_words(r.role).lower()})",
          _words(r.priority), _words(r.result), f"{r.duration_h:g} h / {r.downtime_h:g} h"] for r in wo.itertuples()])

    out["Similar past cases"] = _table(
        ["Case", "Issue", "What it was", "Check", "Severity"],
        [[c["case_id"], _words(c["historical_issue_code"]), c["historical_interpretation"],
          _words(c["recommended_next_check"]), _words(c["severity"])] for c in pack.get("cases", [])])

    docs = [[d["doc_id"], d["title"], _words(d.get("scope", ""))] for d in pack.get("documents", [])]
    rag = _rows(xl.get("RAG_Corpus_Index", xl["Asset_Master"].iloc[0:0]), tag)
    docs += [[r.doc_id, f"{r.file_name} ({_words(r.document_role).lower()})", _words(r.scope)] for r in rag.itertuples()]
    out["Documents"] = _table(["Doc", "Title", "Scope"], docs)

    dq = _rows(xl["Data_Quality"], tag)
    out["Data quality"] = _table(
        ["Issue", "Type", "Severity", "Sheet", "Handling"],
        [[r.issue_id, _words(r.issue_type), _words(r.severity), str(r.source_sheet).replace('_', ' '),
          _words(r.recommended_handling)]
         for r in dq.itertuples()]) + (f"\n\n_Confidence cap for answers: "
                                        f"{pack['data_quality'].get('confidence_cap', 1):.2f}_")

    sc, rows3d = pack.get("scene") or {}, []
    if node:
        e = [x * u2m for x in node["extent"]]
        conf = node.get("label_confidence")
        rows3d += [["3D object", node["id"]],
                   ["Recognised as", f"{node.get('label')} ({node.get('label_source', 'geometry')}"
                                     + (f", {conf:.2f}" if conf is not None else "") + ")"],
                   ["Size", f"{e[0]:.2f} × {e[1]:.2f} × {e[2]:.2f} m"],
                   ["Points in scan", f"{node.get('n_points', 0):,}"]]
    if sc:
        rows3d += [["Mapped by", f"{_words(sc.get('mapping_method', ''))} (confidence {float(sc['mapping_confidence']):.2f})"]]
    an = _rows(xl["Scene_Anchors"], tag)
    if len(an):
        r = an.iloc[0]
        rows3d += [["Excel anchor (placeholder)", f"{r.scene_anchor_id} · {r.scene_path} · {_words(r.mapping_status).lower()}"],
                   ["Identify on site by", r.identity_methods]]
    out["3D object & mapping"] = _table(["Field", "Value"], rows3d)
    return out
