"""Pre-inspection brief: everything a technician reads before walking to one asset, built remotely.

Sections: header, location and route (+ top-down PNG), along the way, diagnosis (cached AI answer or
template), what to verify on site, safety, bring / prepare (static checklist from config.yaml), draft
work order, evidence appendix. No LLM calls: the diagnosis comes from copilot.offline.
Outputs: out/briefs/<tag>.md, <tag>.html (self-contained, printable), <tag>_route.png
Run: python -m copilot.brief P-002      python -m copilot.brief --all
"""
from __future__ import annotations

import base64
import html
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from copilot import tools as T
from copilot.agent import describe_ids
from copilot.card import STATUS, URGENCY, _reading_rows, _words
from copilot.evidence import build_evidence
from copilot.offline import offline_diagnosis
from pipeline.common import load_config, out_path

Cfg = dict[str, Any]
Block = tuple  # ("p", text) | ("list", [text]) | ("olist", [text]) | ("check", [text]) | ("table", head, rows) | ("img", file)

DQ_ADVICE = {
    "sensor_quality_drop": "signal quality low: confirm readings with a handheld meter",
    "timestamp_jitter": "timestamps unreliable: confirm the event order with the control room",
    "scene_mapping_low_confidence": "3D location uncertain: confirm the asset by its tag plate / QR before work",
    "stale_document_link": "document link may be outdated: use the current site procedure",
    "duplicate_event": "duplicate events: alarm counts may be inflated",
    "missing_value": "missing values: some readings are incomplete",
    "unit_metadata_missing": "units missing: confirm units on the local display",
}


# ---------- route ----------

def _box_gap_xz(n: dict, pts: np.ndarray) -> float:
    lo, hi = np.array(n["bbox_min"])[[0, 2]], np.array(n["bbox_max"])[[0, 2]]
    gap = np.maximum(np.maximum(lo - pts, pts - hi), 0)
    return float(np.linalg.norm(gap, axis=1).min())


def route_points(cfg: Cfg, node: dict) -> np.ndarray | None:
    from pipeline.tour import WalkPlanner
    with open(out_path(cfg, "path.json"), encoding="utf-8") as f:
        start = json.load(f)["start"]
    plan = WalkPlanner(cfg).to_object(start[0], start[2], node)
    return np.array(plan["points"]) if plan["ok"] else None


def along_the_way(cfg: Cfg, node: dict, pts: np.ndarray | None) -> list[dict]:
    """Non-healthy mapped assets near the route or near the target (tools.nearby)."""
    bc, u2m = cfg.get("brief", {}), cfg["scene"]["units_to_m"]
    near = T.nearby(node["id"], bc.get("nearby_radius_m", 1.5))
    hits = {a["tag"]: {**a, "where": f"{a['distance_m']:.1f} m from the target"}
            for a in near.get("assets", []) if a["condition_state"] != "healthy"}
    if pts is not None:
        r = bc.get("route_radius_m", 1.0) / u2m
        for m in T.scene()["objects"]:
            a = m.get("asset")
            if not a or m["id"] == node["id"] or a["condition_state"] == "healthy" or a["tag"] in hits:
                continue
            d = _box_gap_xz(m, pts[:, [0, 2]])
            if d <= r:
                hits[a["tag"]] = {**T.summary(m), "where": f"{d * u2m:.1f} m from the route"}
    order = {"attention_required": 0, "degraded": 1, "watch": 2}
    return sorted(hits.values(), key=lambda a: (order.get(a["condition_state"], 9), a["health_score"]))


def plot_route(cfg: Cfg, node: dict, pts: np.ndarray | None, path: Path) -> None:
    """Top-down navgrid + route + target box, styled like pipeline.tour.plot_tour."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from pipeline.tour import load_inputs
    grid, layers, _, _ = load_inputs(cfg)
    img = np.ones((grid.nz, grid.nx, 3))
    img[layers["floor"].T] = (0.80, 0.92, 0.80)
    img[(layers["inflated"] & layers["floor"]).T] = (0.98, 0.85, 0.65)
    img[layers["obstacles"].T] = (0.25, 0.25, 0.30)
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ext = (grid.x0, grid.x0 + grid.nx * grid.cell, grid.z0, grid.z0 + grid.nz * grid.cell)
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
    if pts is not None:
        ax.plot(pts[:, 0], pts[:, 2], "-", color="crimson", lw=2)
        ax.plot(pts[0, 0], pts[0, 2], "o", color="green", ms=11, label="start (door)")
    lo, hi = node["bbox_min"], node["bbox_max"]
    ax.add_patch(Rectangle((lo[0], lo[2]), hi[0] - lo[0], hi[2] - lo[2], fill=False, ec="red", lw=2.5))
    ax.annotate(node["asset"]["tag"], (hi[0], hi[2]), xytext=(6, 6), textcoords="offset points", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))
    ax.set_title(f"Route from the door to {node['asset']['tag']}" + ("" if pts is not None else " (no route found)"))
    ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_aspect("equal")
    if pts is not None:
        ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ---------- brief ----------

def checklist(cfg: Cfg, cls: str, issue: str) -> list[str]:
    cl = cfg.get("brief", {}).get("checklist", {})
    items = [*cl.get("ALL", []), *cl.get(cls, []), *cl.get(f"{cls}:{issue}", [])]
    return list(dict.fromkeys(items))


def _ids(ids: list[str]) -> str:
    return " ".join(f"`{i}`" for i in ids)


def build_brief(tag: str, cfg: Cfg | None = None) -> dict:
    """Brief as ordered sections of render-neutral blocks, plus the data behind them."""
    cfg = cfg or load_config()
    node = T.resolve(tag)
    if node is None or not node.get("asset"):
        raise KeyError(f"{tag} is not a mapped asset with Excel data")
    pack = build_evidence(node["asset"]["tag"], cfg)
    a, s, tag = pack["asset"], pack["current_state"], pack["asset"]["tag"]
    diag = offline_diagnosis(pack, cfg)
    route = T.route_to(tag)
    pts = route_points(cfg, node) if route.get("ok") else None
    allowed = set(pack["allowed_ids"])
    cited: list[str] = []

    def cite(ids: list[str]) -> str:
        ok = [i for i in ids if i in allowed]
        cited.extend(ok)
        return _ids(ok)

    dot, state = STATUS.get(s["condition_state"], ("", _words(s["condition_state"])))
    secs: list[tuple[str, list[Block]]] = []
    secs.append(("Header", [
        ("table", ["Tag", "Asset ID", "Class", "Status", "Urgency", "Health"],
         [[f"**{tag}**", a["asset_id"], _words(a["asset_class"]), f"{dot} {state}",
           URGENCY.get(s["urgency"], _words(s["urgency"])), f"{s['health_score']}/100"]]),
        ("p", f"{a.get('area_name', '')} · {_words(a.get('process_context', ''))} · criticality "
              f"{a.get('criticality', '—')} · **{_words(s['issue_code'])}**: {s['issue_summary']} {cite([a['asset_id']])}")]))

    loc: list[Block] = []
    if route.get("ok"):
        loc.append(("p", f"**{route['length_m']:.1f} m** walk from the door, minimum clearance "
                         f"{route['min_clearance_m']:.2f} m (collision-free route on the scanned floor)."))
    else:
        loc.append(("p", f"No route: {route.get('message', 'unknown')}"))
    if pack["scene"]:
        sc = pack["scene"]
        loc.append(("p", f"3D object {sc['object_id']}, mapped by {_words(sc['mapping_method']).lower()} "
                         f"(confidence {float(sc['mapping_confidence']):.2f}) {cite([sc['id']])}"))
    loc.append(("img", f"{tag}_route.png"))
    secs.append(("Location and route", loc))

    way = along_the_way(cfg, node, pts)
    secs.append(("Along the way", [
        ("table", ["Tag", "Class", "Status", "Health", "Issue", "Where"],
         [[w["tag"], _words(w["asset_class"]), _words(w["condition_state"]), w["health_score"],
           _words(w["issue_code"]), w["where"]] for w in way]) if way else
        ("p", "No other equipment needing attention near the route.")]))

    kind = "Cached AI answer (same evidence)" if diag["meta"]["kind"] == "cached_llm" else "Template (rules over the Excel evidence, no AI)"
    secs.append(("Diagnosis", [
        ("p", f"_Source: {kind}._ {diag.get('condition_summary', '')}"),
        ("p", "**Observations**"),
        ("list", [f"{o['text']} {cite(o['evidence_ids'])}" for o in diag.get("observations", [])]),
        ("p", "**Hypothesis**"),
        ("list", [f"{h['text']} (confidence {h.get('confidence')}) {cite(h.get('evidence_ids', []))}"
                  for h in diag.get("hypotheses", [])]),
        ("p", f"**Confidence {float(diag.get('confidence') or 0):.2f}**: {diag.get('confidence_rationale', '')}")]))

    moved = sorted(pack["telemetry"], key=lambda t: -abs(t.get("change_pct") or 0))[:3]
    rec = s["recommended_next_check"].lower()
    same = [c["case_id"] for c in pack["cases"] if c["match"] == "class+issue" and c["recommended_next_check"].lower() == rec]
    checks = [f"**{_words(rec)}** (recommended for {_words(s['issue_code']).lower()}"
              + (", also in past cases" if same else "") + f") {cite([s['knowledge_doc_id'], *same])}"]
    seen = {rec}
    for c in pack["cases"]:
        chk = c["recommended_next_check"].lower()
        if c["match"] == "class+issue" and chk not in seen:
            seen.add(chk)
            checks.append(f"{_words(chk)} (past case: {c['historical_interpretation'].lower()}) {cite([c['case_id']])}")
    secs.append(("What to verify on site", [
        ("olist", checks),
        ("p", "Key readings behind it:"),
        ("table", ["Reading", "Then → now", "Trend", "Healthy peers"],
         [[c.strip() for c in r.strip("|").split("|")] for r in _reading_rows(moved)]),
        ("p", "Readings: " + cite([t["id"] for t in moved]))]))

    safety_doc = next((d["content"] for d in pack["documents"] if d["doc_id"] == "DOC-FIELD-SAFETY"), "")
    safe: list[Block] = [("list", [
        f"{safety_doc} {cite(['DOC-FIELD-SAFETY'])}",
        "Follow site procedures: lockout-tagout (LOTO) and work permits apply before any physical work.",
        "Verify isolation before touching; confirm the asset identity on site (tag plate / QR).",
        "Verify and escalate: involve the responsible maintenance engineer if checks confirm the trend."])]
    dq = pack["data_quality"]["issues"]
    if dq:
        safe += [("p", f"**Data quality flags** (answer confidence capped at {pack['data_quality']['confidence_cap']:.2f}):"),
                 ("list", [f"{DQ_ADVICE.get(q['issue_type'], _words(q['issue_type']).lower())} "
                           f"({q['severity']}) {cite([q['issue_id']])}" for q in dq])]
    secs.append(("Safety", safe))

    secs.append(("Bring / prepare", [("p", "_Standard checklist from site configuration, not AI output._"),
                                     ("check", checklist(cfg, a["asset_class"], s["issue_code"]))]))

    wo = T.draft_work_order(tag)
    secs.append(("Draft work order", [
        ("p", f"**{wo['status']}**"),
        ("table", ["Title", "Priority", "Role"], [[wo["title"], wo["priority"], wo["recommended_role"]]]),
        ("p", f"{wo['description']} {cite(wo['evidence_ids'])}")]))

    ids = sorted(set(cited))
    desc = describe_ids(pack, ids)
    secs.append(("Evidence appendix", [("table", ["ID", "What it is"], [[f"`{i}`", desc.get(i, "")] for i in ids])]))
    return {"tag": tag, "title": f"Pre-inspection brief · {tag}", "date": date.today().isoformat(),
            "sections": secs, "cited_ids": ids, "allowed_ids": sorted(allowed),
            "diagnosis_source": diag["meta"]["source"], "route": route, "route_points": pts, "node": node}


# ---------- renderers ----------

def to_markdown(b: dict) -> str:
    md = [f"# {b['title']}", f"_{b['date']} · read before going on site · diagnosis: {b['diagnosis_source']}_"]
    for i, (name, blocks) in enumerate(b["sections"], 1):
        md += ["", f"## {i}. {name}"]
        for bl in blocks:
            if bl[0] == "p":
                md += ["", bl[1]]
            elif bl[0] in ("list", "olist", "check"):
                mark = {"list": "-", "check": "- [ ]"}.get(bl[0])
                md += [""] + [f"{mark or f'{k}.'} {t}" for k, t in enumerate(bl[1], 1)]
            elif bl[0] == "table":
                md += ["", "| " + " | ".join(bl[1]) + " |", "|" + "---|" * len(bl[1])]
                md += ["| " + " | ".join(str(c) for c in r) + " |" for r in bl[2]]
            elif bl[0] == "img":
                md += ["", f"![Route]({bl[1]})"]
    return "\n".join(md) + "\n"


def _inline(t: Any) -> str:
    t = html.escape(str(t))
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    return re.sub(r"(?<![\w])_(.+?)_(?![\w])", r"<i>\1</i>", t)


CSS = """body{font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;
color:#1b1f24;background:#fff}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:22px 0 6px;padding-bottom:3px;
border-bottom:2px solid #e3e6ea;break-after:avoid}table{border-collapse:collapse;width:100%;margin:6px 0}
th,td{border:1px solid #d5d9de;padding:4px 7px;text-align:left;vertical-align:top}th{background:#f2f4f6}
code{font-size:12px;background:#f2f4f6;padding:0 3px;border-radius:3px}img{max-width:100%;border:1px solid #d5d9de}
.sub{color:#5a6470}ul.check{list-style:none;padding-left:4px}ul.check li:before{content:"\\2610  "}
section{break-inside:avoid-page}@media print{body{margin:0;max-width:none}h2{margin-top:14px}}"""


def to_html(b: dict, image: bytes | None) -> str:
    out = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
           f"<title>{html.escape(b['title'])}</title><style>{CSS}</style></head><body>",
           f"<h1>{html.escape(b['title'])}</h1><div class='sub'>{b['date']} · read before going on site · "
           f"diagnosis: {html.escape(b['diagnosis_source'])}</div>"]
    for i, (name, blocks) in enumerate(b["sections"], 1):
        out.append(f"<section><h2>{i}. {html.escape(name)}</h2>")
        for bl in blocks:
            if bl[0] == "p":
                out.append(f"<p>{_inline(bl[1])}</p>")
            elif bl[0] in ("list", "olist", "check"):
                tag = "ol" if bl[0] == "olist" else "ul"
                cls = " class='check'" if bl[0] == "check" else ""
                out.append(f"<{tag}{cls}>" + "".join(f"<li>{_inline(t)}</li>" for t in bl[1]) + f"</{tag}>")
            elif bl[0] == "table":
                out.append("<table><tr>" + "".join(f"<th>{_inline(h)}</th>" for h in bl[1]) + "</tr>"
                           + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in bl[2])
                           + "</table>")
            elif bl[0] == "img" and image:
                out.append(f"<img alt='Route' src='data:image/png;base64,{base64.b64encode(image).decode()}'>")
        out.append("</section>")
    return "\n".join(out) + "</body></html>\n"


def write_brief(tag: str, cfg: Cfg | None = None) -> dict:
    """Build and write out/briefs/<tag>.md, .html and _route.png. Returns the paths and the markdown."""
    cfg = cfg or load_config()
    b = build_brief(tag, cfg)
    d = out_path(cfg, "briefs")
    d.mkdir(exist_ok=True)
    png = d / f"{b['tag']}_route.png"
    plot_route(cfg, b["node"], b["route_points"], png)
    md = to_markdown(b)
    (d / f"{b['tag']}.md").write_text(md, encoding="utf-8")
    (d / f"{b['tag']}.html").write_text(to_html(b, png.read_bytes()), encoding="utf-8")
    return {"ok": True, "tag": b["tag"], "md": str(d / f"{b['tag']}.md"), "html": str(d / f"{b['tag']}.html"),
            "png": str(png), "markdown": md, "cited_ids": b["cited_ids"]}


def attention_tags() -> list[str]:
    return [a["tag"] for a in T.search_assets()["assets"] if a["condition_state"] != "healthy"]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Pre-inspection brief (no LLM calls) -> out/briefs/")
    ap.add_argument("tag", nargs="?", help="asset tag, e.g. P-002")
    ap.add_argument("--all", action="store_true", help="every mapped asset that is not healthy")
    args = ap.parse_args()
    if not args.tag and not args.all:
        ap.error("give a tag or --all")
    c = load_config()
    for t in attention_tags() if args.all else [args.tag]:
        try:
            r = write_brief(t, c)
            print(f"[brief] {r['tag']}: {r['html']}  ({len(r['cited_ids'])} cited ids)")
        except Exception as e:  # noqa: BLE001 - one bad asset must not stop --all
            print(f"[brief] {t}: failed: {type(e).__name__}: {e}", file=sys.stderr)
