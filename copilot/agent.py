"""Grounded diagnosis: one LLM call over the evidence pack, then citation validation.

System prompt = DOC-RAG-GUIDE + DOC-FIELD-SAFETY verbatim + output contract.
Every cited ID must exist in the evidence pack; unknown IDs are removed and reported.
Falls back to a deterministic template if no API key / the call fails, so the demo never breaks.
Output: out/diagnosis_<tag>.json
Run: python -m copilot.agent [TAG] ["question"]
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from typing import Any

from copilot.evidence import build_evidence, save as save_evidence
from copilot.llm import LLMUnavailable, generate
from pipeline.common import load_config, out_path

Cfg = dict[str, Any]
ID_RE = re.compile(r"\b(?:TEL-[A-Z]+-\d+-[a-z_]+|EVT-\d+|WO-\d+|CASE-\d+|DQ-\d+|DOC-[A-Z0-9_-]*[A-Z0-9])\b")

OUTPUT_CONTRACT = """Return ONLY a JSON object with exactly these keys:
{
 "asset": "<tag> (<asset_id>, <asset_class>)",
 "condition_summary": "2-3 sentences",
 "observations": [{"text": "factual observation with numbers", "evidence_ids": ["..."]}],
 "hypotheses": [{"text": "possible cause, clearly a hypothesis", "confidence": 0.0, "evidence_ids": ["..."]}],
 "confidence": 0.0,
 "confidence_rationale": "why, incl. data-quality issues and mapping confidence",
 "next_safe_checks": [{"text": "verification step", "evidence_ids": ["..."]}],
 "escalation": "who to involve and when",
 "work_order_draft": {"title": "...", "priority": "low|medium|high", "recommended_role": "...",
                      "description": "...", "evidence_ids": ["..."]}
}
Style (a technician reads this on a tablet): condition_summary max 2 short sentences; 3-5 observations,
max 25 words each, with the actual numbers; 1-2 hypotheses; 3-4 next_safe_checks, max 20 words each,
starting with a verb; escalation 1 sentence. Plain language. Do NOT write IDs inside text fields; put them
only in evidence_ids.
Rules: cite ONLY IDs listed in allowed_ids. Facts from current_state (health, issue, urgency) are cited
with the asset_id; facts about the 3D location/mapping with scene.id (MAP-...). Never cite section names. Observations are facts from the evidence; hypotheses are
interpretations. Overall confidence must not exceed data_quality.confidence_cap. Recommend verification
and escalation, never autonomous physical action; mention that site procedures, LOTO and permits apply."""


def system_prompt(pack: dict) -> str:
    docs = {d["doc_id"]: d["content"] for d in pack["documents"]}
    return ("You are a field-service copilot for an industrial plant.\n\n"
            f"[DOC-RAG-GUIDE] {docs.get('DOC-RAG-GUIDE', '')}\n"
            f"[DOC-FIELD-SAFETY] {docs.get('DOC-FIELD-SAFETY', '')}\n\n" + OUTPUT_CONTRACT)


def user_prompt(pack: dict, question: str | None) -> str:
    q = question or "The technician has arrived at this asset. Diagnose its condition and propose next safe steps."
    return f"Question: {q}\n\nEvidence pack (JSON):\n{json.dumps(pack, default=str)}"


# ---------- LLM ----------

def call_llm(system: str, user: str, cc: Cfg) -> tuple[dict, str]:
    """One paid request through the provider-neutral layer (Featherless). No retry."""
    text, model = generate(user, cc, json_mode=True, system=system, purpose="diagnosis",
                           max_tokens=cc.get("max_tokens", 2000))
    return parse_json(text), model


def parse_json(text: str) -> dict:
    """Parse model output; tolerates ```json fences and leading/trailing prose."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return json.loads(t[t.index("{"): t.rindex("}") + 1])


# ---------- validation ----------

def _cited_lists(d: dict) -> list[list[str]]:
    lists = [o.setdefault("evidence_ids", []) for k in ("observations", "hypotheses", "next_safe_checks")
             for o in d.get(k, []) if isinstance(o, dict)]
    if isinstance(d.get("work_order_draft"), dict):
        lists.append(d["work_order_draft"].setdefault("evidence_ids", []))
    return lists


def validate_citations(d: dict, pack: dict) -> dict:
    """Drop unknown IDs from evidence lists; flag unknown IDs mentioned in free text; cap confidence."""
    allowed = set(pack["allowed_ids"])
    alias = {"current_state": pack["asset"]["asset_id"], "asset": pack["asset"]["asset_id"]}
    if pack["scene"]:
        alias["scene"] = pack["scene"]["id"]
    unknown: set[str] = set()
    cited: set[str] = set()
    for lst in _cited_lists(d):
        lst[:] = list(dict.fromkeys(alias.get(i, i) for i in lst))   # section names -> real IDs
        keep = [i for i in lst if i in allowed]
        unknown |= set(lst) - allowed
        cited |= set(keep)
        lst[:] = keep
    text_ids = set(ID_RE.findall(json.dumps(d)))
    unknown |= text_ids - allowed
    cited |= text_ids & allowed
    cap = pack["data_quality"]["confidence_cap"]
    try:
        conf = float(d.get("confidence", 0))
    except (TypeError, ValueError):
        conf = 0.0
    d["confidence"] = round(min(conf, cap), 2)
    n_obs = len(d.get("observations", []))
    grounded = sum(1 for o in d.get("observations", []) if isinstance(o, dict) and o["evidence_ids"])
    d["citation_check"] = {"cited_ids": sorted(cited), "unknown_ids_removed": sorted(unknown),
                           "observations_grounded": f"{grounded}/{n_obs}", "confidence_cap": cap,
                           "passed": not unknown and grounded == n_obs}
    return d


# ---------- deterministic fallback ----------

def template_diagnosis(pack: dict) -> dict:
    a, s = pack["asset"], pack["current_state"]
    moved = sorted((t for t in pack["telemetry"] if t["change_pct"] and abs(t["change_pct"]) >= 5),
                   key=lambda t: -abs(t["change_pct"]))
    obs = [{"text": f"{t['metric']} {t['first_day_mean']} -> {t['last_day_mean']} ({t['change_pct']:+}%) "
                    f"over 3 days; healthy peers {t['healthy_peer_mean']}.", "evidence_ids": [t["id"]]}
           for t in moved]
    ev = pack["events"]
    if ev["issue_code_events"]:
        obs.append({"text": f"{ev['counts_by_code'].get(s['issue_code'], 0)} {s['issue_code']} events recorded.",
                    "evidence_ids": [e["event_id"] for e in ev["issue_code_events"][:3]]})
    case = pack["cases"][0] if pack["cases"] else None
    doc = s["knowledge_doc_id"]
    return {"asset": f"{a['tag']} ({a['asset_id']}, {a['asset_class']})",
            "condition_summary": f"{a['tag']} is {s['condition_state']} (health {s['health_score']}) "
                                 f"with {s['issue_code']}: {s['issue_summary']}.",
            "observations": obs,
            "hypotheses": [{"text": case["historical_interpretation"] if case else s["issue_summary"],
                            "confidence": 0.6, "evidence_ids": [case["case_id"]] if case else []}],
            "confidence": s["model_confidence"],
            "confidence_rationale": "Template fallback (LLM unavailable); based on Current_State model confidence.",
            "next_safe_checks": [{"text": s["recommended_next_check"], "evidence_ids": [doc]},
                                 {"text": "Follow site procedures, LOTO and permits before any physical work.",
                                  "evidence_ids": ["DOC-FIELD-SAFETY"]}],
            "escalation": "Escalate to the responsible maintenance engineer if verification confirms the trend.",
            "work_order_draft": {"title": f"Verify {s['issue_code']} on {a['tag']}", "priority": s["urgency"],
                                 "recommended_role": "field_technician", "description": s["recommended_next_check"],
                                 "evidence_ids": [o["evidence_ids"][0] for o in obs[:3]]}}


# ---------- entry points ----------

def cache_path(cfg: Cfg, tag: str, question: str | None, pack: dict | None = None):
    """Keyed by question + model + evidence pack, so a cached answer is reused only for the same evidence."""
    key = (question or "").strip().lower() + "|" + cfg["copilot"]["model"]
    if pack is not None:
        key += "|" + json.dumps(pack, sort_keys=True, default=str)
    qh = hashlib.sha1(key.encode()).hexdigest()[:10]
    d = out_path(cfg, "diagnosis_cache")
    d.mkdir(exist_ok=True)
    return d / f"{tag}__{qh}.json"


def diagnose(key: str, question: str | None = None, cfg: Cfg | None = None, allow_paid: bool = True) -> dict:
    """Cached answer for the same evidence -> paid LLM call (if allowed and within budget) -> template.
    allow_paid=False is used for automatic viewer calls so they never spend credits."""
    cfg = cfg or load_config()
    pack = build_evidence(key, cfg)
    save_evidence(pack, cfg)
    tag = pack["asset"]["tag"]
    cp = cache_path(cfg, tag, question, pack)
    t0 = time.time()
    try:
        if cp.exists():
            c = json.loads(cp.read_text(encoding="utf-8"))
            raw, model = c["raw"], f"{c['model']} (cached answer from {c['at']}, same evidence)"
            d = validate_citations(json.loads(json.dumps(raw)), pack)
            d["meta"] = {"source": model, "latency_s": round(time.time() - t0, 1), "question": question,
                         "scene": pack["scene"], "status": pack["current_state"],
                         "sources": describe_ids(pack, d["citation_check"]["cited_ids"])}
            return d
        if not allow_paid:
            raise LLMUnavailable("automatic call: paid requests disabled (press 'Ask copilot')")
        raw, model = call_llm(system_prompt(pack), user_prompt(pack, question), cfg["copilot"])
        cp.write_text(json.dumps({"raw": raw, "model": model, "at": time.strftime("%Y-%m-%d %H:%M")}),
                      encoding="utf-8")
        d, source = raw, model
    except Exception as e:  # noqa: BLE001 - demo must not break; reason is recorded
        if cp.exists():
            c = json.loads(cp.read_text(encoding="utf-8"))
            d, source = c["raw"], f"{c['model']} (cached answer from {c['at']}; live call failed)"
        else:
            d = template_diagnosis(pack)
            source = f"template_fallback ({type(e).__name__}: {str(e)[:120]})"
    d = validate_citations(json.loads(json.dumps(d)), pack)
    d["meta"] = {"source": source, "latency_s": round(time.time() - t0, 1), "question": question,
                 "scene": pack["scene"], "status": pack["current_state"],
                 "sources": describe_ids(pack, d["citation_check"]["cited_ids"])}
    with open(out_path(cfg, f"diagnosis_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1, default=str)
    return d


def describe_ids(pack: dict, ids: list[str]) -> dict[str, str]:
    """Short human description per cited ID, for the Sources list."""
    desc: dict[str, str] = {}
    for t in pack["telemetry"]:
        desc[t["id"]] = (f"{t['metric'].replace('_', ' ')}: {t['first_day_mean']} → {t['last_day_mean']} "
                         f"(healthy peers {t['healthy_peer_mean']})")
    for k in ("issue_code_events", "other_important_events"):
        for e in pack["events"][k]:
            desc[e["event_id"]] = f"{e['event_code']} {e['event_type'].lower()} · {e['severity']} · {e['timestamp'][:16]}"
    for w in pack["work_orders"]:
        desc[w["work_order_id"]] = f"work order: {w['action']} → {w['result']} ({str(w['opened_at'])[:10]})"
    for c in pack["cases"]:
        desc[c["case_id"]] = f"past case: {c['historical_interpretation']} → {c['recommended_next_check']}"
    for d in pack["documents"]:
        desc[d["doc_id"]] = d["title"]
    for q in pack["data_quality"]["issues"]:
        desc[q["issue_id"]] = f"data quality: {q['issue_type']} ({q['severity']})"
    desc[pack["asset"]["asset_id"]] = "asset master + current state"
    desc[pack["asset"]["tag"]] = "asset master + current state"
    if pack["scene"]:
        sc = pack["scene"]
        desc[sc["id"]] = f"3D mapping: {sc['object_id']} via {sc['mapping_method']} (confidence {sc['mapping_confidence']})"
    return {i: desc.get(i, "") for i in ids}


STATUS_DOT = {"attention_required": "🔴", "degraded": "🟠", "watch": "🟡", "healthy": "🟢"}


def to_markdown(d: dict) -> str:
    """Readable panel text: short sections, numbered source refs instead of raw IDs inline."""
    refs: dict[str, int] = {}

    def cite(ids: list[str]) -> str:
        nums = [refs.setdefault(i, len(refs) + 1) for i in ids]
        return f" _[{', '.join(map(str, nums))}]_" if nums else ""

    meta = d.get("meta", {})
    st = meta.get("status") or {}
    conf = float(d.get("confidence") or 0)
    bar = "▰" * round(conf * 10) + "▱" * (10 - round(conf * 10))
    md = [f"### {STATUS_DOT.get(st.get('condition_state'), '⚪')} {d.get('asset', '')}"]
    if meta.get("question"):
        md += [f"> **Q:** {meta['question']}", ""]
    md += [d.get("condition_summary", ""), "",
           f"**Confidence** {bar} **{conf:.2f}**  ", f"_{d.get('confidence_rationale', '')}_", "",
           "#### What the data shows"]
    md += [f"- {o['text']}{cite(o['evidence_ids'])}" for o in d.get("observations", [])]
    md += ["", "#### Likely cause"]
    md += [f"- {h['text']} (confidence {h.get('confidence')}){cite(h.get('evidence_ids', []))}"
           for h in d.get("hypotheses", [])]
    md += ["", "#### Next safe steps"]
    md += [f"{i + 1}. {c['text']}{cite(c['evidence_ids'])}" for i, c in enumerate(d.get("next_safe_checks", []))]
    md += ["", "#### Escalation", d.get("escalation", "")]
    wo = d.get("work_order_draft") or {}
    if wo:
        md += ["", "#### Draft work order", "| Title | Priority | Role |", "|---|---|---|",
               f"| {wo.get('title')} | {wo.get('priority')} | {wo.get('recommended_role')} |",
               "", f"{wo.get('description', '')}{cite(wo.get('evidence_ids', []))}"]
    src = meta.get("sources", {})
    md += ["", "#### Sources"]
    md += [f"{n}. `{i}` {src.get(i, '')}" for i, n in sorted(refs.items(), key=lambda kv: kv[1])]
    cc = d["citation_check"]
    ok = "✅ all citations verified" if cc["passed"] else f"⚠️ {len(cc['unknown_ids_removed'])} unknown IDs removed"
    md += ["", f"_{ok} · {cc['observations_grounded']} observations grounded · "
               f"{meta.get('source', '')} · {meta.get('latency_s')} s_"]
    return "\n".join(md)


if __name__ == "__main__":
    cfg = load_config()
    tag = sys.argv[1] if len(sys.argv) > 1 else cfg["mapping"]["target_tag"]
    q = sys.argv[2] if len(sys.argv) > 2 else None
    diag = diagnose(tag, q, cfg)
    print(to_markdown(diag))
    print(f"\n[check] {json.dumps(diag['citation_check'])}")
