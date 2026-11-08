"""Training mode: scenario quiz over real assets (UI-free; the viewer's Train tab is a thin layer on top).

The trainee sees redacted evidence (telemetry, masked alarms, maintenance, data quality), picks what is
wrong (Q1) and the next safe step (Q2), then gets the answer with the free cached/template diagnosis,
the most similar past case and the safety note. No LLM calls, no files written; scores stay in memory.
"""
from __future__ import annotations

import copy
import json
import random
from typing import Any

from copilot.card import _reading_rows, _stamp, _day, _words
from copilot.evidence import build_evidence, load_pack
from copilot.offline import offline_diagnosis
from pipeline.common import load_config, out_path, repo_path

Cfg = dict[str, Any]
HEALTHY = "NORMAL"
HIDDEN = "hidden"
URGENCY = {"attention_required": 0, "degraded": 1, "watch": 2, "healthy": 3}


def issue_label(code: str) -> str:
    return "No issue / healthy" if code == HEALTHY else _words(code)


def answer_vocab(xl: dict) -> list[str]:
    """Every issue code a Q1 option can be (Current_State), incl. NORMAL."""
    return sorted(set(xl["Current_State"]["issue_code"].astype(str)) | {HEALTHY})


def _excel(cfg: Cfg) -> dict:
    return load_pack(str(repo_path(cfg["paths"]["xlsx"])))


# ---------- scenario pool ----------

def scenario_pool(cfg: Cfg) -> list[str]:
    """Mapped non-healthy assets (most urgent first) plus training.n_healthy healthy ones."""
    tc = cfg.get("training", {})
    with open(out_path(cfg, "scene_graph.json"), encoding="utf-8") as f:
        assets = [n["asset"] for n in json.load(f)["objects"] if n.get("asset")]
    sick = sorted((a for a in assets if a["condition_state"] != "healthy"),
                  key=lambda a: (URGENCY.get(a["condition_state"], 9), a["health_score"], a["tag"]))
    healthy = sorted(a["tag"] for a in assets if a["condition_state"] == "healthy")
    rng = random.Random(tc.get("seed") if tc.get("seed") is not None else 0)
    pick = rng.sample(healthy, min(int(tc.get("n_healthy", 2)), len(healthy)))
    return [a["tag"] for a in sick] + pick


# ---------- redaction ----------

def redact(pack: dict, vocab: list[str]) -> dict:
    """Evidence without the answer: no current_state (issue, condition, health, next check), no similar
    cases or docs (they are matched on the issue), no healthy-peer tag lists, and every event code containing an answer-vocabulary
    code masked. Masking is the same for every asset, so a hidden code carries no signal by itself."""
    def masked(code: str) -> str:
        return HIDDEN if any(v in str(code).upper() for v in vocab) else code

    out = {k: copy.deepcopy(pack[k]) for k in ("asset", "telemetry", "work_orders", "scene")}
    for t in out["telemetry"]:
        t.pop("healthy_peers", None)            # a healthy asset is listed among its own peers
    ev = pack["events"]
    events = [{**e, "event_code": masked(e["event_code"])}
              for e in ev["issue_code_events"] + ev["other_important_events"]]
    counts: dict[str, int] = {}
    for code, n in ev["counts_by_code"].items():
        counts[masked(code)] = counts.get(masked(code), 0) + n
    out["events"] = {"total": ev["total"], "counts_by_code": counts,
                     "recent": sorted(events, key=lambda e: str(e["timestamp"]), reverse=True)}
    out["data_quality"] = {"issues": copy.deepcopy(pack["data_quality"]["issues"])}
    return out


def evidence_markdown(red: dict) -> str:
    """What the trainee sees before answering (redacted evidence only)."""
    a, ev = red["asset"], red["events"]
    md = [f"**{a['tag']}** · {_words(a['asset_class'])} · {a.get('area_name', '')} · criticality "
          f"{a.get('criticality', '—')}"]
    if red["telemetry"]:
        md += ["", "#### Readings · last 3 days", "| Reading | Then → now | Trend | Healthy peers |",
               "|---|---|---|---|", *_reading_rows(red["telemetry"]),
               "", "_⚠️ = now more than 10% away from healthy equipment of the same type._"]
    counts = sorted(ev["counts_by_code"].items(), key=lambda kv: -kv[1])
    md += ["", f"#### Alarms & events · {ev['total']} in the log",
           " · ".join(f"**{n}** {'code hidden' if c == HIDDEN else _words(c).lower()}" for c, n in counts)]
    if ev["recent"]:
        md += ["", "| When | Type | Severity | Code | Status | ID |", "|---|---|---|---|---|---|"]
        md += [f"| {_stamp(e['timestamp'])} | {_words(e['event_type'])} | {_words(e['severity'])} | "
               f"{'_hidden_' if e['event_code'] == HIDDEN else _words(e['event_code'])} | {_words(e['status'])} | "
               f"`{e['event_id']}` |" for e in ev["recent"][:8]]
    if red["work_orders"]:
        md += ["", "#### Maintenance history", "| Opened | Work | Result | ID |", "|---|---|---|---|"]
        md += [f"| {_day(w['opened_at'])} | {_words(w['action'])} ({w['maintenance_type']}) | {_words(w['result'])} | "
               f"`{w['work_order_id']}` |" for w in red["work_orders"]]
    dq = red["data_quality"]["issues"]
    if dq:
        md += ["", "#### Data quality", " · ".join(f"{_words(q['issue_type']).lower()} ({q['severity']}, "
                                                   f"`{q['issue_id']}`)" for q in dq)]
    return "\n".join(md)


# ---------- questions ----------

def issue_options(tag: str, xl: dict, rng: random.Random) -> list[str]:
    """Q1: the right issue code + 3 distractors (same class first, NORMAL always present), shuffled."""
    cs = xl["Current_State"]
    me = cs[cs["tag"] == tag].iloc[0]
    answer, cls = str(me["issue_code"]), me["asset_class"]
    same = sorted(set(cs[cs["asset_class"] == cls]["issue_code"].astype(str)) - {answer, HEALTHY})
    other = sorted(set(cs["issue_code"].astype(str)) - set(same) - {answer, HEALTHY})
    rng.shuffle(same)
    rng.shuffle(other)
    picks = ([] if answer == HEALTHY else [HEALTHY]) + same + other
    opts = [answer] + picks[:3]
    rng.shuffle(opts)
    return opts


def step_options(tag: str, xl: dict, rng: random.Random, tc: Cfg) -> tuple[list[str], str | None]:
    """Q2: this asset's recommended check + 2 other assets' checks (+ one unsafe step). Returns
    (option labels, the unsafe label or None)."""
    cs = xl["Current_State"]
    answer = str(cs[cs["tag"] == tag].iloc[0]["recommended_next_check"])
    others = sorted(set(cs["recommended_next_check"].astype(str)) - {answer})
    rng.shuffle(others)
    opts = [_words(answer)] + [_words(o) for o in others[:2]]
    unsafe = None
    if tc.get("unsafe_distractor", True) and tc.get("unsafe_steps"):
        unsafe = rng.choice(list(tc["unsafe_steps"]))
        opts.append(unsafe)
    rng.shuffle(opts)
    return opts, unsafe


def build_scenario(tag: str, cfg: Cfg, rng: random.Random) -> dict:
    xl = _excel(cfg)
    pack = build_evidence(tag, cfg)
    red = redact(pack, answer_vocab(xl))
    s = pack["current_state"]
    q1 = issue_options(tag, xl, rng)
    q2, unsafe = step_options(tag, xl, rng, cfg.get("training", {}))
    return {"tag": tag, "object_id": (pack["scene"] or {}).get("object_id"),
            "asset_class": pack["asset"]["asset_class"], "evidence": red, "evidence_md": evidence_markdown(red),
            "q1": {"options": [issue_label(c) for c in q1], "answer": issue_label(s["issue_code"])},
            "q2": {"options": q2, "answer": _words(s["recommended_next_check"]), "unsafe": unsafe},
            "pack": pack}


def grade(sc: dict, q1: str | None, q2: str | None) -> dict:
    return {"tag": sc["tag"], "q1": q1, "q2": q2, "q1_ok": q1 == sc["q1"]["answer"],
            "q2_ok": q2 == sc["q2"]["answer"], "unsafe": q2 is not None and q2 == sc["q2"]["unsafe"],
            "answer": sc["q1"]["answer"]}


def reveal_markdown(sc: dict, res: dict, cfg: Cfg) -> str:
    """Answer + why: offline diagnosis (cited IDs, confidence), most similar past case, safety note."""
    from copilot.agent import to_markdown
    pack, s = sc["pack"], sc["pack"]["current_state"]
    mark = lambda ok: "✅ Correct" if ok else "❌ Not quite"  # noqa: E731
    md = [f"#### {mark(res['q1_ok'])}: what's wrong", f"Answer: **{sc['q1']['answer']}**"
          + ("" if res["q1_ok"] else f" (you chose {res['q1'] or 'nothing'})"),
          f"Excel status: {_words(s['condition_state'])} · health {s['health_score']}/100 · urgency "
          f"{_words(s['urgency'])} · `{pack['asset']['asset_id']}`",
          "", f"#### {mark(res['q2_ok'])}: next safe step", f"Answer: **{sc['q2']['answer']}** "
          f"(recommended check, `{s['knowledge_doc_id']}`)" + ("" if res["q2_ok"] else f" (you chose {res['q2'] or 'nothing'})")]
    if res["unsafe"]:
        md += ["", "🛑 **Unsafe choice.** Never skip lockout-tagout or permits, and never open or touch equipment "
                   "before it is isolated and verified (`DOC-FIELD-SAFETY`). Verify, then escalate."]
    d = offline_diagnosis(pack, cfg)
    kind = "Cached AI answer" if d["meta"]["kind"] == "cached_llm" else "Template diagnosis"
    md += ["", f"#### Why ({kind})", to_markdown(d).split("\n", 1)[1]]
    case = next((c for c in pack["cases"] if c["match"] == "class+issue"), None)
    md += ["", "#### Most similar past case"]
    md += ([f"`{case['case_id']}` · {case['historical_interpretation']} → {_words(case['recommended_next_check'])} "
            f"({case['severity']} severity)"] if case else ["_No past case with this issue for this equipment type._"])
    safety = next((x["content"] for x in pack["documents"] if x["doc_id"] == "DOC-FIELD-SAFETY"), "")
    md += ["", f"🦺 **Safety** (`DOC-FIELD-SAFETY`): {safety}"]
    return "\n".join(md)


# ---------- session ----------

class Session:
    """One quiz run in memory: scenarios are built when reached; answer() grades the current one."""

    def __init__(self, cfg: Cfg | None = None, tags: list[str] | None = None) -> None:
        self.cfg = cfg or load_config()
        tc = self.cfg.get("training", {})
        self.rng = random.Random(tc.get("seed"))
        tags = list(tags) if tags is not None else scenario_pool(self.cfg)
        if tc.get("order", "random") == "random":
            self.rng.shuffle(tags)
        size = tc.get("session_size")
        self.tags = tags[:size] if size else tags
        self.i = 0
        self.results: list[dict] = []
        self._sc: dict | None = None

    @property
    def done(self) -> bool:
        return self.i >= len(self.tags)

    def current(self) -> dict | None:
        if self.done:
            return None
        if self._sc is None or self._sc["tag"] != self.tags[self.i]:
            self._sc = build_scenario(self.tags[self.i], self.cfg, self.rng)
        return self._sc

    def answer(self, q1: str | None, q2: str | None) -> tuple[dict, str]:
        """Grade the current scenario once; answering it again (e.g. a double click) returns the first result."""
        sc = self.current()
        if sc is None:
            raise ValueError("the session is finished")
        if len(self.results) > self.i:
            return self.results[self.i], self._reveal
        res = grade(sc, q1, q2)
        self.results.append(res)
        self._reveal = reveal_markdown(sc, res, self.cfg)
        return res, self._reveal

    def next(self) -> dict | None:
        self.i += 1
        return self.current()

    def score(self) -> str:
        n = len(self.results)
        both = sum(r["q1_ok"] and r["q2_ok"] for r in self.results)
        return (f"{both}/{n} fully correct · diagnosis {sum(r['q1_ok'] for r in self.results)}/{n} · "
                f"safe step {sum(r['q2_ok'] for r in self.results)}/{n}")

    def summary(self) -> str:
        missed = [f"{r['answer'].lower()} on {r['tag']}" for r in self.results if not r["q1_ok"]]
        steps = [r["tag"] for r in self.results if r["q1_ok"] and not r["q2_ok"]]
        unsafe = [r["tag"] for r in self.results if r["unsafe"]]
        md = [f"**{self.score()}**"]
        if missed:
            md.append("Missed: " + ", ".join(missed))
        if steps:
            md.append("Right diagnosis, wrong next step: " + ", ".join(steps))
        if unsafe:
            md.append("🛑 Unsafe step chosen on: " + ", ".join(unsafe) + ". LOTO and permits always apply.")
        return "  \n".join(md)
