"""Training quiz: nothing shown before the reveal gives the answer away; options and grading."""
import json
import random
import re

import pytest

from copilot.card import _words
from copilot.evidence import build_evidence, load_pack
from copilot.training import (HEALTHY, Session, answer_vocab, build_scenario, grade, issue_label, redact,
                              scenario_pool)
from pipeline.common import load_config, out_path, repo_path

CFG = load_config()
need = pytest.mark.skipif(not (repo_path(CFG["paths"]["xlsx"]).exists() and out_path(CFG, "scene_graph.json").exists()),
                          reason="Excel pack or out/scene_graph.json missing")


def all_tags():
    return list(load_pack(str(repo_path(CFG["paths"]["xlsx"])))["Current_State"]["tag"])


def has(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word.lower())}\b", text.lower()) is not None


def leaks(text: str, issue: str, state: str) -> list[str]:
    bad = [w for w in (issue, _words(issue)) if has(text, w)]
    if state != "healthy":                       # "healthy" is a column name (healthy peers)
        bad += [w for w in (state, state.replace("_", " ")) if has(text, w)]
    return bad


@need
@pytest.mark.parametrize("tag", all_tags() if repo_path(CFG["paths"]["xlsx"]).exists() else [])
def test_pre_reveal_evidence_does_not_leak(tag):
    sc = build_scenario(tag, CFG, random.Random(0))
    s = sc["pack"]["current_state"]
    shown = json.dumps(sc["evidence"]) + "\n" + sc["evidence_md"]
    assert not leaks(shown, s["issue_code"], s["condition_state"])
    assert not has(shown, s["recommended_next_check"])
    for key in ("current_state", "health_score", "condition_state", "issue_code", "issue_summary", "cases"):
        assert f'"{key}"' not in shown
    assert '"healthy_peers"' not in shown


@need
def test_redact_masks_every_answer_code():
    pack = build_evidence("P-002", CFG)
    vocab = answer_vocab(load_pack(str(repo_path(CFG["paths"]["xlsx"]))))
    red = redact(pack, vocab)
    assert "current_state" not in red and "cases" not in red and "documents" not in red
    codes = {e["event_code"] for e in red["events"]["recent"]} | set(red["events"]["counts_by_code"])
    assert not any(v in c.upper() for c in codes for v in vocab)
    assert sum(red["events"]["counts_by_code"].values()) == pack["events"]["total"]
    assert pack["current_state"]["issue_code"] == "CAVITATION_RISK"          # original untouched


@need
def test_questions_and_grading():
    sc = build_scenario("P-005", CFG, random.Random(1))
    q1, q2 = sc["q1"], sc["q2"]
    assert len(q1["options"]) == 4 == len(set(q1["options"]))
    assert q1["answer"] == "Fouling" and q1["answer"] in q1["options"] and issue_label(HEALTHY) in q1["options"]
    assert q2["answer"] in q2["options"] and len(set(q2["options"])) == len(q2["options"]) == 4
    assert q2["unsafe"] in q2["options"]
    r = grade(sc, q1["answer"], q2["unsafe"])
    assert r["q1_ok"] and not r["q2_ok"] and r["unsafe"]


@need
def test_healthy_scenario_answer_is_no_issue():
    healthy = [t for t in scenario_pool(CFG)
               if load_pack(str(repo_path(CFG["paths"]["xlsx"])))["Current_State"].set_index("tag").loc[t, "issue_code"] == HEALTHY]
    assert 1 <= len(healthy) <= CFG["training"]["n_healthy"]
    sc = build_scenario(healthy[0], CFG, random.Random(2))
    assert sc["q1"]["answer"] == "No issue / healthy" and sc["q2"]["answer"] == "Monitor"


@need
def test_session_end_to_end_and_summary():
    cfg = {**CFG, "training": {**CFG["training"], "order": "fixed", "session_size": None, "seed": 3}}
    s = Session(cfg)
    assert s.tags == scenario_pool(cfg)
    while not s.done:
        sc = s.current()
        wrong = next(o for o in sc["q1"]["options"] if o != sc["q1"]["answer"])
        _, md = s.answer(sc["q1"]["answer"] if s.i % 2 else wrong, sc["q2"]["answer"])
        assert "Answer:" in md and "DOC-FIELD-SAFETY" in md
        s.next()
    n = len(s.tags)
    assert s.score().startswith(f"{n // 2}/{n} fully correct")
    assert "Missed:" in s.summary()


class _W:
    """Stand-in for a Viser widget/handle: plain attributes, on_click stores the callback."""
    def __init__(self, value=None, options=None, **kw):
        self.value, self.options, self.visible, self.label, self.content, self.cb = value, options, True, "", "", None
        self.__dict__.update(kw)

    def on_click(self, fn):
        self.cb = fn
        return fn

    def click(self):
        self.cb(None)


class _G:
    def add_markdown(self, content="", **kw):
        return _W(content=content)

    def add_button(self, label, **kw):
        return _W(label=label, visible=kw.get("visible", True))

    def add_dropdown(self, label, options, **kw):
        return _W(value=options[0], options=list(options), visible=kw.get("visible", True))


@need
def test_train_tab_runs_all_scenarios_and_restores_scene():
    from types import SimpleNamespace
    from viewer.train_tab import TrainTab
    walked, hidden_on_walk = [], []
    sim = SimpleNamespace(cfg={**CFG, "training": {**CFG["training"], "session_size": None}},
                          dd_labels=_W(value="Needs attention"), cb_health=_W(value=True), visit_active=None,
                          quiz_hidden=False, h_target_label=_W(), dd_walk=_W(), dd_equip=_W())
    sim.find_node = lambda tag: {"tag": tag}
    sim.go_to = lambda n: (walked.append(n["tag"]), hidden_on_walk.append(sim.quiz_hidden))
    tab = TrainTab(sim, _G())
    tab.b_start.click()
    assert sim.dd_labels.value == "Off" and sim.cb_health.value is False and not sim.h_target_label.visible
    n = len(tab.session.tags)
    for _ in range(n):
        sc = tab.session.current()
        assert sim.quiz_hidden and sc["q1"]["answer"] not in tab.md_evidence.content
        assert not sim.dd_walk.visible and not sim.dd_equip.visible
        tab.dd_q1.value, tab.dd_q2.value = sc["q1"]["answer"], sc["q2"]["answer"]
        tab.b_submit.click()
        assert not sim.quiz_hidden and "Answer:" in tab.md_reveal.content
        assert sim.dd_walk.visible and sim.dd_equip.visible
        tab.b_next.click()
    assert walked == tab.session.tags and all(hidden_on_walk)
    assert tab.md_head.content == "### Session complete" and f"{n}/{n} fully correct" in tab.md_score.content
    assert sim.dd_labels.value == "Needs attention" and sim.cb_health.value is True and sim.h_target_label.visible


@need
def test_answer_counts_once_and_rejects_finished_session():
    s = Session({**CFG, "training": {**CFG["training"], "session_size": 1}})
    sc = s.current()
    first, _ = s.answer(sc["q1"]["answer"], sc["q2"]["answer"])
    again, _ = s.answer("wrong", "wrong")                  # double submit
    assert again is first and len(s.results) == 1
    s.next()
    with pytest.raises(ValueError):
        s.answer("x", "y")
