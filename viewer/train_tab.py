"""Train tab (scenario quiz) and the Info tab's pre-inspection brief button. Logic lives in
copilot/training.py and copilot/brief.py; this only wires Viser widgets to it. Both are optional extras:
viewer.app builds them inside try/except so a failure here never affects the rest of the viewer."""
from __future__ import annotations

import threading
from typing import Any

import numpy as np

CHOOSE = "— choose —"


class TrainTab:
    def __init__(self, sim: Any, g: Any) -> None:
        from copilot.training import Session, scenario_pool
        self.sim, self.Session = sim, Session
        self.session: Any = None
        self._saved: tuple[str, bool] | None = None
        tc = sim.cfg.get("training", {})
        pool = scenario_pool(sim.cfg)
        g.add_markdown(f"**Training mode.** The worker walks to a real asset; read its evidence and answer two "
                       f"questions. Pool: {len(pool)} assets ({tc.get('n_healthy', 2)} healthy), "
                       f"{tc.get('session_size') or len(pool)} per session, {tc.get('order', 'random')} order. "
                       f"No AI requests; the score stays in this session.")
        self.b_start = g.add_button("Start training session", icon="school")
        self.md_head = g.add_markdown("")
        self.md_evidence = g.add_markdown("")
        self.dd_q1 = g.add_dropdown("1. What's wrong?", [CHOOSE], visible=False)
        self.dd_q2 = g.add_dropdown("2. Next safe step?", [CHOOSE], visible=False)
        self.b_submit = g.add_button("Submit answer", icon="check", visible=False)
        self.md_reveal = g.add_markdown("")
        self.b_next = g.add_button("Next scenario", icon="arrow-right", visible=False)
        self.md_score = g.add_markdown("")
        self.b_stop = g.add_button("End session", color="gray", icon="x", visible=False)

        self.b_start.on_click(lambda _e: self.guard(self.start))
        self.b_submit.on_click(lambda _e: self.guard(self.submit))
        self.b_next.on_click(lambda _e: self.guard(self.next))
        self.b_stop.on_click(lambda _e: self.guard(self.end))

    def guard(self, fn: Any) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - the quiz must never break the viewer
            self.md_reveal.content = f"⚠️ _Training error: {type(e).__name__}: {e}_"
            try:                                                # give the rest of the viewer its normal state back
                self.leave()
            except Exception:  # noqa: BLE001
                self.sim.quiz_hidden = False

    # --- 3D: hide every status cue while a question is open ---
    def set_hidden(self, on: bool) -> None:
        sim = self.sim
        sim.quiz_hidden = on
        h = getattr(sim, "h_target_label", None)
        if h is not None:
            h.visible = not on
        if sim.visit_active is not None:                       # redraw the stop outline / tint / label
            sim.highlight_stop(sim.visit_active)

    def enter(self) -> None:
        if self._saved is None:
            self._saved = (self.sim.dd_labels.value, bool(self.sim.cb_health.value))
        self.sim.dd_labels.value = "Off"
        self.sim.cb_health.value = False

    def leave(self) -> None:
        self.set_hidden(False)
        if self._saved is not None:
            self.sim.dd_labels.value, self.sim.cb_health.value = self._saved
            self._saved = None

    # --- flow ---
    def start(self) -> None:
        self.session = self.Session(self.sim.cfg)
        self.enter()
        self.md_score.content = ""
        self.b_stop.visible = True
        self.b_start.label = "Restart session"
        self.show()

    def show(self) -> None:
        sc = self.session.current()
        if sc is None:
            return self.finish()
        self.enter()
        self.set_hidden(True)
        n = self.session.i + 1
        self.md_head.content = f"### Scenario {n}/{len(self.session.tags)}: {sc['tag']}"
        self.md_evidence.content = sc["evidence_md"]
        self.dd_q1.options = [CHOOSE, *sc["q1"]["options"]]
        self.dd_q2.options = [CHOOSE, *sc["q2"]["options"]]
        self.dd_q1.value = self.dd_q2.value = CHOOSE
        self.dd_q1.visible = self.dd_q2.visible = self.b_submit.visible = True
        self.b_next.visible = False
        self.md_reveal.content = ""
        node = self.sim.find_node(sc["tag"])
        if node is not None:
            self.sim.go_to(node)

    def submit(self) -> None:
        if self.session is None or self.session.done:
            return
        q1 = None if self.dd_q1.value == CHOOSE else self.dd_q1.value
        q2 = None if self.dd_q2.value == CHOOSE else self.dd_q2.value
        if q1 is None or q2 is None:
            self.md_reveal.content = "_Pick an answer for both questions first._"
            return
        _, md = self.session.answer(q1, q2)
        self.set_hidden(False)
        self.md_reveal.content = md
        self.md_score.content = f"Score so far: {self.session.score()}"
        self.dd_q1.visible = self.dd_q2.visible = self.b_submit.visible = False
        self.b_next.visible = True
        self.b_next.label = "Next scenario" if self.session.i + 1 < len(self.session.tags) else "See summary"

    def next(self) -> None:
        self.session.next()
        self.show()

    def finish(self) -> None:
        self.md_head.content = "### Session complete"
        self.md_evidence.content = ""
        self.md_reveal.content = ""
        self.md_score.content = self.session.summary()
        self.dd_q1.visible = self.dd_q2.visible = self.b_submit.visible = self.b_next.visible = False
        self.b_stop.visible = False
        self.leave()

    def end(self) -> None:
        if self.session is not None and self.session.results:
            self.md_score.content = self.session.summary()
        self.md_head.content = self.md_evidence.content = self.md_reveal.content = ""
        self.dd_q1.visible = self.dd_q2.visible = self.b_submit.visible = self.b_next.visible = False
        self.b_stop.visible = False
        self.session = None
        self.leave()


def add_brief_button(sim: Any, g: Any) -> None:
    """Info tab: write the selected equipment's pre-inspection brief and show it below the button."""
    b = g.add_button("Pre-inspection brief", color="gray", icon="clipboard-check",
                     hint="Printable brief for a technician before going on site: route, nearby issues, diagnosis, "
                          "what to verify, safety, checklist, draft work order. No AI request.")
    with g.add_folder("Pre-inspection brief", expand_by_default=True, visible=False) as fo:
        md_paths = g.add_markdown("")
        img = g.add_image(np.full((8, 24, 3), 235, np.uint8))
        md = g.add_markdown("")

    @b.on_click
    def _(_e: Any) -> None:
        n = getattr(sim, "info_node", None)
        if not n or not n.get("asset"):
            md_paths.content, fo.visible = "_Pick an equipment with an Excel record first._", True
            return
        tag = n["asset"]["tag"]
        md_paths.content, md.content, fo.visible = f"_Writing the brief for {tag}…_", "", True

        def work() -> None:
            try:
                from copilot.brief import write_brief
                import matplotlib.image as mpimg
                r = write_brief(tag, sim.cfg)
                img.image = (mpimg.imread(r["png"])[..., :3] * 255).astype(np.uint8)
                md_paths.content = f"Saved: `{r['html']}` (printable) and `{r['md']}`"
                md.content = "\n".join(ln for ln in r["markdown"].split("\n")[2:] if not ln.startswith("!["))
            except Exception as e:  # noqa: BLE001 - optional feature
                md_paths.content = f"⚠️ _Brief failed: {type(e).__name__}: {e}_"
        threading.Thread(target=work, daemon=True).start()
