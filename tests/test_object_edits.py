"""Object movement uses the same undo and navigation hooks as dragging."""
from types import SimpleNamespace as NS

import numpy as np
import pytest
from viser.transforms import SO3

from viewer import scene_edits as se
from viewer.app import Sim
from viewer.nudge import add_nudge_gui, floor_axes, up_vector


@pytest.mark.parametrize("up", ["-y", "+z"])
def test_floor_axes_follow_camera(up):
    vertical = up_vector(up)
    forward = np.array([0., 0., 1.]) if up == "-y" else np.array([0., 1., 0.])
    client = NS(camera=NS(position=np.zeros(3), look_at=forward + vertical))
    f, r = floor_axes(client, vertical)
    assert np.allclose(f, forward) and np.allclose(r, [1, 0, 0])
    client.camera.look_at = vertical
    f, r = floor_axes(client, vertical)
    assert np.isclose(np.linalg.norm(f), 1) and np.isclose(np.linalg.norm(r), 1)
    assert np.isclose(f @ vertical, 0) and np.isclose(r @ vertical, 0)


def test_nudge_records_each_step_and_invalidates_navigation():
    s = Sim.__new__(Sim)
    s.cfg = {"scene": {"up_direction": "-y"}}
    s.sel_inst = "OBJ-0001"
    s.inst = {s.sel_inst: {"h": NS(position=(0, 0, 0), wxyz=(1, 0, 0, 0)), "removed": False}}
    s.edits = se.new_state({}, {s.sel_inst: [0, 0, 0]})
    s._gizmo_dirty = True  # a preceding drag must not swallow the first button's undo step
    s.gizmo = NS(position=(0, 0, 0), wxyz=(1, 0, 0, 0))
    s.h_selected = NS(points=None)
    s.visit_active = {"object_id": s.sel_inst}
    changed, highlighted = [], []
    s.inst_box = lambda name: np.array(s.inst[name]["h"].position)
    s.lookup = lambda name: name
    s._draw_stop_box = highlighted.append
    s.layout_changed = lambda: changed.append(True)

    def save():
        h = s.inst[s.sel_inst]["h"]
        s.edits["instances"][s.sel_inst] = se.instance(s.sel_inst, [0, 0, 0], position=h.position, wxyz=h.wxyz)
    s.save_edits = save
    s.nudge(np.array([0.2, 0, 0]), 0)
    s.nudge(np.zeros(3), np.pi / 2)
    assert len(changed) == len(highlighted) == len(s.edits["history"]) == 2
    assert np.allclose(s.gizmo.position, [0.2, 0, 0])
    assert np.allclose(SO3(np.array(s.gizmo.wxyz)).as_matrix() @ [0, 0, 1], [-1, 0, 0])
    se.undo(s.edits)
    assert np.allclose(s.edits["instances"][s.sel_inst]["wxyz"], [1, 0, 0, 0])
    se.undo(s.edits)
    assert np.allclose(s.edits["instances"][s.sel_inst]["position"], [0, 0, 0])


def test_step_buttons_convert_metres_and_turn_direction():
    class Widget:
        def __init__(self, value):
            self.value = value
        def on_click(self, cb):
            self.click = cb
    class Gui:
        def add_slider(self, name, **kwargs):
            return Widget(kwargs["initial_value"])
        def add_button_group(self, name, options, **kwargs):
            w = Widget(options[0])
            setattr(self, name.lower(), w)
            return w
    gui, calls = Gui(), []
    add_nudge_gui(gui, {"nudge_step_m": .2, "nudge_turn_deg": 30}, up_vector("-y"), 2,
                  lambda dp, yaw: calls.append((dp, yaw)))
    gui.lift.click(NS(target=gui.lift))
    assert np.allclose(calls[-1][0], [0, -.1, 0])
    gui.turn.click(NS(target=gui.turn))
    assert np.isclose(calls[-1][1], np.pi / 6)


@pytest.mark.parametrize("text, expected", [
    ("move it 20 cm up", {"cmd": "move", "target": "", "forward_m": 0., "right_m": 0., "up_m": .2}),
    ("move OBJ-0269_copy1 .5 m left and 10 mm back",
     {"cmd": "move", "target": "OBJ-0269_copy1", "forward_m": -.01, "right_m": -.5, "up_m": 0.}),
    ("move it -20 cm up", {"cmd": "move", "target": "", "forward_m": 0., "right_m": 0., "up_m": -.2}),
    ("turn it 30 degrees right", {"cmd": "turn", "target": "", "degrees": -30.}),
    ("turn OBJ-0269 -30 degrees left", {"cmd": "turn", "target": "OBJ-0269", "degrees": -30.}),
    ("rotate this 90 counter-clockwise", {"cmd": "turn", "target": "", "degrees": 90.}),
    ("remove OBJ-0269", {"cmd": "remove", "target": "OBJ-0269"}),
    ("put it back", {"cmd": "restore", "target": ""}),
])
def test_quick_edit_grammar(text, expected):
    from copilot.chat import parse_quick
    assert parse_quick(text) == expected


@pytest.mark.parametrize("text", [
    "remove it only if it is a copy", "remove OBJ-0269 and restore OBJ-0030",
    "move it 1 m left and 20 up unless blocked", "move the green chair 1 m up",
    "rotate it 2 radians right", "move it 2 m left then remove it",
])
def test_qualified_edit_is_not_partially_executed(text):
    from copilot.chat import parse_quick
    assert parse_quick(text) is None


@pytest.mark.parametrize("cmd, field, value", [
    ("move", "up_m", float("nan")), ("move", "right_m", float("inf")),
    ("turn", "degrees", "invalid"), ("turn", "degrees", None),
])
def test_invalid_edit_amount_does_not_change_selection(cmd, field, value):
    s = Sim.__new__(Sim)  # no selection state: validation must run before any mutation
    assert not s.edit_command({"cmd": cmd, field: value})["ok"]
