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
