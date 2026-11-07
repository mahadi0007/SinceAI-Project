"""Visited equipment retains a steady health highlight until the worker leaves."""
from types import SimpleNamespace

import numpy as np
import pytest

from viewer.app import Sim, visit_style


class Handle(SimpleNamespace):
    def remove(self):
        self.removed = True


class Scene:
    def add_box(self, name, color, dimensions, **kwargs):
        return Handle(name=name, color=color, dimensions=dimensions, visible=True, removed=False, **kwargs)

    def add_label(self, name, text, **kwargs):
        return Handle(name=name, text=text, visible=True, removed=False, **kwargs)


def node(key='A', state='healthy'):
    return dict(id=key, label='pump', asset=None if state is None else
                dict(tag='P-' + key, condition_state=state),
                bbox_min=[-.2, 0, -.2], bbox_max=[.2, .5, .2], center=[0, .25, 0])


@pytest.fixture
def sim():
    s = Sim.__new__(Sim)
    s.nodes = [node(), node('B', 'watch')]
    s.stop_active = s.face_target = None
    s.visit_active = None
    s.visit_arrival_gap = 0.0
    s.h_stop_fill = s.h_stop_label = None
    s.h_stop_box = Handle(visible=False)
    s.h_highlight = Handle(visible=True)
    s.h_selected = Handle(visible=True)
    s.h_all_labels = [Handle(visible=True), Handle(visible=True)]
    s.is_equipment = [True, True]
    s.selected = s.nodes[0]
    s.dd_labels = SimpleNamespace(value='Equipment')
    s.sl_radius = SimpleNamespace(value=1.0)
    s.vc = dict(colors={'healthy': [100, 100, 100], 'watch': [100, 100, 100]},
                stop_fill_pad=.03, stop_fill_opacity=.14, stop_label_scale=1.2,
                visit_release_margin=.25)
    s.server = SimpleNamespace(scene=Scene())
    s.worker_pos = np.array([.8, 1, 0])
    s.position = lambda: s.worker_pos
    s.pose_avatar = lambda *args, **kwargs: None
    return s


@pytest.mark.parametrize('state,expected', [('healthy', 'green'), ('watch', 'red'),
    ('degraded', 'red'), ('attention_required', 'red'), (None, 'neutral'), ('unknown', 'neutral')])
def test_health_colors(state, expected):
    color, title = visit_style(node(state=state))
    if expected == 'green':
        assert color[1] > color[0] and 'HEALTHY' in title
    elif expected == 'red':
        assert color[0] > color[1] and 'HEALTHY' not in title
    else:
        assert 'STATUS UNKNOWN' in title and max(color) - min(color) < 30


def test_highlight_is_steady_after_pause_expires_and_while_walking_away(sim):
    sim.highlight_stop(dict(object_id='A', kind='stop'))
    fill = sim.h_stop_fill
    sim.end_stop()
    assert sim.stop_active is None  # tour briefing state is released independently
    for now in (0.0, .5, 100000.0):
        sim.blink_stop(now)
        assert sim.h_stop_box.visible and fill.visible
    sim.worker_pos = np.array([1.3, 1, 0])  # gap 1.1: just outside entry radius, within release band
    sim.update_visit_highlight(sim.worker_pos)
    assert sim.visit_active['object_id'] == 'A' and not fill.removed
    sim.worker_pos = np.array([1.5, 1, 0])  # gap 1.3: departed
    sim.update_visit_highlight(sim.worker_pos)
    assert sim.visit_active is None and fill.removed and sim.h_stop_label is None
    assert sim.h_all_labels[0].visible and sim.h_selected.visible


def test_arrival_gap_is_respected_when_route_stops_outside_slider_radius(sim):
    sim.worker_pos = np.array([2, 1, 0])
    sim.highlight_stop(dict(object_id='A', kind='goto'))
    sim.update_visit_highlight(sim.worker_pos)
    assert sim.visit_active is not None
    sim.update_visit_highlight(np.array([2.3, 1, 0]))
    assert sim.visit_active is None


def test_next_visit_replaces_old_fill_and_suppresses_duplicate_label(sim):
    sim.highlight_stop(dict(object_id='A', kind='target'))
    old_fill, old_label = sim.h_stop_fill, sim.h_stop_label
    assert not sim.h_all_labels[0].visible and not sim.h_selected.visible
    sim.highlight_stop(dict(object_id='B', kind='goto'))
    assert old_fill.removed and old_label.removed
    assert sim.h_stop_fill.color == (255, 0, 20)
    assert sim.h_all_labels[0].visible and not sim.h_all_labels[1].visible
    assert sim.h_selected.visible and sim.stop_active['object_id'] == 'B'


def test_new_route_keeps_previous_visit_while_worker_is_still_beside_it(sim):
    # Route planning must not remove the highlight before the worker starts to leave.
    sim.highlight_stop(dict(object_id='A', kind='goto'))
    sim.sl_pos = SimpleNamespace(max=0., value=0.)
    sim.draw_path = lambda: None
    sim.set_route([sim.worker_pos, [4, 1, 0]], [])
    assert sim.visit_active['object_id'] == 'A' and sim.h_stop_fill.visible
    assert sim.pause_left == 0 and sim.face_target is None


def test_persistent_visual_does_not_delay_target_briefing(sim):
    sim.stops, sim.stop_s, sim.next_stop = [], [], 0
    sim.playing = False
    sim.highlight_stop(dict(object_id='A', kind='stop'))
    assert sim.warning_stop_due()
    sim.end_stop()
    assert not sim.warning_stop_due()
    assert sim.visit_active['object_id'] == 'A' and sim.h_stop_fill.visible


def test_oren_selection_fill_is_hidden_only_for_the_active_original(sim):
    sim.h_selected_fill = Handle(visible=True)
    sim.highlight_stop(dict(object_id='A', kind='goto'))
    assert not sim.h_selected.visible and not sim.h_selected_fill.visible
    sim.sel_inst = 'A-copy-1'
    sim._sync_visit_labels()
    assert sim.h_selected.visible and sim.h_selected_fill.visible
    sim.sel_inst = None
    sim.clear_stop_highlight()
    assert sim.h_selected_fill.visible
    assert sim.visit_active is None and sim.stop_active is None
