"""Open in 3D: the object's own Gaussians on a distant stage, framed for the browser that opened it."""
import contextlib
from types import SimpleNamespace

import numpy as np
import pytest

from viewer.app import INSPECT_STAGE, Sim


class Handle(SimpleNamespace):
    def remove(self):
        self.removed = True


class Scene:
    def __init__(self):
        self.added = []

    def _add(self, name, **kwargs):
        h = Handle(name=name, removed=False, **kwargs)
        self.added.append(h)
        return h

    def add_frame(self, name, **kwargs):
        return self._add(name, **kwargs)

    def add_gaussian_splats(self, name, **kwargs):
        return self._add(name, **kwargs)


class Camera:
    """Like viser's client camera: setting position moves look_at along with it."""

    def __init__(self):
        self._position = np.array([1.0, -1.0, 3.0])
        self.look_at = np.array([0.0, 0.0, 0.0])
        self.fov = 1.0

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, p):
        p = np.asarray(p, np.float64)
        self.look_at = np.asarray(self.look_at) + (p - self._position)
        self._position = p


def obj(key, lo, hi):
    lo, hi = np.array(lo, float), np.array(hi, float)
    return dict(id=key, label='pump', asset=None, bbox_min=lo.tolist(), bbox_max=hi.tolist(),
                center=((lo + hi) / 2).tolist(), extent=(hi - lo).tolist())


@pytest.fixture
def sim():
    s = Sim.__new__(Sim)
    a, b = obj('A', [0, 0, 0], [0.4, 0.4, 0.4]), obj('B', [0.42, 0, 0], [0.8, 0.4, 0.4])
    s.sg = {'objects': [a, b]}
    s.node_by_id = {'A': a, 'B': b}
    s.obj_index = {'A': 0, 'B': 1}
    s.movable = {}
    s.cfg = {'label': {'member_dist': 0.03}}
    s.vc = {'splat_owner': {'max_size': 0.04}, 'render_splats': True}
    g = np.arange(0, 0.41, 0.02)
    grid = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
    s.pick_xyz = np.concatenate([grid, grid + [0.42, 0, 0]])
    s.pick_lab = np.repeat([0, 1], len(grid))
    s._seg_tree = None
    sx = np.concatenate([grid + 0.005, grid + [0.425, 0.005, 0.005], [[5.0, 5.0, 5.0]]])  # A, B, far stray
    s.sp = {'xyz': sx, 'rgb': np.full((len(sx), 3), 0.5), 'opacity': np.ones((len(sx), 1)),
            'cov': np.tile(np.eye(3) * 1e-4, (len(sx), 1, 1))}
    s.server = SimpleNamespace(scene=Scene(), get_clients=lambda: {7: s._client})
    s._client = SimpleNamespace(client_id=7, camera=Camera(), atomic=contextlib.nullcontext)
    s.inspected, s.inspect_cams, s.h_inspect = None, {}, []
    s.b_open3d = [Handle(label='Open in 3D', icon='cube'), Handle(label='Open in 3D', icon='cube')]
    s.md_copilot = Handle(content='')
    return s


def test_object_splat_keeps_only_its_own_gaussians(sim):
    g = sim.object_splat(sim.node_by_id['A'])
    n_grid = (len(sim.sp['xyz']) - 1) // 2
    assert len(g['xyz']) == n_grid
    assert np.abs(g['xyz']).max() <= 0.21                      # relative to A's box centre, B excluded


def test_open_frames_the_stage_and_close_restores_camera(sim):
    cam = sim._client.camera
    before = (cam.position.copy(), np.array(cam.look_at).copy())
    sim.toggle_3d(sim.node_by_id['A'], sim._client)
    np.testing.assert_allclose(cam.look_at, INSPECT_STAGE)
    assert 0 < np.linalg.norm(cam.position - INSPECT_STAGE) < 5
    assert [b.label for b in sim.b_open3d] == ['Back to the room'] * 2
    shown = list(sim.server.scene.added)
    assert [h.name for h in shown] == ['/inspect', '/inspect/splat']

    sim.toggle_3d(None, sim._client)
    np.testing.assert_allclose(cam.position, before[0])
    np.testing.assert_allclose(cam.look_at, before[1])
    assert all(h.removed for h in shown)
    assert sim.inspected is None and sim.inspect_cams == {}
    assert [b.label for b in sim.b_open3d] == ['Open in 3D'] * 2


def test_open_without_scan_points_does_nothing(sim):
    sim.sp['xyz'] = sim.sp['xyz'] + 100.0
    sim.toggle_3d(sim.node_by_id['A'], sim._client)
    assert sim.inspected is None and sim.server.scene.added == []
    assert 'No scan points' in sim.md_copilot.content
