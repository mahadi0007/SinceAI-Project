"""Open in 3D: the object's own Gaussians in full scan quality on a distant stage, framed for the browser that
opened it, with view-dependent (SH) colour baked for that browser's eye."""
import contextlib
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.load_splat import SH_C0, SH_C1, sh_rgb
from viewer.app import INSPECT_STAGE, Sim

PROPS = (["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_1", "rot_2", "rot_3", "rot_0",
          "f_dc_0", "f_dc_1", "f_dc_2", "opacity"] + [f"f_rest_{i}" for i in range(45)])


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


def write_ply(path, xyz, opacity, log_scale, rest_r0):
    rows = np.zeros((len(xyz), len(PROPS)), np.float32)
    c = {k: i for i, k in enumerate(PROPS)}
    rows[:, [c["x"], c["y"], c["z"]]] = xyz
    rows[:, [c["scale_0"], c["scale_1"], c["scale_2"]]] = log_scale[:, None]
    rows[:, c["rot_0"]] = 1.0
    rows[:, c["opacity"]] = np.log(opacity / (1 - opacity))
    rows[:, c["f_rest_0"]] = rest_r0                                   # red, first degree-1 coefficient
    head = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(xyz)
            + "".join(f"property float {k}\n" for k in PROPS) + "end_header\n")
    path.write_bytes(head.encode() + rows.tobytes())


@pytest.fixture
def sim(tmp_path):
    s = Sim.__new__(Sim)
    a, b = obj('A', [0, 0, 0], [0.4, 0.4, 0.4]), obj('B', [0.42, 0, 0], [0.8, 0.4, 0.4])
    s.sg = {'objects': [a, b]}
    s.node_by_id = {'A': a, 'B': b}
    s.obj_index = {'A': 0, 'B': 1}
    s.movable = {}
    g = np.arange(0, 0.41, 0.02)
    grid = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
    s.pick_xyz = np.concatenate([grid, grid + [0.42, 0, 0]])
    s.pick_lab = np.repeat([0, 1], len(grid))
    s._seg_tree = s._ply = s._inspect = None
    # Gaussians on A and B, a faint one and a large edge one on A (the room view's filters drop both),
    # a blob bigger than A next to it, and a far stray
    xyz = np.concatenate([grid + 0.005, grid + [0.425, 0.005, 0.005],
                          [[0.2, 0.2, 0.2], [-0.06, 0.2, 0.2], [-0.03, 0.2, 0.2], [5.0, 5.0, 5.0]]])
    opacity = np.full(len(xyz), 0.9)
    opacity[-4] = 0.05
    log_scale = np.full(len(xyz), np.log(0.005))
    log_scale[-3], log_scale[-2] = np.log(0.04), np.log(0.5)
    ply = tmp_path / 'scan.ply'
    write_ply(ply, xyz, opacity, log_scale, rest_r0=np.full(len(xyz), 0.5))
    s.cfg = {'label': {'member_dist': 0.03}, 'paths': {'ply': str(ply)}}
    s.vc = {'splat_owner': {'max_size': 0.04}, 'render_splats': True}
    s.server = SimpleNamespace(scene=Scene(), get_clients=lambda: {7: s._client})
    s._client = SimpleNamespace(client_id=7, camera=Camera(), atomic=contextlib.nullcontext)
    s.inspected, s.inspect_cams, s.h_inspect = None, {}, []
    s.b_open3d = [Handle(label='Open in 3D', icon='cube'), Handle(label='Open in 3D', icon='cube')]
    s.md_copilot = Handle(content='')
    s.n_grid = len(grid)
    return s


def test_object_splat_keeps_faint_and_edge_gaussians_only_of_its_object(sim):
    g = sim.object_splat(sim.node_by_id['A'])
    assert len(g['xyz']) == sim.n_grid + 2                      # + the faint and the large edge Gaussian
    assert g['op'].min() == pytest.approx(0.05)
    assert g['xyz'][:, 0].min() == pytest.approx(-0.26)          # edge Gaussian, relative to A's centre
    assert g['sh_rest'].shape == (len(g['xyz']), 3, 15)


def test_sh_rgb_matches_degree_one_by_hand():
    f_dc = np.zeros((1, 3), np.float32)
    rest = np.zeros((1, 3, 15), np.float32)
    rest[0, 0, 0] = 0.5                                          # red, basis -C1 * y
    up, down = sh_rgb(f_dc, rest, np.array([[0.0, 1.0, 0.0]])), sh_rgb(f_dc, rest, np.array([[0.0, -2.0, 0.0]]))
    assert up[0, 0] == pytest.approx(0.5 - SH_C1 * 0.5)
    assert down[0, 0] == pytest.approx(0.5 + SH_C1 * 0.5)
    assert up[0, 1] == pytest.approx(0.5 + SH_C0 * 0)


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
    assert sim.inspected is None and sim.inspect_cams == {} and sim._inspect is None
    assert [b.label for b in sim.b_open3d] == ['Open in 3D'] * 2


def test_colour_follows_the_viewing_browser(sim):
    sim.toggle_3d(sim.node_by_id['A'], sim._client)
    splat = sim.server.scene.added[1]
    first = splat.rgbs.copy()                                   # baked on open
    cam = sim._client.camera
    cam.position = INSPECT_STAGE + (INSPECT_STAGE - cam.position)    # orbit to the other side
    sim.update_inspect_colors(now=1.0)
    assert not np.allclose(splat.rgbs[:, 0], first[:, 0])        # red changes with the view
    np.testing.assert_allclose(splat.rgbs[:, 1:], first[:, 1:])  # green / blue have no SH here
    kept = splat.rgbs.copy()
    sim.update_inspect_colors(now=1.05)                          # throttled
    np.testing.assert_allclose(splat.rgbs, kept)


def test_open_without_scan_points_does_nothing(sim):
    sim.pick_lab = np.full_like(sim.pick_lab, 1)                # every segmented point belongs to B
    sim.toggle_3d(sim.node_by_id['A'], sim._client)
    assert sim.inspected is None and sim.server.scene.added == []
    assert 'No scan points' in sim.md_copilot.content
