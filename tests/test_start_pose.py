"""Initial worker direction at the door: into the room, away from the door wall."""
import numpy as np

from viewer.app import start_facing


def cfg(facing="auto", wall_z=-3.758):
    return {"viewer": {"start_facing": facing}, "scene": {"back_wall_z": wall_z}}


def test_auto_faces_away_from_door_wall():
    door = np.array([0.0, 1.17, -3.29])
    assert np.allclose(start_facing(cfg(), door), [0, 0, 1])
    assert np.allclose(start_facing(cfg(wall_z=3.1), np.array([0.0, 1.17, 2.9])), [0, 0, -1])


def test_explicit_direction_is_normalised():
    d = start_facing(cfg([3, 4]), np.zeros(3))
    assert np.allclose(d, [0.6, 0, 0.8])
