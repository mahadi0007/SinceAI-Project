"""Easier object moving for viewer.app and viewer.bim_view: a gizmo sized to the object and step buttons."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

MOVES = {"Left": (0, -1), "Forward": (1, 0), "Back": (-1, 0), "Right": (0, 1)}


def up_vector(up_direction: str) -> np.ndarray:
    """'-y' -> (0, -1, 0)."""
    v = np.zeros(3)
    v["xyz".index(up_direction[-1])] = -1.0 if up_direction.startswith("-") else 1.0
    return v


def gizmo_scale(extent: float, vc: dict) -> float:
    """Gizmo size from the object's largest side, so its handles reach past the object."""
    return max(vc.get("gizmo_size", 1.2) * extent, vc.get("gizmo_min_scale", 0.3))


def floor_axes(client: Any, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera forward / right projected onto the floor, so 'Forward' moves away from the viewer."""
    f = np.zeros(3)
    if client is not None:
        f = np.asarray(client.camera.look_at, float) - np.asarray(client.camera.position, float)
        f -= f.dot(up) * up
    if np.linalg.norm(f) < 1e-6:                                           # looking straight down / no client
        f = np.roll(np.abs(up), 1)
    f /= np.linalg.norm(f)
    return f, np.cross(f, up)


def add_nudge_gui(gui: Any, vc: dict, up: np.ndarray, units_to_m: float,
                  on_nudge: Callable[[np.ndarray, float], None]) -> None:
    """Step buttons; on_nudge(offset in scene units, yaw in radians about up)."""
    step = gui.add_slider("Step (m)", min=0.01, max=1.0, step=0.01, initial_value=vc.get("nudge_step_m", 0.1))
    turn = vc.get("nudge_turn_deg", 15)
    g_move = gui.add_button_group("Move", list(MOVES), hint="Relative to the camera")
    g_lift = gui.add_button_group("Lift", ["Up", "Down"])
    g_turn = gui.add_button_group("Turn", [f"Left {turn}°", f"Right {turn}°"])

    @g_move.on_click
    def _(e: Any) -> None:
        f, r = floor_axes(client=e.client, up=up)
        a, b = MOVES[e.target.value]
        on_nudge(step.value / units_to_m * (a * f + b * r), 0.0)

    @g_lift.on_click
    def _(e: Any) -> None:
        on_nudge(step.value / units_to_m * (1 if e.target.value == "Up" else -1) * up, 0.0)

    @g_turn.on_click
    def _(e: Any) -> None:
        on_nudge(np.zeros(3), np.radians(turn if e.target.value.startswith("Left") else -turn))
