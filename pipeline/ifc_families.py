"""Parametric equipment families for the IFC export (option B: replace scan clusters with catalog shapes).

Each family fits a box of dims (L, W, H) in metres, local frame: x = long horizontal axis, z = up,
origin = bottom centre. Parts are (kind, args) tuples the writer turns into extruded solids:
  ("cyl", base, axis, radius, length)   ("box", bottom_centre, (dx, dy, dz))
Shapes are representative, not measured.
"""
from __future__ import annotations

import numpy as np

X, Y, Z = np.eye(3)
Part = tuple


def pump(L: float, W: float, H: float) -> list[Part]:
    """End-suction centrifugal pump on a baseplate with its motor."""
    s = min(W, H)
    zc = 0.5 * H
    return [
        ("box", np.zeros(3), (L, W, 0.08 * H)),                                   # baseplate
        ("cyl", np.array([-0.48 * L, 0, zc]), X, 0.32 * s, 0.5 * L),             # motor
        ("box", np.array([0.07 * L, 0, zc - 0.12 * s]), (0.1 * L, 0.24 * s, 0.24 * s)),   # coupling guard
        ("cyl", np.array([0.14 * L, 0, zc]), X, 0.44 * s, 0.18 * L),             # volute
        ("cyl", np.array([0.32 * L, 0, zc]), X, 0.14 * s, 0.18 * L),             # suction nozzle
        ("cyl", np.array([0.23 * L, 0, zc + 0.3 * s]), Z, 0.12 * s, H - zc - 0.3 * s),     # discharge
    ]


def valve(L: float, W: float, H: float) -> list[Part]:
    """Control valve: flanged globe body in the line, bonnet, diaphragm actuator on top."""
    s = min(L, W)
    zc = min(0.3 * H, 0.5 * W)
    r = 0.22 * min(W, H)
    return [
        ("cyl", np.array([-0.5 * L, 0, zc]), X, 0.5 * r, L),                     # line
        ("cyl", np.array([-0.5 * L, 0, zc]), X, r, 0.08 * L),                    # flanges
        ("cyl", np.array([0.42 * L, 0, zc]), X, r, 0.08 * L),
        ("cyl", np.array([-0.18 * L, 0, zc]), X, 0.85 * r, 0.36 * L),            # body
        ("cyl", np.array([0, 0, zc]), Z, 0.12 * s, 0.68 * H - zc),               # bonnet / yoke
        ("cyl", np.array([0, 0, 0.68 * H]), Z, 0.42 * s, 0.22 * H),              # actuator
    ]


def motor(L: float, W: float, H: float) -> list[Part]:
    """Foot-mounted induction motor: feet, finned body, fan cover, shaft, terminal box."""
    s = min(W, H)
    zc = 0.48 * H
    return [
        ("box", np.zeros(3), (0.7 * L, 0.85 * W, 0.12 * H)),                     # feet
        ("cyl", np.array([-0.38 * L, 0, zc]), X, 0.44 * s, 0.68 * L),            # body
        ("cyl", np.array([-0.5 * L, 0, zc]), X, 0.4 * s, 0.12 * L),              # fan cover
        ("cyl", np.array([0.3 * L, 0, zc]), X, 0.07 * s, 0.2 * L),               # shaft
        ("box", np.array([0, 0, zc + 0.36 * s]), (0.22 * L, 0.3 * W, 0.22 * s)),  # terminal box
    ]


def sensor(L: float, W: float, H: float) -> list[Part]:
    """Transmitter: process stem, electronics housing with a display cap."""
    s = min(L, W, H)
    return [
        ("cyl", np.zeros(3), Z, 0.12 * s, 0.5 * H),                              # stem
        ("box", np.array([0, 0, 0.45 * H]), (0.5 * s, 0.5 * s, 0.35 * H)),        # housing
        ("cyl", np.array([0, 0, 0.62 * H]), X, 0.22 * s, 0.35 * s),              # display cap
    ]


def flange(radius: float) -> list[Part]:
    """Flange pair on a pipe of the given radius, centred on the local origin along x."""
    t, R = max(0.4 * radius, 0.012), 1.8 * radius
    return [
        ("cyl", np.array([-t - 0.002, 0, 0]), X, R, t),
        ("cyl", np.array([0.002, 0, 0]), X, R, t),
        ("cyl", np.array([-2.5 * t, 0, 0]), X, radius * 1.05, 5 * t),           # hub through both
    ]


FAMILIES = {"IfcPump": pump, "IfcValve": valve, "IfcElectricMotor": motor, "IfcSensor": sensor}
