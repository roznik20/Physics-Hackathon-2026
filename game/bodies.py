"""Game bodies: Ball, Launcher (driven by a Motion), Hoop (fixed target)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import (BALL_RADIUS_M, G, HOOP_RIM_RADIUS_M, HOOP_SCORE_BAND_M,
                     HOOP_POLE_HEIGHT_M, PX_PER_M)
from .motion import Motion


class Ball:
    def __init__(self, pos_m: Tuple[float, float] = (0.0, 0.0),
                 vel_mps: Tuple[float, float] = (0.0, 0.0), radius_m: float = BALL_RADIUS_M):
        self.pos = np.array(pos_m, dtype=float)
        self.vel = np.array(vel_mps, dtype=float)
        self.r = radius_m
        self.attached = True

    def attach_to(self, launcher: "Launcher"):
        self.attached = True
        self.pos = launcher.bob_pos()
        self.vel = np.zeros(2)

    def release_from(self, launcher: "Launcher"):
        self.attached = False
        self.pos = launcher.bob_pos()
        self.vel = launcher.bob_vel()

    def step(self, dt: float, g: float = G):
        if self.attached:
            return
        # y-down world: gravity is +g in y
        self.vel = self.vel + np.array([0.0, g]) * dt
        self.pos = self.pos + self.vel * dt


class Launcher:
    """Holds the ball on the free end of a physics apparatus.

    The driven node's mean is the launcher ROOT (fixed in the world). Each frame
    the ball sits on the driven node and, on release, departs with its velocity —
    exactly the "release a pendulum bob" mechanic, now for every system.
    """

    def __init__(self, motion: Motion, root_m: Tuple[float, float]):
        self.motion = motion
        self.root = np.array(root_m, dtype=float)
        self.idx = 0
        self.n = motion.n
        self.dt = 1.0 / 60.0

    def reset(self):
        self.idx = 0

    def step(self):
        if self.idx < self.n - 1:
            self.idx += 1

    def _i(self) -> int:
        return int(np.clip(self.idx, 0, self.n - 1))

    def bob_pos(self) -> np.ndarray:
        return self.motion.driven_world(self._i(), tuple(self.root))

    def bob_vel(self) -> np.ndarray:
        return self.motion.driven_velocity(self._i(), self.dt)

    def node_world(self, name: str) -> np.ndarray:
        return self.motion.node_world(name, self._i(), tuple(self.root))


class Hoop:
    """Fixed basketball hoop. The rim center is the scoring point; a pole and a
    backboard make it read as a proper, mounted goal (not a floating ring)."""

    def __init__(self, center_m: Tuple[float, float]):
        self.c = np.array(center_m, dtype=float)
        self.rim_radius = HOOP_RIM_RADIUS_M
        self.band = HOOP_SCORE_BAND_M
        self.pole_height_m = HOOP_POLE_HEIGHT_M

    def set_center(self, center_m: Tuple[float, float]):
        self.c = np.array(center_m, dtype=float)

    def scored(self, ball: Ball) -> bool:
        dx = ball.pos[0] - self.c[0]
        dy = ball.pos[1] - self.c[1]
        return (dx * dx + dy * dy) <= (self.rim_radius * self.rim_radius) \
            and abs(ball.pos[1] - self.c[1]) <= self.band
