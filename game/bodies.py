"""Game bodies.

Architecture (the core of the game):
  * The **ball** hangs on a simple **pendulum** launcher — the player times the
    release so the ball flies toward the hoop.
  * The **hoop** rides on a **physics system** (a Lagrangian/Newton many-body
    motion). It is the *moving target*. The ball must be released at the right
    instant so the arc meets the rim where it will be.

So "the hoop is on a custom pendulum/system, not the ball."
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .config import (BALL_RADIUS_M, G, HOOP_RIM_RADIUS_M, HOOP_SCORE_BAND_M,
                     PX_PER_M)
from .motion import Motion


class Pendulum:
    """The launcher: an analytic small-angle pendulum.

    ``theta = A cos(omega t + phi)``, ``omega = sqrt(g/L)``. The ball sits on the
    bob (y-down world). This is *not* one of the physics systems — it is always
    the same, and it is what the player aims by timing the release.
    """

    def __init__(self, pivot_m=(1.5, 1.2), L=1.2, A=0.75, g=G, phi=0.0):
        self.pivot = np.array(pivot_m, dtype=float)
        self.L = L
        self.A = A
        self.g = g
        self.phi = phi
        self.t = 0.0

    def omega(self):
        return math.sqrt(max(self.g, 0.0) / max(self.L, 1e-6))

    def theta(self):
        return self.A * math.cos(self.omega() * self.t + self.phi)

    def theta_dot(self):
        return -self.A * self.omega() * math.sin(self.omega() * self.t + self.phi)

    def bob_pos(self) -> np.ndarray:
        th = self.theta()
        return np.array([self.pivot[0] + self.L * math.sin(th),
                         self.pivot[1] + self.L * math.cos(th)])

    def bob_vel(self) -> np.ndarray:
        th = self.theta()
        thd = self.theta_dot()
        return np.array([self.L * thd * math.cos(th),
                         -self.L * thd * math.sin(th)])

    def reset(self):
        self.t = 0.0

    def step(self, dt):
        self.t += dt


class Ball:
    def __init__(self, pos_m=(0.0, 0.0), vel_mps=(0.0, 0.0), radius_m: float = BALL_RADIUS_M):
        self.pos = np.array(pos_m, dtype=float)
        self.vel = np.array(vel_mps, dtype=float)
        self.r = radius_m
        self.attached = True

    def attach_to(self, pend: Pendulum):
        self.attached = True
        self.pos = pend.bob_pos()
        self.vel = np.zeros(2)

    def release_from(self, pend: Pendulum):
        self.attached = False
        self.pos = pend.bob_pos()
        self.vel = pend.bob_vel()

    def step(self, dt: float, g: float = G):
        if self.attached:
            return
        self.vel = self.vel + np.array([0.0, g]) * dt
        self.pos = self.pos + self.vel * dt


class Hoop:
    """Scoring geometry for a (possibly moving) hoop. ``c`` is the rim center in
    world meters (y-down); ``scored`` is a band test around the rim."""

    def __init__(self, center_m, radius_m: float = HOOP_RIM_RADIUS_M):
        self.c = np.array(center_m, dtype=float)
        self.rim_radius = radius_m
        self.band = HOOP_SCORE_BAND_M

    def set_center(self, center_m):
        self.c = np.array(center_m, dtype=float)

    def scored(self, ball: Ball) -> bool:
        dx = ball.pos[0] - self.c[0]
        dy = ball.pos[1] - self.c[1]
        return (dx * dx + dy * dy) <= (self.rim_radius * self.rim_radius) \
            and abs(dy) <= self.band

    def collide(self, ball: Ball, restitution: float = 0.5) -> bool:
        """Simple rim collision: the rim is a horizontal opening whose two ends
        (the rim wire) are circular obstacles. If the ball overlaps an end, push
        it out along the normal and reflect the velocity (with restitution) so
        the ball clanks off the rim instead of passing straight through. Returns
        True if a collision happened (so the engine can play a sound)."""
        wire = 0.03  # rim-wire radius (m)
        hit = False
        for side in (-1.0, 1.0):
            edge = np.array([self.c[0] + side * self.rim_radius, self.c[1]])
            d = ball.pos - edge
            dist = float(np.hypot(d[0], d[1]))
            min_d = ball.r + wire
            if 0.0 < dist < min_d:
                n = d / dist
                # push the ball out of the rim wire
                ball.pos = edge + n * min_d
                # reflect the velocity about the normal
                vn = float(np.dot(ball.vel, n))
                if vn < 0:
                    ball.vel = ball.vel - (1 + restitution) * vn * n
                hit = True
        return hit


class HoopRig:
    """The hoop driven by a physics-system motion.

    ``Motion`` gives the driven node's world offset ``off[i]`` (mean-centered,
    scaled, y-flipped). The hoop's rim is the driven node, placed at
    ``root + off[i]``. ``root`` is the on-screen base center chosen so the whole
    motion envelope stays visible (see ``choose_hoop_base``).
    """

    def __init__(self, motion: Motion, root_m: Tuple[float, float]):
        self.motion = motion
        self.root = np.array(root_m, dtype=float)
        self.idx = 0
        self.hoop = Hoop(self.rim_center())

    def reset(self):
        self.idx = 0

    def step(self):
        if self.idx < self.motion.n - 1:
            self.idx += 1

    def _i(self) -> int:
        return int(np.clip(self.idx, 0, self.motion.n - 1))

    def rim_center(self) -> np.ndarray:
        off = self.motion.off[self.motion.mr.driven][self._i()]
        return self.root + np.array(off, dtype=float)

    def update_hoop(self):
        self.hoop.set_center(self.rim_center())


def choose_hoop_base(motion: Motion, W: int, H: int,
                     hoop_frac: Tuple[float, float],
                     px: float = PX_PER_M) -> Tuple[float, float]:
    """Choose the hoop's base center (world meters, y-down) so the *entire*
    apparatus envelope (every node, not just the driven one) stays on screen.
    Starts at ``hoop_frac`` of the window, then clamps by the full min/max
    displacement across all nodes. Mirrors the original ``choose_base_center_m``
    but generalized to the whole apparatus (so tall stacks like the vertical
    double-spring keep their ceiling and floor visible).
    """
    # full apparatus envelope in offset space (relative to the driven-node mean)
    min_x, min_y, max_x, max_y = motion.envelope()

    MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 120, 120, 90, 60
    lo_x = MARGIN_L - min_x * px
    hi_x = (W - MARGIN_R) - max_x * px
    lo_y = MARGIN_T - min_y * px
    hi_y = (H - MARGIN_B) - max_y * px

    base_px = (hoop_frac[0] * W, hoop_frac[1] * H)
    bx = float(np.clip(base_px[0], lo_x, hi_x)) if lo_x <= hi_x else W / 2
    by = float(np.clip(base_px[1], lo_y, hi_y)) if lo_y <= hi_y else H / 2
    return (bx / px, by / px)
