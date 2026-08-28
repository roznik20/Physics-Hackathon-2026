"""Motion pipeline.

Wraps a :class:`physics.common.MotionResult` (local meters, y-UP) into a
:class:`Motion` that yields node positions in the game's WORLD frame (meters,
y-DOWN, absolute). The driven node's mean is the launcher ROOT; every other node
is a scaled, y-flipped offset from that root, so the whole apparatus repositions
and rescales together.

The scale is chosen so the whole apparatus fits in the LEFT half of the screen
without reaching the hoop, while giving the driven node a target peak amplitude.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from physics.common import MotionResult

from .config import (AMP_PER_LEVEL_M, BASE_LAUNCHER_AMP_M, HOOP_FRAC,
                     LAUNCHER_FRAC, MAX_LAUNCHER_AMP_M, MOTION_T_MAX, PX_PER_M)


class Motion:
    def __init__(self, mr: MotionResult, target_amp: float,
                 W: int, H: int, launcher_frac: Tuple[float, float] = LAUNCHER_FRAC,
                 hoop_frac: Tuple[float, float] = HOOP_FRAC):
        self.mr = mr
        self.t = mr.t
        self.n = len(mr.t)
        self.driven = mr.driven
        self.fixed = list(mr.fixed)
        self.connectors = list(mr.connectors)
        self.anchors = list(mr.anchors)
        self.label = mr.label

        # --- scale: fit the driven node to the target amplitude, but never let
        # the whole apparatus reach the hoop, leave the left half, or overflow
        # the screen vertically. ---
        mean = mr.points[mr.driven].mean(axis=0)
        d_off = mr.points[mr.driven] - mean
        max_abs = float(max(np.max(np.abs(d_off)), 1e-9))
        scale_amp = target_amp / max_abs

        # spread of ALL nodes (offset space)
        allx = np.concatenate([p[:, 0] for p in mr.points.values()])
        ally = np.concatenate([p[:, 1] for p in mr.points.values()])
        hspread = float(allx.max() - allx.min())
        vspread = float(ally.max() - ally.min())
        # available room: horizontal from left margin to ~72% of the hoop x;
        # vertical is the same placeable height that choose_hoop_base uses
        # (window minus its top/bottom margins), so the apparatus is scaled to
        # fit exactly where it will actually be centered.
        margin = 24.0 / PX_PER_M
        avail_w = (hoop_frac[0] - 0.07) * (W / PX_PER_M) - margin * 2
        avail_h = (H - 150) / PX_PER_M   # matches choose_hoop_base margins
        scale_fit_w = avail_w / max(hspread, 1e-6)
        scale_fit_h = avail_h / max(vspread, 1e-6)

        self.scale = float(min(scale_amp, scale_fit_w, scale_fit_h, 1.6))
        self.target_amp = float(target_amp)

        # World-frame (y-down) offsets, relative to the driven node's mean.
        self.off: Dict[str, np.ndarray] = {}
        for name, p in mr.points.items():
            o = p - mean
            self.off[name] = np.column_stack([o[:, 0] * self.scale, -o[:, 1] * self.scale])

    # -- trajectory ----------------------------------------------------------
    def node(self, name: str, i: int) -> np.ndarray:
        return self.off[name][i]

    def driven_world(self, i: int, root: Tuple[float, float]) -> np.ndarray:
        return np.array([root[0], root[1]]) + self.off[self.driven][i]

    def node_world(self, name: str, i: int, root: Tuple[float, float]) -> np.ndarray:
        return np.array([root[0], root[1]]) + self.off[name][i]

    def driven_velocity(self, i: int, dt: float) -> np.ndarray:
        i = int(np.clip(i, 0, self.n - 1))
        j = min(i + 1, self.n - 1)
        k = max(i - 1, 0)
        a, b = self.off[self.driven][j], self.off[self.driven][k]
        span = (j - k) * dt
        return (a - b) / span if span > 0 else np.zeros(2)

    def envelope(self) -> Tuple[float, float, float, float]:
        xs = np.concatenate([o[:, 0] for o in self.off.values()])
        ys = np.concatenate([o[:, 1] for o in self.off.values()])
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    def root_position(self, W: int, H: int,
                      launcher_frac: Tuple[float, float] = LAUNCHER_FRAC) -> Tuple[float, float]:
        """Anchor the root at the launcher screen position, clamped vertically so
        the apparatus stays on screen. Horizontal is fixed (left side)."""
        minx, miny, maxx, maxy = self.envelope()
        m = 24.0 / PX_PER_M
        root_x = (launcher_frac[0] * W) / PX_PER_M
        lo_y, hi_y = m - miny, (H / PX_PER_M) - m - maxy
        want_y = (launcher_frac[1] * H) / PX_PER_M
        root_y = want_y if lo_y <= want_y <= hi_y else float((min(lo_y, hi_y) + max(lo_y, hi_y)) / 2)
        return (root_x, root_y)


def launcher_amp_for_level(level: int) -> float:
    return min(BASE_LAUNCHER_AMP_M + AMP_PER_LEVEL_M * (max(1, int(level)) - 1),
               MAX_LAUNCHER_AMP_M)
