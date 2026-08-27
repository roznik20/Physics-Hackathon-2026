"""Shared types for the physics package.

Every simulator returns plain numpy arrays (no pygame/matplotlib), so the game
can import them directly. The game layer (``game.apparatus``) is the only place
that knows how to interpret the raw channel tuples into named nodes + connectors.

Coordinate convention:
  - All positions are in METERS.
  - Physics modules use a y-UP frame (standard for the ODEs: ``y = -L cos theta``).
  - The game converts to its y-DOWN world frame in one place (``game.motion``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class MotionResult:
    """A fully resolved apparatus trajectory in LOCAL meters (y-up).

    ``points`` maps a node name to an ``(N, 2)`` array of local positions.
    ``fixed`` lists node names that are static anchors (walls, pivots, rails).
    ``driven`` is the node name whose position drives the game (the launcher bob,
    or the hoop center if a map drives the hoop).
    ``connectors`` are ``(from, to, kind)`` edges for faithful rendering.
    ``anchors`` are ``(name, kind)`` markers for fixed supports to draw.
    """
    name: str
    t: np.ndarray
    points: Dict[str, np.ndarray]
    fixed: List[str] = field(default_factory=list)
    driven: str = ""
    connectors: List[Tuple[str, str, str]] = field(default_factory=list)
    anchors: List[Tuple[str, str]] = field(default_factory=list)
    label: str = ""
    short: str = ""
    extra: Dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.t.shape[0])

    def node(self, name: str, i: int) -> np.ndarray:
        return self.points[name][i]

    def envelope(self) -> Tuple[float, float, float, float]:
        """Bounding box of every node (local meters): (minx, miny, maxx, maxy)."""
        allx = np.concatenate([p[:, 0] for p in self.points.values()])
        ally = np.concatenate([p[:, 1] for p in self.points.values()])
        return float(allx.min()), float(ally.min()), float(allx.max()), float(ally.max())


def const_point(value: Tuple[float, float], n: int) -> np.ndarray:
    """Return an (n,2) array of a constant point (for fixed anchors)."""
    out = np.zeros((n, 2))
    out[:, 0] = value[0]
    out[:, 1] = value[1]
    return out
