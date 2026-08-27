"""Apparatus interpreter.

Maps each simulator's raw channel tuples into a :class:`MotionResult` — a set of
NAMED nodes (pivot, bob, cart, wall, ...) plus fixed anchors and connectors — so
the game can draw every system the way it physically looks and so each system can
drive the launcher.

Convention: all node coordinates are in LOCAL meters, y-UP, with the driven node
(mean) at the origin after the game centers them. Fixed anchors are stored as
constant arrays at a fixed offset from the driven node's mean, so they stay
attached to the apparatus no matter how the game re-centers / scales it.

To add a new system:
  1. Add a ``simulate()`` function in its own module under ``physics/``.
  2. Add a ``build_<name>()`` function here.
  3. Register it in ``SYSTEMS`` below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np

from .common import MotionResult, const_point


# ============================================================================
# Build functions (one per system). Each returns a MotionResult.
# ============================================================================

def _mean(a: np.ndarray) -> float:
    return float(np.mean(a))


def _fixed_at(mean: float, offset: float, n: int) -> np.ndarray:
    """A 1-D constant array (n,) at `mean + offset` — used for fixed 1D anchors."""
    return np.full(n, mean + offset)


def build_simple_pendulum(out) -> MotionResult:
    t, x, y = out
    n = len(t)
    mx, my = _mean(x), _mean(y)
    pivot = np.column_stack([np.full(n, mx), np.full(n, my + _amp(x, y))])
    bob = np.column_stack([x, y])
    return MotionResult(
        name="simple_pendulum", label="Simple pendulum", short="Pendulum",
        t=t,
        points={"pivot": pivot, "bob": bob},
        fixed=["pivot"], driven="bob",
        connectors=[("pivot", "bob", "rod")],
        anchors=[("pivot", "ceiling")],
    )


def _amp(x, y) -> float:
    """Approximate rod length from the motion envelope (max distance from mean)."""
    mx, my = _mean(x), _mean(y)
    d = np.hypot(x - mx, y - my)
    return float(np.max(d)) + 0.05


def build_double_pendulum(out) -> MotionResult:
    t, x1, y1, x2, y2 = out
    n = len(t)
    mx, my = _mean(x2), _mean(y2)
    # pivot sits above the driven bob by the full rod length
    L = _amp(x1, y1) + _amp(x2 - x1, y2 - y1)
    pivot = np.column_stack([np.full(n, mx), np.full(n, my + L)])
    mid = np.column_stack([x1, y1])
    bob = np.column_stack([x2, y2])
    return MotionResult(
        name="double_pendulum", label="Double pendulum (chaos)", short="Double pend.",
        t=t,
        points={"pivot": pivot, "mid": mid, "bob": bob},
        fixed=["pivot"], driven="bob",
        connectors=[("pivot", "mid", "rod"), ("mid", "bob", "rod")],
        anchors=[("pivot", "ceiling")],
        extra={"mass_at_mid": np.ones(n)},
    )


def build_spring_pendulum(out) -> MotionResult:
    t, x, y = out
    n = len(t)
    mx, my = _mean(x), _mean(y)
    pivot = np.column_stack([np.full(n, mx), np.full(n, my + _amp(x, y))])
    bob = np.column_stack([x, y])
    return MotionResult(
        name="spring_pendulum", label="Spring pendulum", short="Spring pend.",
        t=t,
        points={"pivot": pivot, "bob": bob},
        fixed=["pivot"], driven="bob",
        connectors=[("pivot", "bob", "spring")],
        anchors=[("pivot", "ceiling")],
    )


def build_pendulum_cart(out) -> MotionResult:
    t, x_pend, y_pend, x_cart, _y_cart = out
    n = len(t)
    mx, my = _mean(x_pend), _mean(y_pend)
    # cart node (moving support on the rail); the pendulum BOB is the driven launcher
    cart = np.column_stack([x_cart, np.full(n, my)])
    bob = np.column_stack([x_pend, y_pend])
    # rail: fixed horizontal line at the cart's mean height, spanning the cart range
    rail_y = my
    rail_x_min = float(np.min(x_cart)) - 0.4
    rail_x_max = float(np.max(x_cart)) + 0.4
    rail_l = np.column_stack([np.full(n, rail_x_min), np.full(n, rail_y)])
    rail_r = np.column_stack([np.full(n, rail_x_max), np.full(n, rail_y)])
    return MotionResult(
        name="pendulum_cart", label="Pendulum cart", short="Pendulum cart",
        t=t,
        points={"cart": cart, "bob": bob, "rail_l": rail_l, "rail_r": rail_r},
        fixed=["rail_l", "rail_r"], driven="bob",
        connectors=[("cart", "bob", "rod")],
        anchors=[("rail", "rail"), ("rail_l", "wall"), ("rail_r", "wall")],
    )


def build_horizontal_spring(out) -> MotionResult:
    t, x, _x_wall = out
    n = len(t)
    mx = _mean(x)
    # wall to the left of the mass by a fixed offset
    off = max(0.5, 2.0 * _amp(x, np.zeros(n)))
    wall = _fixed_at(mx, -off, n)
    mass = x
    # build 2-D nodes: mass on y=0, wall on y=0
    mass2 = np.column_stack([mass, np.zeros(n)])
    wall2 = np.column_stack([wall, np.zeros(n)])
    return MotionResult(
        name="horizontal_spring", label="Horizontal spring", short="H. spring",
        t=t,
        points={"mass": mass2, "wall": wall2},
        fixed=["wall"], driven="mass",
        connectors=[("wall", "mass", "spring")],
        anchors=[("wall", "wall")],
    )


def build_horizontal_three_pend(out) -> MotionResult:
    t, x1, x2, x3, _x_wall = out
    n = len(t)
    m1, m2, m3 = _mean(x1), _mean(x2), _mean(x3)
    # wall to the left of m1 by the mean m1->wall spacing
    off = max(0.3, m1 - 0.0 + 0.3)
    wall = _fixed_at(m1, -off, n)
    mass1 = np.column_stack([x1, np.zeros(n)])
    mass2 = np.column_stack([x2, np.zeros(n)])
    mass3 = np.column_stack([x3, np.zeros(n)])
    wall2 = np.column_stack([wall, np.zeros(n)])
    return MotionResult(
        name="horizontal_three_pend", label="3-mass spring chain", short="3-mass chain",
        t=t,
        points={"mass1": mass1, "mass2": mass2, "mass3": mass3, "wall": wall2},
        fixed=["wall"], driven="mass3",
        connectors=[("wall", "mass1", "spring"), ("mass1", "mass2", "spring"),
                    ("mass2", "mass3", "spring")],
        anchors=[("wall", "wall")],
    )


def build_damped_spring(out) -> MotionResult:
    t, x_mass, x_wall = out
    n = len(t)
    mx = _mean(x_mass)
    # wall oscillates; place it to the LEFT of the mass by a fixed mean offset
    off = max(0.5, _amp(x_wall, x_mass) * 1.5 + 0.4)
    # wall node: centered on the mass's mean minus offset (constant), so it
    # oscillates relative to the mass after centering
    wall2 = np.column_stack([x_wall - (mx + off - _mean(x_wall)), np.zeros(n)])
    mass2 = np.column_stack([x_mass, np.zeros(n)])
    return MotionResult(
        name="damped_spring", label="Driven damped spring", short="Driven spring",
        t=t,
        points={"mass": mass2, "wall": wall2},
        fixed=[], driven="mass",
        connectors=[("wall", "mass", "spring")],
        anchors=[("wall", "wall")],
    )


def build_2d_springs(out) -> MotionResult:
    (t, x, y, vwx, vwy, hwx, hwy) = out
    n = len(t)
    mx, my = _mean(x), _mean(y)
    ax = _amp(x, np.zeros(n))
    ay = _amp(y, np.zeros(n))
    # horizontal spring: wall to the RIGHT of the mass
    h_wall_x = _fixed_at(mx, ax + 0.5, n)
    h_wall_y = np.zeros(n)
    # vertical spring: wall ABOVE the mass
    v_wall_x = np.full(n, mx)
    v_wall_y = _fixed_at(my, ay + 0.5, n)
    mass = np.column_stack([x, y])
    return MotionResult(
        name="springs_2d", label="2D springs", short="2D springs",
        t=t,
        points={"mass": mass, "h_wall": np.column_stack([h_wall_x, h_wall_y]),
                "v_wall": np.column_stack([v_wall_x, v_wall_y])},
        fixed=["h_wall", "v_wall"], driven="mass",
        connectors=[("h_wall", "mass", "spring"), ("v_wall", "mass", "spring")],
        anchors=[("h_wall", "wall"), ("v_wall", "ceiling")],
    )


def build_stationary(out) -> MotionResult:
    t, x, y = out
    n = len(t)
    return MotionResult(
        name="stationary", label="Stationary (fixed)", short="Static",
        t=t,
        points={"mass": np.column_stack([x, y])},
        fixed=["mass"], driven="mass",
        connectors=[],
        anchors=[("mass", "post")],
    )


def build_vertical_double_spring(out) -> MotionResult:
    t, x1, x2, total_length = out
    n = len(t)
    m1, m2 = _mean(x1), _mean(x2)
    a1 = _amp(x1, np.zeros(n))
    a2 = _amp(x2, np.zeros(n))
    # ceiling fixed ABOVE the upper mass; floor fixed BELOW the lower mass
    ceil_y = _fixed_at(m1, a1 + 0.5, n)
    floor_y = _fixed_at(m2, -(a2 + 0.5), n)
    mass1 = np.column_stack([np.zeros(n), x1])
    mass2 = np.column_stack([np.zeros(n), x2])
    ceil = np.column_stack([np.zeros(n), ceil_y])
    floor = np.column_stack([np.zeros(n), floor_y])
    return MotionResult(
        name="vertical_double_spring", label="Vertical 2-mass springs", short="V. springs",
        t=t,
        points={"mass1": mass1, "mass2": mass2, "ceiling": ceil, "floor": floor},
        fixed=["ceiling", "floor"], driven="mass2",
        connectors=[("ceiling", "mass1", "spring"), ("mass1", "mass2", "spring")],
        anchors=[("ceiling", "ceiling"), ("floor", "floor")],
    )


# ============================================================================
# Registry
# ============================================================================
@dataclass
class System:
    id: str
    label: str
    short: str
    module: str
    func: str
    build: Callable
    default_kwargs: Dict[str, Any] = field(default_factory=dict)


SYSTEMS: List[System] = [
    System("horizontal_spring", "Horizontal spring", "H. spring",
           "horizontal_spring", "simulate", build_horizontal_spring,
           {"x0": 0.5, "m": 2.0, "k": 6.0}),
    System("spring_pendulum", "Spring pendulum", "Spring pend.",
           "spring_pendulum", "simulate", build_spring_pendulum,
           {"r0": 1.1, "theta0": 0.9, "rdot0": 1.0, "thetadot0": 1.5, "k": 20.0}),
    System("pendulum_cart", "Pendulum cart", "Pendulum cart",
           "pendulum_cart", "simulate", build_pendulum_cart,
           # v chosen so total horizontal momentum is ~0 (bounds the free cart):
           #   v = -m2*r*omega*cos(theta)/(m1+m2) = -(-0.8)*cos(-0.5)/2
           {"r": 1.0, "theta": -0.5, "omega": -0.8, "x": 0.0, "v": 0.351}),
    System("horizontal_three_pend", "3-mass spring chain", "3-mass chain",
           "horizontal_three_pend", "simulate", build_horizontal_three_pend,
           {"L1": 0.12, "L2": 0.12, "L3": 0.12, "x10": -0.15, "v30": 0.8}),
    System("damped_spring", "Driven damped spring", "Driven spring",
           "damped_spring", "simulate", build_damped_spring,
           {"b": 0.1, "omega": 3.0, "d": 0.3, "x0": -1.0, "x_dis": -0.2, "k": 8.0}),
    System("double_pendulum", "Double pendulum (chaos)", "Double pend.",
           "double_pendulum", "simulate", build_double_pendulum,
           {"theta1_0": 1.0, "omega1_0": -2.0, "theta2_0": -2.0, "omega2_0": 1.0}),
    System("springs_2d", "2D springs", "2D springs",
           "springs_2d", "simulate", build_2d_springs,
           {"L_ext_x": 0.3, "L_ext_y": 0.6, "k_x": 4.0, "k_y": 6.0}),
    System("simple_pendulum", "Simple pendulum", "Pendulum",
           "simple_pendulum", "simulate", build_simple_pendulum,
           {"L": 1.3, "theta0": 0.9, "omega0": 2.0}),
    System("vertical_double_spring", "Vertical 2-mass springs", "V. springs",
           "verticle_double_spring", "simulate", build_vertical_double_spring,
           {"x10": 1.0, "x20": 3.0, "v10": 2.0, "v20": -2.3}),
    System("stationary", "Stationary (fixed)", "Static",
           "stationiary", "simulate", build_stationary, {}),
]


def system_by_id(sid: str) -> System:
    for s in SYSTEMS:
        if s.id == sid:
            return s
    raise KeyError(f"Unknown system id: {sid!r}")


def system_by_index(i: int) -> System:
    return SYSTEMS[i % len(SYSTEMS)]


def run(system: System, t_max: float = 60.0, fps: int = 60) -> MotionResult:
    """Run one system's simulator and return its MotionResult."""
    import importlib
    import inspect
    mod = importlib.import_module(f"physics.{system.module}")
    fn = getattr(mod, system.func)
    sig = inspect.signature(fn)
    kw = dict(system.default_kwargs)
    if "t_max" in sig.parameters:
        kw["t_max"] = t_max
    if "fps" in sig.parameters:
        kw["fps"] = fps
    kw = {k: v for k, v in kw.items() if k in sig.parameters}
    out = fn(**kw)
    return system.build(out)
