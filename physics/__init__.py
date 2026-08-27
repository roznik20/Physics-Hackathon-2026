"""Physics package.

Each module exposes a ``simulate(...)`` function returning plain numpy arrays in
a y-UP frame (pivot / origin at (0,0) where applicable). The game interprets the
raw channel tuples via :mod:`physics.apparatus` (which builds :class:`MotionResult`
objects) and :mod:`game.apparatus` (which draws them). No pygame or matplotlib is
imported here, so the sims stay importable anywhere.
"""
from . import (  # noqa: F401
    simple_pendulum,
    double_pendulum,
    spring_pendulum,
    pendulum_cart,
    horizontal_spring,
    horizontal_three_pend,
    damped_spring,
    springs_2d,
    stationiary,
    verticle_double_spring,
)
