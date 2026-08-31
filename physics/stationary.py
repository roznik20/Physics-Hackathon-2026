"""Stationary (fixed) hoop: no driving motion.

Returned: (t, x, y) both constant at 0. The game treats this as a hoop locked
to its base position (the classic static target).
"""
import numpy as np


def simulate(t_max=60.0, fps=60):
    n = int(t_max * fps)
    t_eval = np.linspace(0.0, t_max, n)
    x = np.zeros(n)
    y = np.zeros(n)
    return t_eval, x, y
