"""Driven, damped 1D spring-mass.

A wall oscillates sinusoidally (the driver); a spring connects it to a mass.
The mass obeys a driven-damped oscillator about the moving wall:
    x_eq(t) = x0 + d sin(omega t)          (wall / equilibrium position)
    x'' = -(b/m) x' - (k/m)(x - x_eq(t))
Returned: (t, x_mass, x_wall) where x_wall = x_eq (the moving wall position).
"""
import math

import numpy as np
from scipy.integrate import solve_ivp


def simulate(b=0.1, omega=3.0, d=0.3, m=2.0,
             x0=-1.0, x_dis=-0.2, v_ini=0.0, k=8.0,
             t_max=60.0, fps=60):
    def f(t, y):
        x, v = y
        x_eq = x0 + d * math.sin(omega * t)
        a = -(b / m) * v - (k / m) * (x - x_eq)
        return [v, a]

    n = int(t_max * fps)
    t_eval = np.linspace(0.0, t_max, n)
    sol = solve_ivp(f, (0.0, t_max), [x0 + x_dis, v_ini], t_eval=t_eval,
                    rtol=1e-8, atol=1e-10)
    x = sol.y[0]
    x_wall = x0 + d * np.sin(omega * t_eval)
    return t_eval, x, x_wall
