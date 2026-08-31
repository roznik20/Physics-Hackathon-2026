"""Simple pendulum: rigid rod, point mass (small and large angle).

ODE (theta measured from the downward vertical):
    theta'' = -(g / L) sin(theta)

Returned coordinates are in a y-UP frame with the pivot at the origin:
    x = L sin(theta),  y = -L cos(theta)   (y < 0 while hanging down)
"""
import math

import numpy as np
from scipy.integrate import solve_ivp


def simulate(L=3.0, g=9.81, theta0=1.0, omega0=2.0, t_max=60.0, fps=60):
    def f(_t, y):
        theta, omega = y
        return [omega, -(g / L) * math.sin(theta)]

    n = int(t_max * fps)
    t_eval = np.linspace(0.0, t_max, n)
    sol = solve_ivp(f, (0.0, t_max), [theta0, omega0], t_eval=t_eval,
                    rtol=1e-8, atol=1e-10)
    theta = sol.y[0]
    x = L * np.sin(theta)
    y = -L * np.cos(theta)
    return t_eval, x, y
