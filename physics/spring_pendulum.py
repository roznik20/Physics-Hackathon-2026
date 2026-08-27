"""Spring pendulum (elastic pendulum): a mass on a spring that also swings.

From the Lagrangian (r = spring length, theta = swing angle, pivot at origin):
    r''   = r theta'^2 - (k / m)(r - L0) + g cos(theta)
    theta'' = -(2 r' theta') / r - (g / r) sin(theta)

Returned coordinates are y-UP, pivot at the origin:
    x = r sin(theta),  y = -r cos(theta)
"""
import math

import numpy as np
from scipy.integrate import solve_ivp


def simulate(m=1.0, k=20.0, L0=1.0, g=9.81,
             r0=1.1, theta0=1.0, rdot0=1.0, thetadot0=2.0,
             t_max=60.0, fps=60):
    def f(_t, y):
        r, theta, rdot, thetadot = y
        if r < 0.001:
            r = 0.001
        rddot = r * thetadot * thetadot - (k / m) * (r - L0) + g * math.cos(theta)
        thetaddot = -(2.0 * rdot * thetadot) / r - (g / r) * math.sin(theta)
        return [rdot, thetadot, rddot, thetaddot]

    n = int(t_max * fps)
    t_eval = np.linspace(0.0, t_max, n)
    sol = solve_ivp(f, (0.0, t_max), [r0, theta0, rdot0, thetadot0], t_eval=t_eval,
                    rtol=1e-8, atol=1e-10)
    r, theta = sol.y[0], sol.y[1]
    x = r * np.sin(theta)
    y = -r * np.cos(theta)
    return t_eval, x, y
