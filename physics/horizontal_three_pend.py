"""Horizontal 3-mass spring chain: wall - s1 - m1 - s2 - m2 - s3 - m3.

The third mass (m3) is the hoop mass. Rest lengths L1, L2, L3 place the masses
to the right of a fixed wall at x = 0. Undamped normal modes.
    F1 = -k1 e1 + k2 e2
    F2 = -k2 e2 + k3 e3
    F3 = -k3 e3
Returned: (t, x1, x2, x3, x_wall) with x_wall constant at 0.
"""
import numpy as np
from scipy.integrate import solve_ivp


def simulate(m1=2.0, m2=2.0, m3=2.0,
             k1=15.0, k2=15.0, k3=15.0,
             L1=0.10, L2=0.10, L3=0.10,
             x10=-0.05, x20=0.0, x30=0.0,
             v10=0.0, v20=0.0, v30=0.0,
             t_max=60.0, fps=60):
    def f(_t, y):
        x1, x2, x3, v1, v2, v3 = y
        e1 = x1 - L1
        e2 = (x2 - x1) - L2
        e3 = (x3 - x2) - L3
        F1 = -k1 * e1 + k2 * e2
        F2 = -k2 * e2 + k3 * e3
        F3 = -k3 * e3
        return [v1, v2, v3, F1 / m1, F2 / m2, F3 / m3]

    x10 += L1
    x20 += L1 + L2
    x30 += L1 + L2 + L3

    n = int(t_max * fps)
    t_eval = np.linspace(0.0, t_max, n)
    sol = solve_ivp(f, (0.0, t_max), [x10, x20, x30, v10, v20, v30],
                    t_eval=t_eval, rtol=1e-8, atol=1e-10)
    x1, x2, x3 = sol.y[0], sol.y[1], sol.y[2]
    x_wall = np.zeros_like(x1)
    return t_eval, x1, x2, x3, x_wall
