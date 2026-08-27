"""Vertical two-mass spring stack (ceiling-s1-m1-s2-m2-s3-floor).

Positions x1, x2 are measured upward from the floor (y = 0); the ceiling sits
at y = total_length = L1 + L2 + L3. Three springs: ceiling->m1, m1->m2,
m2->floor. The lower mass m2 (the hoop) is the driven node.

Returned: (t, x1, x2, total_length).
"""
import numpy as np
from scipy.integrate import solve_ivp


def simulate(m1=1.0, m2=1.3,
             k1=15.0, k2=20.0, k3=20.0,
             L1=2.3, L2=1.7, L3=1.0,
             g=9.81,
             x10=1.0, x20=3.0,
             v10=2.0, v20=-2.3,
             t_max=60.0, fps=60):
    total_length = L1 + L2 + L3

    def f(_t, y):
        x1, x2, v1, v2 = y
        e1 = x1 - L1
        e2 = (x2 - x1) - L2
        e3 = (total_length - x2) - L3
        F1 = -k1 * e1 + k2 * e2 - m1 * g
        F2 = -k2 * e2 + k3 * e3 - m2 * g
        return [v1, v2, F1 / m1, F2 / m2]

    n = int(t_max * fps)
    t_eval = np.linspace(0.0, t_max, n)
    sol = solve_ivp(f, (0.0, t_max), [x10, x20, v10, v20], t_eval=t_eval,
                    rtol=1e-8, atol=1e-10)
    x1, x2 = sol.y[0], sol.y[1]
    return t_eval, x1, x2, total_length
