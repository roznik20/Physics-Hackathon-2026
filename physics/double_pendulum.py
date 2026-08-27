"""Double pendulum (the chaotic one).

Standard two-link equations, theta measured from the downward vertical for each
link. The second bob hangs from the first bob.

Returned coordinates are y-UP, pivot at the origin:
    x1 = L1 sin(theta1),  y1 = -L1 cos(theta1)
    x2 = x1 + L2 sin(theta2),  y2 = y1 - L2 cos(theta2)
"""
import numpy as np
from scipy.integrate import solve_ivp


def simulate(m1=2.0, m2=1.0, L1=1.2, L2=1.0,
             theta1_0=1.0, omega1_0=-2.0, theta2_0=-2.0, omega2_0=1.0,
             g=9.81, t_max=60.0, fps=60):
    def f(_t, y):
        t1, w1, t2, w2 = y
        delta = t2 - t1
        denom1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2
        theta1_tt = (
            m2 * L1 * w1 ** 2 * np.sin(delta) * np.cos(delta)
            + m2 * g * np.sin(t2) * np.cos(delta)
            + m2 * L2 * w2 ** 2 * np.sin(delta)
            - (m1 + m2) * g * np.sin(t1)
        ) / denom1
        denom2 = (L2 / L1) * denom1
        theta2_tt = (
            -m2 * L2 * w2 ** 2 * np.sin(delta) * np.cos(delta)
            + (m1 + m2)
            * (g * np.sin(t1) * np.cos(delta)
               - L1 * w1 ** 2 * np.sin(delta)
               - g * np.sin(t2))
        ) / denom2
        return [w1, theta1_tt, w2, theta2_tt]

    n = int(t_max * fps)
    t_eval = np.linspace(0.0, t_max, n)
    sol = solve_ivp(f, (0.0, t_max),
                    [theta1_0, omega1_0, theta2_0, omega2_0], t_eval=t_eval,
                    rtol=1e-8, atol=1e-10)
    theta1, theta2 = sol.y[0], sol.y[2]
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return t_eval, x1, y1, x2, y2
