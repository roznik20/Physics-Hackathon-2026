"""Pendulum cart (coupled cart-pendulum, no external force).

State: theta (pendulum angle from vertical), cart position x.
    theta'' = [ -m2 w^2 sin(theta) cos(theta) - (g/r)(m1+m2) sin(theta) ]
              / [ m1 + m2 - m2 cos(theta)^2 ]
    x''     = -(m2 r/(m1+m2)) [ theta'' cos(theta) - w^2 sin(theta) ]

Returned coordinates are y-UP. The cart sits on a horizontal rail at y = 0; the
pendulum hangs from the cart center:
    x_pend = x_cart + r sin(theta),  y_pend = -r cos(theta)
"""
import numpy as np
from scipy.integrate import solve_ivp


def simulate(m1=1.0, m2=1.0, g=9.81, r=1.0,
             theta=-0.5, omega=-1.0, x=-1.0, v=1.0,
             t_max=60.0, fps=60):
    def f(_t, y):
        th, w, x, v = y
        theta_tt = (
            (-(m2 * (w ** 2) * np.sin(th) * np.cos(th))
             - ((g / r) * (m1 + m2) * np.sin(th)))
            / (m1 + m2 - m2 * np.cos(th) ** 2)
        )
        x_tt = -(m2 * r / (m1 + m2)) * ((theta_tt * np.cos(th)) - (w ** 2) * np.sin(th))
        return [w, theta_tt, v, x_tt]

    n = int(t_max * fps)
    t_eval = np.linspace(0.0, t_max, n)
    sol = solve_ivp(f, (0.0, t_max), [theta, omega, x, v], t_eval=t_eval,
                    rtol=1e-8, atol=1e-10)
    theta = sol.y[0]
    x_cart = sol.y[2]
    y_cart = np.zeros_like(x_cart)
    x_pend = x_cart + r * np.sin(theta)
    y_pend = -r * np.cos(theta)
    return t_eval, x_pend, y_pend, x_cart, y_cart
