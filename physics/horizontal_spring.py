"""1D horizontal spring-mass (SHM), wall fixed at x = +1.

State: mass position x (meters, y = 0 rail line), wall fixed at x = 1.
    x'' = -(k / m) x
Returned: (t, x_mass, x_wall) where x_wall is a constant array.
"""
import numpy as np
from scipy.integrate import solve_ivp


def simulate(x0=0.5, m=2.0, k=6.0, wall_x=1.0, t_max=60.0, fps=60):
    def f(_t, y):
        x, v = y
        return [v, -(k / m) * x]

    n = int(t_max * fps)
    t_eval = np.linspace(0.0, t_max, n)
    sol = solve_ivp(f, (0.0, t_max), [x0, 0.0], t_eval=t_eval,
                    rtol=1e-8, atol=1e-10)
    x = sol.y[0]
    x_wall = np.full_like(x, wall_x)
    return t_eval, x, x_wall
