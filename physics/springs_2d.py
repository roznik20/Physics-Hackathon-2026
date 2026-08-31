"""2D springs: a mass held by an independent horizontal and vertical spring.

Two decoupled 1D oscillators drive the x and y motion of the mass:
    x'' = -(k_x / m) x
    y'' = -(k_y / m) y
A horizontal spring anchors to a wall on the right (x = +1) and a vertical
spring anchors to a wall above (y = -1); the mass (hoop) hangs at their free
end and moves in a Lissajous-like figure.

Returned: (t, x, y, vwall_x, vwall_y, hwall_x, hwall_y) where vwall_* is the
vertical spring's wall anchor (0, -1) and hwall_* is the horizontal spring's
wall anchor (1, 0). The extra arrays exist for faithful connector rendering.
"""
import numpy as np
from scipy.integrate import solve_ivp


def simulate(L_ext_x=0.3, L_ext_y=0.6, m=2.0, k_x=4.0, k_y=6.0,
             t_max=60.0, fps=60):
    def f(_t, y, kk):
        x, v = y
        return [v, -(kk / m) * x]

    n = int(t_max * fps)
    t_eval = np.linspace(0.0, t_max, n)
    solx = solve_ivp(lambda t, y: f(t, y, k_x), (0.0, t_max), [L_ext_x, 0.0],
                     t_eval=t_eval, rtol=1e-8, atol=1e-10)
    soly = solve_ivp(lambda t, y: f(t, y, k_y), (0.0, t_max), [L_ext_y, 0.0],
                     t_eval=t_eval, rtol=1e-8, atol=1e-10)
    x = solx.y[0]
    y = soly.y[0]

    vertical_wall_x = np.zeros(n)        # vertical spring anchors at x = 0
    vertical_wall_y = -np.ones(n)        # ... at y = -1 (above, y-up)
    horizontal_wall_x = np.ones(n)       # horizontal spring anchors at x = 1
    horizontal_wall_y = np.zeros(n)      # ... at y = 0
    return (t_eval, x, y,
            vertical_wall_x, vertical_wall_y,
            horizontal_wall_x, horizontal_wall_y)
