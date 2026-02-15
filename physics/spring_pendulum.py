import math
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation


def simulate_spring_pendulum(
    m=1.0, k=20.0, L0=1.0, g=9.81,
    r0=1.1, theta0=1, rdot0=1, thetadot0=2.0,
    t_max=600, fps=60):
    def f(t, y):
        # y = [r, theta, r_dot, theta_dot]
        r = y[0]
        theta = y[1]
        rdot = y[2]
        thetadot = y[3]

        # Avoid division by zero if someone sets r ~ 0
        if r < 0.001:
            r = 0.001

        # Equations from Lagrange:
        rddot = r * thetadot * thetadot - (k / m) * (r - L0) + g * math.cos(theta)
        thetaddot = -(2.0 * rdot * thetadot) / r - (g / r) * math.sin(theta)

        return [rdot, thetadot, rddot, thetaddot]

    t_eval = np.linspace(0.0, t_max, int(t_max * fps))
    y0 = [r0, theta0, rdot0, thetadot0]

    sol = solve_ivp(
        f, (0.0, t_max), y0,
        t_eval=t_eval
    )
    r = sol.y[0]
    theta = sol.y[1]

    # Convert to Cartesian (pivot at origin, y up)
    x = r * np.sin(theta)
    y = -r * np.cos(theta)

    return t_eval, r, theta, x, y

import matplotlib.pyplot as plt

t_eval, _, _, x, y = simulate_spring_pendulum()

fig, ax = plt.subplots()
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-2, 2)
ax.set_ylim(-3, 0)

line, = ax.plot([], [], "o-", lw=2, color="tab:blue")

def update(i):
    x_coords = [0.0, x[i]]
    y_coords = [0.0, y[i]]
    line.set_data(x_coords, y_coords)
    return (line,)

ani = FuncAnimation(fig, update, frames=len(t_eval), interval=1000/60, blit=True)
plt.show()