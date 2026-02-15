import numpy as np
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

def simulate_vertical_2mass(
    m1=1.0, m2=1.3,
    k1=15.0, k2=20.0, k3=20.0,
    L1=1.5, L2=1.5, L3=1.5,
    g=9.81,
    x10=1, x20=3,
    v10=2.0, v20=-2.3,
    t_max=600.0, fps=60
):
    total_length = L1 + L2 + L3

    def f(t, y):
        x1, x2, v1, v2 = y

        #spring extensions
        e1 = x1 - L1
        e2 = (x2 - x1) - L2
        e3 = (total_length - x2) - L3

        F1 = -k1*e1 + k2*e2 - m1*g
        F2 = -k2*e2 + k3*e3 - m2*g
        a1 = F1 / m1
        a2 = F2 / m2
        return [v1, v2, a1, a2]

    t_eval = np.linspace(0.0, t_max, int(t_max * fps))
    y0 = [x10, x20, v10, v20]

    sol = solve_ivp(f, (0, t_max), y0, t_eval=t_eval)
    x1 = sol.y[0]
    x2 = sol.y[1]
    return t_eval, x1, x2, total_length
import matplotlib.pyplot as plt



t_eval, x1, x2, total_length = simulate_vertical_2mass()

fig, ax = plt.subplots()
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-0.2, 0.2)
ax.set_ylim(0.0, total_length)

line, = ax.plot([], [], "o-", lw=2, color="tab:blue")

def update(i):
    line.set_data([0, 0, 0, 0], [0, x1[i], x2[i], total_length])
    return line,

ani = FuncAnimation(fig, update, frames=len(t_eval), interval=1000/60, blit=True)
plt.show()
