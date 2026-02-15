import math
from scipy.integrate import solve_ivp
import numpy as np
from matplotlib.animation import FuncAnimation
# --- Numerical solution and coordinates ---
import matplotlib.pyplot as plt
from sympy import fps

def simulate_pendelum(L_extention = 0.5, m= 2, k=6, t_max=600, fps=60):
    def f(t, y):
        x = y[0]
        v = y[1]
        dxdt = v
        dvdt = -(k / m) * x
        return [dxdt, dvdt]
    t_eval = np.linspace(0.0, t_max, int(t_max * fps))
    sol = solve_ivp(f, (0.0, t_max), [L_extention, 0], t_eval=t_eval)
    x = sol.y[0]
    x_wall = np.zeros_like(x)
    x_wall += 1
    return t_eval, x, x_wall

#put a wall on x coordiant 1, or shift it as buch as u want it to the right
t, x, x_wall = simulate_pendelum()

fig, ax = plt.subplots()
ax.set_xlim(-1.2 * np.max(np.abs(x)), 1.2)
ax.set_ylim(-0.5, 0.5)
ax.set_aspect("equal", adjustable="box")
line, = ax.plot([], [], "o-", lw=2)

def init():
    line.set_data([], [])
    return (line,)

def update(i):
    line.set_data([x_wall[i], x[i]], [0, 0])
    return (line,)

ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=1000 / 60)
plt.show()
