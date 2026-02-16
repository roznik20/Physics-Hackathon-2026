import math
from scipy.integrate import solve_ivp
import numpy as np
from matplotlib.animation import FuncAnimation
# --- Numerical solution and coordinates ---
import matplotlib.pyplot as plt
from sympy import fps

def simulate_pendulum(b = 0.1, omega=3 ,d=0.3, m= 2, x0=-1, x_dis = -0.2, v_ini = 0, k=8, t_max=100.0, fps=60):
    def f(t, y):
        x, v = y
        
        x_eq = x0 + d * math.sin(omega * t)
        a = -(b/m) * v - (k/m) * (x - x_eq)
        dxdt = v
        dvdt = a
        return [dxdt, dvdt]
    t_eval = np.linspace(0.0, t_max, int(t_max * fps))
    x_ini = x_dis + x0
    sol = solve_ivp(f, (0.0, t_max), [x_ini, v_ini], t_eval=t_eval)
    
    x = sol.y[0]
    x_wall = d * np.sin(omega * t_eval)
    return t_eval, x, x_wall




t, x, x_wall = simulate_pendulum()
fig, ax = plt.subplots()
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-0.2, 0.2)
ax.set_aspect("equal", "box")
ax.grid(True)

mass_line, = ax.plot([], [], "o-", lw=2)
wall_line, = ax.plot([], [], "-", lw=2)

def init():
    mass_line.set_data([], [])
    wall_line.set_data([], [])
    return (mass_line, wall_line)

def update(i):
    xs = [x_wall[i], x[i]]
    ys = [0.0, 0.0]
    mass_line.set_data(xs, ys)
    wall_line.set_data([x_wall[i], x_wall[i]], [-0.2, 0.2])
    return (mass_line, wall_line)

ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=1000 / 60)
plt.show()
