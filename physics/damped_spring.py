import math
from scipy.integrate import solve_ivp
import numpy as np
from matplotlib.animation import FuncAnimation
# --- Numerical solution and coordinates ---
import matplotlib.pyplot as plt
from sympy import fps

def simulate_pendelum(b = 0.1, omega=3.14 ,d=1, m= 2, x0=-1, x_dis = 0, v_ini = 0, k=6, t_max=100.0, fps=60):
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
    x_eq = x0 + d * np.sin(omega * t_eval)
    return t_eval, x, x_eq


t, x, x_eq = simulate_pendelum()

fig, ax = plt.subplots()
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-0.2, 0.2)
ax.set_aspect("equal", "box")
ax.grid(True)

line, = ax.plot([], [], "o-", lw=2)

def init():
    line.set_data([], [])
    return (line,)

def update(i):
    xs = [x_eq[i], x[i]]
    ys = [0.0, 0]
    line.set_data(xs, ys)
    return (line,)

ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=1000 / 60)
plt.show()