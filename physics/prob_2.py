import math
import sympy as sp
from scipy.integrate import solve_ivp
import numpy as np
from matplotlib.animation import FuncAnimation

# --- Symbolic Lagrangian and EOM ---
t = sp.symbols('t')
m, L, g = sp.symbols('m L g', positive=True)
theta = sp.Function('theta')(t)

T = sp.Rational(1, 2) * m * (L * sp.diff(theta, t))**2
V = m * g * L * (1 - sp.cos(theta))
Lagrangian = sp.simplify(T - V)

EL = sp.simplify(sp.diff(sp.diff(Lagrangian, sp.diff(theta, t)), t) - sp.diff(Lagrangian, theta))
# print(EL)  # uncomment if you want to see it

# --- Numerical solution and coordinates ---
def pendulum_xy(time_t, L_val=0.5, g_val=9.81, theta0=1, omega0=0.0):
    def f(_t, y):
        th, om = y
        return [om, -(g_val / L_val) * math.sin(th)]

    sol = solve_ivp(f, (0.0, time_t), [theta0, omega0], t_eval=[time_t])
    th = sol.y[0, -1]
    x = L_val * math.sin(th)
    y = -L_val * math.cos(th)
    return x, y

for t in [0.000000001, 0.5, 1.0, 1.5, 2.0]:
    print(pendulum_xy(t))

import matplotlib.pyplot as plt

def simulate_pendulum(L_val=0.5, g_val=9.81, theta0=1, omega0=0.0, t_max=10.0, fps=60):
    def f(_t, y):
        th, om = y
        return [om, -(g_val / L_val) * math.sin(th)]

    t_eval = np.linspace(0.0, t_max, int(t_max * fps))
    sol = solve_ivp(f, (0.0, t_max), [theta0, omega0], t_eval=t_eval)
    th = sol.y[0]
    x = L_val * np.sin(th)
    y = -L_val * np.cos(th)
    return t_eval, x, y

t_eval, x, y = simulate_pendulum()

fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 0.2)
line, = ax.plot([], [], 'o-', lw=2)

def update(i):
    line.set_data([0, x[i]], [0, y[i]])
    return line,

ani = FuncAnimation(fig, update, frames=len(t_eval), interval=1000/60, blit=True)
plt.show()