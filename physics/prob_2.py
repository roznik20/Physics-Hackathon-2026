import math
from scipy.integrate import solve_ivp
import numpy as np
from matplotlib.animation import FuncAnimation
# --- Numerical solution and coordinates ---
import matplotlib.pyplot as plt

def simulate_pendulum(L_val=0.5, g_val=9.81, theta0=1, omega0=5, t_max=10.0, fps=60):
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
#plt.show()

'''

def simulate_mid_bearing_pendulum(L=1.0, m1=2.0, m2=6.0, g=9.81, theta1_0=1, omega1_0=2, theta2_0=1, omega2_0=3, t_max=100, fps=60):
    L1 = L / 2.0
    L2 = L / 2.0

    def f(_t, y):
        th1, om1, th2, om2 = y
        delta = th1 - th2
        den1 = L1 * (2 * m1 + m2 - m2 * math.cos(2 * delta))
        den2 = L2 * (2 * m1 + m2 - m2 * math.cos(2 * delta))

        domega1 = (
            -g * (2 * m1 + m2) * math.sin(th1)
            - m2 * g * math.sin(th1 - 2 * th2)
            - 2 * math.sin(delta) * m2 * (om2**2 * L2 + om1**2 * L1 * math.cos(delta))
        ) / den1

        domega2 = (
            2 * math.sin(delta)
            * (om1**2 * L1 * (m1 + m2) + g * (m1 + m2) * math.cos(th1) + om2**2 * L2 * m2 * math.cos(delta))
        ) / den2

        return [om1, domega1, om2, domega2]

    t_eval = np.linspace(0.0, t_max, int(t_max * fps))
    sol = solve_ivp(f, (0.0, t_max), [theta1_0, omega1_0, theta2_0, omega2_0], t_eval=t_eval)

    th1, th2 = sol.y[0], sol.y[2]
    x1, y1 = L1 * np.sin(th1), -L1 * np.cos(th1)
    x2, y2 = x1 + L2 * np.sin(th2), y1 - L2 * np.cos(th2)
    return t_eval, x1, y1, x2, y2

t_eval2, x1, y1, x2, y2 = simulate_mid_bearing_pendulum()

fig2, ax2 = plt.subplots()
ax2.set_aspect('equal', adjustable='box')
ax2.set_xlim(-1.2, 1.2)
ax2.set_ylim(-1.2, 0.2)
line1, = ax2.plot([], [], 'o-', lw=2, color='tab:blue')
line2, = ax2.plot([], [], 'o-', lw=2, color='tab:orange')

def update2(i):
    line1.set_data([0, x1[i]], [0, y1[i]])
    line2.set_data([x1[i], x2[i]], [y1[i], y2[i]])
    return line1, line2

ani2 = FuncAnimation(fig2, update2, frames=len(t_eval2), interval=1000/60, blit=True)
print(t_eval2)
#plt.show()


'''