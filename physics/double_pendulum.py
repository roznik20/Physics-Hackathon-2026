import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

def simulate_pendulum(m1 = 2, m2 = 1, r1 = 1.2, r2= 1, t_max= 600, theta1_0 = 1, omega1_0 = -2, theta2_0 = -2, omega2_0 = 10, fps = 60, g = 9.81):
    def f(t, y):
        t1, w1, t2, w2 = y
        delta = t2 - t1

        denom1 = (m1 + m2) * r1 - m2 * r1 * np.cos(delta) ** 2
        theta1_tt = (
            m2 * r1 * w1 ** 2 * np.sin(delta) * np.cos(delta)
            + m2 * g * np.sin(t2) * np.cos(delta)
            + m2 * r2 * w2 ** 2 * np.sin(delta)
            - (m1 + m2) * g * np.sin(t1)
        ) / denom1

        denom2 = (r2 / r1) * denom1
        theta2_tt = (
            -m2 * r2 * w2 ** 2 * np.sin(delta) * np.cos(delta)
            + (m1 + m2)
            * (
                g * np.sin(t1) * np.cos(delta)
                - r1 * w1 ** 2 * np.sin(delta)
                - g * np.sin(t2)
            )
        ) / denom2

        return [w1, theta1_tt, w2, theta2_tt]
    t_eval = np.linspace(0.0, t_max, int(t_max * fps))
    y0 = [theta1_0, omega1_0, theta2_0, omega2_0]
    sol = solve_ivp(f, (0, t_max), y0, t_eval=t_eval)
    theta1 = sol.y[0]
    theta2 = sol.y[2]
    
    x1 = r1 * np.sin(theta1)
    y1 = -r1 * np.cos(theta1)
    x2 = x1 + r2 * np.sin(theta2)
    y2 = y1 - r2 * np.cos(theta2)
    
    return t_eval, x1, y1, x2, y2

t_eval, x1, y1, x2, y2 = simulate_pendulum()

fig, ax = plt.subplots()
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-2, 2)
ax.set_ylim(-3, 1)

line, = ax.plot([], [], "o-", lw=2, color="tab:blue")

def update(i):
    line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    return line,

ani = FuncAnimation(fig, update, frames=len(t_eval), interval=1000 / 60, blit=True)
plt.show()