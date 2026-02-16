import matplotlib
matplotlib.use("TkAgg")

import scipy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation




def simulate_pendulum(m1 =1 , m2 = 1, g=9.81, r=1, theta = -.5, omega = -1.0, x = -1, v = 1, t_max=600, fps=60):

    def f(t, y):
        th, w, x, v = y
        theta_tt = (((m2*(w**2)*np.sin(th)*np.cos(th)) - ((g/r)*(m1+m2)*np.sin(th)))
                    /(m1+m2-(m2*np.cos(th)**2)))
        x_tt = (-(m2*r/(m1+m2)) * ((theta_tt*np.cos(th)) - (w**2)*np.sin(th)))
        return [w, theta_tt, v, x_tt]

    y0 = [theta, omega, x, v]
    t_eval = np.linspace(0, t_max, int(t_max * fps))
    sol = scipy.integrate.solve_ivp(f, (0, t_max), y0, t_eval=t_eval)

    theta = sol.y[0]
    x_cart = sol.y[2]
    y_cart = np.zeros_like(x_cart) 

    x_pend = x_cart + r * np.sin(theta) 
    y_pend = -r * np.cos(theta)
    return t_eval, x_pend, y_pend, x_cart, y_cart

t, x_pend, y_pend, x_cart, y_cart = simulate_pendulum()

fig, ax = plt.subplots()
ax.set_xlim(-1,1)
ax.set_ylim(-1, 1)
ax.set_aspect("equal", adjustable="box")
line, = ax.plot([], [], "o-", lw=2)

def init():
    line.set_data([], [])
    return (line,)

def update(i):
    line.set_data([x_cart[i], x_pend[i]], [y_cart[i], y_pend[i]])
    return (line,)

ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=1000 / 60)
plt.show()