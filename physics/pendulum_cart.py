import matplotlib
matplotlib.use("TkAgg")

import scipy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

m1 = 1
m2 = 1
g = 10
r = 1

def fun(t, y):
    th, w, x, v = y
    theta_tt = (((m2*(w**2)*np.sin(th)*np.cos(th)) - ((g/r)*(m1+m2)*np.sin(th)))
                /(m1+m2-(m2*np.cos(th)**2)))
    x_tt = (-(m2*r/(m1+m2)) * ((theta_tt*np.cos(th)) - (w**2)*np.sin(th)))
    return [w, theta_tt, v, x_tt]

y0 = [0, -4, 1, 2]
t_eval = np.linspace(0, 10, 1000)
sol = scipy.integrate.solve_ivp(fun, (0, 10), y0, t_eval=t_eval)

# Extract solution
theta = sol.y[0]
x_cart = sol.y[2]

# Pendulum position
x_pend = x_cart + r * np.sin(theta)
y_pend = -r * np.cos(theta)

# --- Animation ---

fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')

# Cart rectangle
cart_width = 0.6
cart_height = 0.3
cart = plt.Rectangle((0, 0), cart_width, cart_height, fc='black')
ax.add_patch(cart)

# Pendulum rod
line, = ax.plot([], [], 'o-', lw=2)


def update(frame):
    # Update cart position
    cart.set_xy((x_cart[frame] - cart_width / 2, -cart_height / 2))

    # Update pendulum rod
    thisx = [x_cart[frame], x_pend[frame]]
    thisy = [0, y_pend[frame]]
    line.set_data(thisx, thisy)

    return cart, line


ani = FuncAnimation(
    fig,
    update,
    frames=len(t_eval),
    interval=15,
    blit=False
)

plt.show()