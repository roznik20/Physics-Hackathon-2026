from matplotlib.animation import FuncAnimation

import physics.simple_pendulum as sp

t_eval, x, y = sp.simulate_pendulum(0.2, 9.81, 1, 5, 10, 60)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set_aspect("equal", "box")
ax.set_xlim(min(x) - 0.2, max(x) + 0.2)
ax.set_ylim(min(y) - 0.2, max(y) + 0.2)

line, = ax.plot([], [], "o-", lw=2)

def init():
    line.set_data([], [])
    return line,

def update(i):
    line.set_data([0, x[i]], [0, y[i]])
    return line,

ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=20)
plt.show()