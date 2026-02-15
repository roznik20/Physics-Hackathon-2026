import math
from scipy.integrate import solve_ivp
import numpy as np
from matplotlib.animation import FuncAnimation
# --- Numerical solution and coordinates ---
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def simulate_pendulum(
    m1=2.0, m2=2.0, m3=2.0,
    k1=15.0, k2=15.0, k3=15.0,
    L1=0.10, L2=0.10, L3=0.10, # rest lengths
    x10=-0.05, x20=0, x30=0, # displacement towards the wall (to the left)
    v10=0.0, v20=0.0, v30=0.0, # initial velocities
    t_max=200, fps=60):
    
    x1_eq = L1
    x2_eq = L1 + L2
    x3_eq = L1 + L2 + L3
    
    x10 += x1_eq
    x20 += x2_eq
    x30 += x3_eq
    

    def f(t, y):
        x1, x2, x3, v1, v2, v3 = y

        # Spring extensions
        e1 = (x1 - 0.0) - L1 # wall to m1
        e2 = (x2 - x1) - L2  # m1 to m2
        e3 = (x3 - x2) - L3  # m2 to m3

        # Forces (+ is to the right)
        F1 = -k1 * e1 + k2 * e2
        F2 = -k2 * e2 + k3 * e3
        F3 = -k3 * e3

        a1 = F1 / m1
        a2 = F2 / m2
        a3 = F3 / m3

        return [v1, v2, v3, a1, a2, a3]

    t_eval = np.linspace(0.0, t_max, int(t_max * fps))
    y0 = [x10, x20, x30, v10, v20, v30]

    sol = solve_ivp(f, (0.0, t_max), y0, t_eval=t_eval)
    x1 = sol.y[0]
    x2 = sol.y[1]
    x3 = sol.y[2]
    wall_x = np.zeros_like(x1)
    wall_x += 0.5
    
    return t_eval, x1, x2, x3, wall_x



t, x1, x2, x3, wall_x = simulate_pendulum()

fig, ax = plt.subplots()
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.2, 0.2)
ax.set_aspect("equal", "box")
ax.grid(True)

line, = ax.plot([], [], "o-", lw=2)

def init():
    line.set_data([], [])
    return (line,)

def update(i):
    xs = [wall_x[i], x1[i], x2[i], x3[i]]
    ys = [0.0, 0.0, 0.0, 0.0]
    line.set_data(xs, ys)
    return (line,)

ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=1000 / 60)
plt.show()
