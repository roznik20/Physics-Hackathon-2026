import math
from scipy.integrate import solve_ivp
import numpy as np
from matplotlib.animation import FuncAnimation
# --- Numerical solution and coordinates ---
import matplotlib.pyplot as plt
from sympy import fps

def simulate_pendulum(L_extention_x = 0.3, L_extention_y = 0.6, m= 2, k_x=4, k_y=6, t_max=600, fps=60):
    def f(t, y):
        x = y[0]
        v = y[1]
        dxdt = v
        dvdt = -(k_x / m) * x
        return [dxdt, dvdt]
    t_eval = np.linspace(0.0, t_max, int(t_max * fps))
    sol = solve_ivp(f, (0.0, t_max), [L_extention_x, 0], t_eval=t_eval)
    x = sol.y[0]
    
    def f(t, y):
        x = y[0]
        v = y[1]
        dxdt = v
        dvdt = -(k_y / m) * x
        return [dxdt, dvdt]
    t_eval = np.linspace(0.0, t_max, int(t_max * fps))
    sol = solve_ivp(f, (0.0, t_max), [L_extention_y, 0], t_eval=t_eval)
    y = sol.y[0]
    
    vertical_spring_y = np.zeros_like(x)
    vertical_spring_y -= 1
    vertical_spring_x = x
    
    
    horizontal_spring_x = np.zeros_like(y)
    horizontal_spring_x += 1
    
    horizontal_spring_y = y
    
    return t_eval, x, y, vertical_spring_x, vertical_spring_y, horizontal_spring_x, horizontal_spring_y
t_eval, x, y, vertical_spring_x, vertical_spring_y, horizontal_spring_x, horizontal_spring_y = simulate_pendulum()

fig, ax = plt.subplots()
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

line, = ax.plot([], [], "o-", lw=2, color="tab:blue")
def update(i):
    x_coords = [vertical_spring_x[i], x[i], horizontal_spring_x[i]]
    y_coords = [vertical_spring_y[i], y[i], horizontal_spring_y[i]]
    line.set_data(x_coords, y_coords)
    return (line,)

ani = FuncAnimation(fig, update, frames=len(t_eval), interval=1000/60, blit=True)
plt.show()
