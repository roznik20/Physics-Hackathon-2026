import math
from scipy.integrate import solve_ivp
import numpy as np
from matplotlib.animation import FuncAnimation

def simulate_pendulum(t_max=10*60, fps=60):
    t_eval = np.linspace(0.0, t_max, int(t_max * fps))
    x = np.linspace(0.0, 0, int(t_max * fps))
    y = x
    return t_eval, x, y