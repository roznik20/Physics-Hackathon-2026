import math
import sympy as sp
from scipy.integrate import solve_ivp

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
def pendulum_xy(time_t, L_val=1.0, g_val=9.81, theta0=0.2, omega0=0.0):
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

    array = [
        (0.19866933079506122, -0.9800665778412416),
        (0.0017213706709042669, -0.9999985184404092),
        (-0.1986237848960794, -0.9800758093502543),
        (-0.004958700978791691, -0.9999877055667249),
        (0.19853028380407317, -0.9800947537930575),
    ]