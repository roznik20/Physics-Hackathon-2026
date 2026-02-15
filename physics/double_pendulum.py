import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

def simulate_pendulum(m1 = 2, m2 = 1, r1 = 1.2, r2= 1, t_max= 600, theta1_0 = 1, omega1_0 = 1.23, theta2_0 = -2, omega2_0 = 1, fps = 60, g = 9.81):

    def fun(t, y):
        t1, w1, t2, w2 = y
        delta = t1 - t2
        denom = 2 * m1 + m2 - m2 * np.cos(2 * delta)

        theta1_tt = (
            -g * (2 * m1 + m2) * np.sin(t1)
            - m2 * g * np.sin(t1 - 2 * t2)
            - 2 * np.sin(delta) * m2 * (w2**2 * r2 + w1**2 * r1 * np.cos(delta))
        ) / (r1 * denom)

        theta2_tt = (
            2 * np.sin(delta)
            * (
                w1**2 * r1 * (m1 + m2)
                + g * (m1 + m2) * np.cos(t1)
                + w2**2 * r2 * m2 * np.cos(delta)
            )
        ) / (r2 * denom)

        return [w1, theta1_tt, w2, theta2_tt]
    t_eval = np.linspace(0.0, t_max, int(t_max * fps))
    y0 = [theta1_0, omega1_0, theta2_0, omega2_0]
    sol = solve_ivp(fun, (0, t_max), y0, t_eval=t_eval)

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


'''
def simulate_pendulum(m1 = 1, m2 = 1, r1 = 1, r2= 1, t_max= 600, theta1_0 = 1, omega1_0 = 1, theta2_0 = 1, omega2_0 = 1, fps = 60):

    def fun(t, y):
        t1, w1, t2, w2 = y
        theta1_tt = ((m2*r1*(w1**2)*np.sin(t1-t2)*np.cos(t1-t2)) + (m2*g*np.sin(t2)*np.cos(t1-t2))
                    + (m2*r2*w2**2*np.sin(t1-t2)) - ((m1+m2)*g*np.sin(t1)) / ((m1+m2)*r1 - m2*r1*np.cos(t1-t2)**2))

        theta2_tt = (-(m2*r2*(w2**2)*np.sin(t1-t2)*np.cos(t1-t2)) + ((m1+m2)*((g*np.sin(t1)*np.cos(t1-t2)) -
                    (r1*(w1**2)*np.sin(t1-t2)) - g*np.sin(t2))) / ((m1+m2)*r2 - m2*r2*np.cos(t1-t2)**2))

        return [w1, theta1_tt, w2, theta2_tt]
    t_eval = np.linspace(0.0, t_max, int(t_max * fps))
    y0 = [theta1_0, omega1_0, theta2_0, omega2_0]
    sol = solve_ivp(fun, (0, t_max), y0, t_eval=t_eval)

    theta1 = sol.y[0]
    theta2 = sol.y[2]
    return t_eval, theta1, theta2


'''

'''
 def fun(t, y):
        t1, w1, t2, w2 = y
        theta1_tt = ((m2*r1*(w1**2)*np.sin(t1-t2)*np.cos(t1-t2)) + (m2*g*np.sin(t2)*np.cos(t1-t2))
                    + (m2*r2*w2**2*np.sin(t1-t2)) - ((m1+m2)*g*np.sin(t1)) / ((m1+m2)*r1 - m2*r1*np.cos(t1-t2)**2))

        theta2_tt = (-(m2*r2*(w2**2)*np.sin(t1-t2)*np.cos(t1-t2)) + ((m1+m2)*((g*np.sin(t1)*np.cos(t1-t2)) -
                    (r1*(w1**2)*np.sin(t1-t2)) - g*np.sin(t2))) / ((m1+m2)*r2 - m2*r2*np.cos(t1-t2)**2))

        return [w1, theta1_tt, w2, theta2_tt]

    sol = scipy.integrate.solve_ivp(fun, t_span, y0)

    plt.plot(sol.t, sol.y[0])
    plt.plot(sol.t, sol.y[3])
    plt.show()


'''