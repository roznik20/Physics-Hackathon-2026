import scipy
import matplotlib.pyplot as plt
import numpy as np

g = 10
t_span = (0, 30)
y0 = [1, 1, 1, 1]

m2 = 1
r1 = 1
m1 = 1
r2 = 1

def fun(t, y):
    t1, w1, t2, w2 = y
    theta1_tt = (((m2*r1*(w1**2)*np.sin(t1-t2)*np.cos(t1-t2)) + (m2*g*np.sin(t2)*np.cos(t1-t2))
                 + (m2*r2*w2**2*np.sin(t1-t2)) - ((m1+m2)*g*np.sin(t1))) / ((m1+m2)*r1 - m2*r1*np.cos(t1-t2)**2))

    theta2_tt = (((-(m2*r2*(w2**2)*np.sin(t1-t2)*np.cos(t1-t2)) + ((m1+m2)*((g*np.sin(t1)*np.cos(t1-t2)) -
                (r1*(w1**2)*np.sin(t1-t2)) - g*np.sin(t2)))) / ((m1+m2)*r2 - m2*r2*np.cos(t1-t2)**2)))

    return [w1, theta1_tt, w2, theta2_tt]

sol = scipy.integrate.solve_ivp(fun, t_span, y0, t_eval=np.linspace(0, 30, 1000))

plt.plot(sol.t, sol.y[0])
plt.plot(sol.t, sol.y[3])
plt.show()

