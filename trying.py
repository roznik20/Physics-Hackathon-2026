import importlib
import physics.prob_2
importlib.reload(physics.prob_2)

t_eval2, x1, y1, x2, y2 = physics.prob_2.simulate_mid_bearing_pendulum()
print("x1:", x1)