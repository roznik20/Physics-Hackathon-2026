"""Diagnostic: what does each launcher actually throw?
For each system, over the full cycle report:
  - max |v| of the driven node (m/s)
  - the release that maximizes horizontal reach, and the ball's x-reach + y at that x
  - the ball's (x,y) envelope: min/max x reached
This tells us how far/right the ball can get, so we can place the hoop to be
reachable-but-timing-dependent.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame
pygame.init()
from game.engine import Game, builtin_run
from game.config import PX_PER_M

W, H = 921, 691
screen = pygame.display.set_mode((W, H))
px = PX_PER_M; g = 9.81; dt = 1/60
ww, wh = W/px, H/px

run = builtin_run()
print(f"{'system':<22} {'maxv':>6} {'reach_x':>8} {'y@reach':>8} {'x_min':>7} {'x_max':>7}  {'bob_x':>6} {'bob_y':>6}")
for li, cfg in enumerate(run, start=1):
    game = Game(W, H, screen, run=run); game.level = li; game._build_level()
    motion = game.motion; root = tuple(game.launcher_root); n = motion.n
    maxv = 0.0
    best_x = -1; best_y = None; x_min = 1e9; x_max = -1e9
    for i in range(0, n, 2):
        p0 = motion.driven_world(i, root); v0 = motion.driven_velocity(i, dt)
        maxv = max(maxv, float(np.hypot(*v0)))
        pos = p0.copy(); vel = v0.copy()
        for step in range(600):
            vel = vel + np.array([0.0, g])*dt; pos = pos + vel*dt
            x_min = min(x_min, float(pos[0])); x_max = max(x_max, float(pos[0]))
            if float(pos[0]) > best_x:
                best_x = float(pos[0]); best_y = float(pos[1])
            if pos[0] < -1 or pos[0] > ww+1 or pos[1] > wh+1:
                break
    bp = motion.driven_world(0, root)
    print(f"{cfg.system:<22} {maxv:>6.2f} {best_x:>8.3f} {best_y:>8.3f} {x_min:>7.2f} {x_max:>7.2f}  {bp[0]:>6.2f} {bp[1]:>6.2f}")
