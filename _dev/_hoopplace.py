"""Find a hoop height (world y) such that EVERY system's best-throw trajectory
crosses that y at an on-screen x to the right of the launcher. Then the hoop can
be auto-placed at (x_cross, y_target) per system -> winnable by timing.
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
print(f"world {ww:.2f} x {wh:.2f} m ; launcher frac x={0.22} -> {0.22*ww:.2f} m")

run = builtin_run()

def best_throw(motion, root, n):
    """release frame with max +vx; return the full trajectory points."""
    best_i, best_vx = 0, -1
    for i in range(n):
        v = motion.driven_velocity(i, dt)
        if v[0] > best_vx:
            best_vx = v[0]; best_i = i
    p0 = motion.driven_world(best_i, root); vel = motion.driven_velocity(best_i, dt).copy()
    pos = p0.copy()
    pts = [pos.copy()]
    for _ in range(720):
        vel = vel + np.array([0.0, g])*dt; pos = pos + vel*dt
        pts.append(pos.copy())
        if pos[0] < -1 or pos[0] > ww+1 or pos[1] > wh+1:
            break
    return np.array(pts), best_i

def x_at_y(pts, y):
    """x of the (descending) crossing of height y. Returns None if never."""
    for k in range(1, len(pts)):
        if (pts[k-1,1] <= y < pts[k,1]) or (pts[k-1,1] >= y > pts[k,1] and pts[k,0] > pts[k-1,0]):
            # interpolate
            y0,y1 = pts[k-1,1], pts[k,1]; x0,x1 = pts[k-1,0], pts[k,0]
            if y1==y0: t=0.5
            else: t=(y-y0)/(y1-y0)
            return x0 + t*(x1-x0)
    return None

launcher_x = 0.22*ww
for ytarget in [1.6, 1.8, 2.0, 2.2, 2.4]:
    ok=True; xs=[]
    for li, cfg in enumerate(run, start=1):
        game=Game(W,H,screen,run=run); game.level=li; game._build_level()
        pts,bi = best_throw(game.motion, tuple(game.launcher_root), game.motion.n)
        x = x_at_y(pts, ytarget)
        good = (x is not None and x > launcher_x+0.3 and x < ww-0.2)
        if not good: ok=False
        xs.append(f"{(x if x is not None else float('nan')):>5.2f}" + (" " if good else "X"))
    print(f"y={ytarget}: " + " ".join(xs) + ("   <-- all reachable" if ok else ""))
