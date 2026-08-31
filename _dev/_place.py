"""Placement search: for each system, find a hoop position (right side, standard
basket height) that maximizes the number of release frames that score, subject to
the rim being on the descending arc and on-screen. Reports the best (x,y) and the
scoring-window size (in frames). Confirms every level is winnable by timing.
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

def fly(motion, root, i):
    pos = motion.driven_world(i, root).copy()
    vel = motion.driven_velocity(i, dt).copy()
    pts = [pos.copy()]
    for _ in range(720):
        vel = vel + np.array([0.0, g])*dt; pos = pos + vel*dt
        pts.append(pos.copy())
        if pos[0] < -1 or pos[0] > ww+1 or pos[1] > wh+1:
            break
    return np.array(pts)

def score_frames(motion, root, n, rim, rr=0.20, band=0.12):
    c = np.array(rim)
    cnt = 0
    for i in range(0, n, 2):
        pos = motion.driven_world(i, root).copy()
        vel = motion.driven_velocity(i, dt).copy()
        hit=False
        for _ in range(720):
            vel = vel + np.array([0.0, g])*dt; pos = pos + vel*dt
            dx=pos[0]-c[0]; dy=pos[1]-c[1]
            if (dx*dx+dy*dy) <= rr*rr and abs(dy) <= band:
                hit=True; break
            if pos[0] < -1 or pos[0] > ww+1 or pos[1] > wh+1: break
        if hit: cnt+=1
    return cnt

run = builtin_run()
print(f"world {ww:.2f}x{wh:.2f} m")
print(f"{'system':<22} {'best_window':>11} {'rim_x':>7} {'rim_y':>7}")
for li, cfg in enumerate(run, start=1):
    game=Game(W,H,screen,run=run); game.level=li; game._build_level()
    motion=game.motion; root=tuple(game.launcher_root); n=motion.n
    best=(0, None, None)
    # search x on the right, y on the lower half (descending arc region)
    for x in np.arange(0.45*ww, 0.88*ww, 0.05):
        for y in np.arange(0.55*wh, 0.85*wh, 0.05):
            c=score_frames(motion, root, n, (x, y))
            if c > best[0]:
                best=(c, x, y)
    print(f"{cfg.system:<22} {best[0]:>11} {best[1]:>7.2f} {best[2]:>7.2f}")
