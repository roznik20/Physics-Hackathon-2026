"""Confirm a SINGLE fixed hoop base center makes all 10 systems winnable, and pick
the one with the best (most forgiving) scoring window."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame
pygame.init()
from game.bodies import Pendulum
from game.engine import Game, builtin_run
from game.config import PX_PER_M

W, H = 921, 691
px = PX_PER_M; g = 9.81; dt = 1/60
ww, wh = W/px, H/px

PIVOT = (0.20 * ww, 0.16 * wh)
L, A, PHI = 1.35, 0.9, 0.3

def ball_points():
    pts = []; pend = Pendulum(PIVOT, L, A, g, PHI)
    for k in range(0, 720, 8):
        pend.t = k * dt
        p = pend.bob_pos().copy(); v = pend.bob_vel().copy()
        for _ in range(400):
            v = v + np.array([0, g]) * dt; p = p + v * dt
            if p[0] > ww+0.3 or p[1] > wh+0.3 or p[0] < -0.3: break
            pts.append(p)
    return np.array(pts)
BP = ball_points()

def score_pairs(base, off, tol=0.22, band=0.12):
    rims = np.array(base) + off; n = len(rims); hits = 0
    for hp in range(0, n, 12):
        rim = rims[hp]
        d2 = (BP[:,0]-rim[0])**2 + (BP[:,1]-rim[1])**2
        if np.any((d2 <= tol*tol) & (np.abs(BP[:,1]-rim[1]) <= band)):
            hits += 1
    return hits

run = builtin_run()
game = Game(W, H, pygame.display.set_mode((W, H)), run=run)

candidates = [(0.60,0.60),(0.62,0.62),(0.64,0.64),(0.66,0.60),(0.66,0.66),(0.68,0.58),(0.60,0.66)]
print("base_frac  ->  min over 10 systems (worst window) / all-win?")
for base in candidates:
    b = (base[0]*ww, base[1]*wh)
    mins = []; allok = True
    for li in range(1, 11):
        game.level = li; game._build_level()
        off = game.motion.off[game.motion.mr.driven]
        s = score_pairs(b, off)
        mins.append(s)
        if s == 0: allok = False
    print(f"  {base}   worst={min(mins)}  sum={sum(mins)}  all_win={allok}")
