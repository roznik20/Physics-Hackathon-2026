"""Fast playability for the new architecture.
Pendulum launcher is fixed (same every level). The hoop moves on each system's
driven node around a base center. A level is winnable iff there exists
(release_phase, hoop_phase) with the ball at the rim (within radius+band).
We test a grid of hoop base centers and report, per system, the best base and
the number of scoring (release, hoop-phase) pairs.
"""
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

# fixed pendulum launcher (same every level), hung from the upper-left
PIVOT = (0.20 * ww, 0.16 * wh)
L, A, PHI = 1.35, 0.9, 0.3

def ball_points(nrel=90, maxsteps=400, step=2):
    pts = []
    pend = Pendulum(PIVOT, L, A, g, PHI)
    for k in range(0, 720, 8):          # release phases
        pend.t = k * dt
        p = pend.bob_pos().copy(); v = pend.bob_vel().copy()
        for s in range(0, maxsteps, step):
            v = v + np.array([0, g]) * dt; p = p + v * dt
            if p[0] > ww + 0.3 or p[1] > wh + 0.3 or p[0] < -0.3:
                break
            pts.append(p)
    return np.array(pts)

BP = ball_points()
print(f"ball points: {len(BP)} ; x[{BP[:,0].min():.2f},{BP[:,0].max():.2f}] y[{BP[:,1].min():.2f},{BP[:,1].max():.2f}]")

def score_pairs(base, off, tol=0.22, band=0.12):
    """# of (ball-point, hoop-phase) with ball within rim radius+band of rim."""
    rims = np.array(base) + off            # (n,2)
    n = len(rims)
    hits = 0
    # vectorize per hoop phase
    for hp in range(0, n, 12):
        rim = rims[hp]
        d2 = ((BP[:, 0] - rim[0]) ** 2 + (BP[:, 1] - rim[1]) ** 2)
        dy = BP[:, 1] - rim[1]
        if np.any((d2 <= tol * tol) & (np.abs(dy) <= band)):
            hits += 1
    return hits

run = builtin_run()
game = Game(W, H, pygame.display.set_mode((W, H)), run=run)
print(f"\n{'system':<22} {'best_pairs':>10}  best_base_frac")
for li, cfg in enumerate(run, start=1):
    game.level = li; game._build_level()
    off = game.motion.off[game.motion.mr.driven]   # (n,2) world offsets
    best = (0, None)
    for fx in np.arange(0.52, 0.86, 0.05):
        for fy in np.arange(0.42, 0.78, 0.05):
            base = (fx * ww, fy * wh)
            s = score_pairs(base, off)
            if s > best[0]:
                best = (s, (fx, fy))
    winnable = best[0] > 0
    print(f"{cfg.system:<22} {best[0]:>10}  {best[1]}  {'WIN' if winnable else 'NO'}")
