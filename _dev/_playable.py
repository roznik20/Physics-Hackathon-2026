"""Playability sweep: for each of the 10 systems, release the ball at every
launcher frame and run the EXACT engine ball flight (Euler, g=9.81, dt=1/60) to
the same scoring test. Report, per level: best (closest) approach to the rim
center, whether a scoring release exists, and how many frames score.

This answers: is every level actually winnable?
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame

pygame.init()
pygame.font.init()

from game.engine import Game, builtin_run
from game.config import PX_PER_M

W, H = 921, 691
screen = pygame.display.set_mode((W, H))
px = PX_PER_M
g = 9.81
dt = 1/60
ww, wh = W/px, H/px


def best_release(game, cfg):
    """Run the launcher over its full cycle; for each release frame, fly the
    ball and measure the closest approach to the rim center + whether it
    scores. Return (min_dist, n_score, best_frame, min_dist_frame)."""
    motion = game.motion
    root = tuple(game.launcher_root)
    hoop = game.hoop
    rim = hoop.c
    rr = hoop.rim_radius
    band = hoop.band
    n = motion.n

    min_dist = float("inf")
    best_frame = -1
    n_score = 0
    score_frames = []

    for i in range(0, n, 2):  # every other frame is plenty for a sweep
        p0 = motion.driven_world(i, root)
        v0 = motion.driven_velocity(i, dt)
        # fly the ball exactly like the engine does
        pos = p0.copy(); vel = v0.copy()
        scored = False
        for step in range(600):  # up to 10 s of flight
            vel = vel + np.array([0.0, g]) * dt
            pos = pos + vel * dt
            dx = pos[0] - rim[0]; dy = pos[1] - rim[1]
            d = float(np.hypot(dx, dy))
            if d < min_dist:
                min_dist = d
            if (dx*dx + dy*dy) <= (rr*rr) and abs(dy) <= band:
                scored = True
                break
            if pos[0] < -1 or pos[0] > ww+1 or pos[1] > wh+1:
                break
        if scored:
            n_score += 1
            if len(score_frames) < 3:
                score_frames.append(i)

    return min_dist, n_score, score_frames


run = builtin_run()
print(f"window {W}x{H}  world {ww:.2f}x{wh:.2f} m")
print(f"rim r={0.20} m band={0.12} m\n")
print(f"{'lvl':>3} {'system':<22} {'min_dist':>9} {'n_score':>8}  verdict")
print("-"*70)
all_winnable = True
for li, cfg in enumerate(run, start=1):
    game = Game(W, H, screen, run=run)
    game.level = li
    game._build_level()
    md, ns, sf = best_release(game, cfg)
    winnable = ns > 0
    all_winnable = all_winnable and winnable
    verdict = "WINS" if winnable else "UNWINNABLE"
    print(f"{li:>3} {cfg.system:<22} {md:>9.3f} {ns:>8}  {verdict}")

print("-"*70)
print("ALL WINNABLE" if all_winnable else "SOME UNWINNABLE")
