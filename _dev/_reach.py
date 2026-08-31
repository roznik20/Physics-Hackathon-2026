import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame
pygame.init()
import game.config as C
from game.bodies import Pendulum, HoopRig, choose_hoop_base
from game.motion import Motion
from physics.apparatus import run as run_system, SYSTEMS

def ball_max_x(W, H, g):
    pend = Pendulum(pivot_m=(0.14 * W, 0.08 * H), L=1.2, A=0.80, g=g, phi=0.0)
    best = -1e9
    dt = 1 / 60
    for ph in range(1500):
        pend.t = ph * dt
        p = list(pend.bob_pos()); vv = list(pend.bob_vel())
        for _ in range(1200):
            vv[1] += g * dt
            p[0] += vv[0] * dt; p[1] += vv[1] * dt
        best = max(best, p[0])
    return best

for (W, H) in [(1440, 900), (921, 691), (1600, 1000)]:
    print("\n=== window %dx%d  world %.2f x %.2f m ===" % (W, H, W / C.PX_PER_M, H / C.PX_PER_M))
    bx = ball_max_x(W, H, C.G)
    for i, s in enumerate(SYSTEMS, 1):
        m = Motion(run_system(s), target_amp=(0.55 + (i - 1) * 0.06), W=W, H=H,
                   launcher_frac=(0.14, 0.08), hoop_frac=(0.60, 0.66))
        hr = HoopRig(m, choose_hoop_base(m, W, H, (0.60, 0.66)))
        xs = [hr.root[0] + m.off[m.mr.driven][k, 0] for k in range(m.n)]
        hmin, hmax = min(xs), max(xs)
        gap = hmin - bx
        print("  L%2d %-12s hoop x[%5.2f,%5.2f]  ball max_x=%5.2f  %s"
              % (i, s.short, hmin, hmax, bx, ("OK" if gap < 0 else "SHORT %.2fm" % gap)))
