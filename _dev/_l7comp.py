import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame
pygame.init()
import game.config as C
from game.engine import Game, builtin_run
import numpy as np

W, H = 1440, 900
screen = pygame.display.set_mode((W, H))
game = Game(W, H, screen, run=builtin_run())
game.level = 7
game._build_level()

m = game.motion
# find frame where vertical spring (v_wall -> mass) is shortest
def vlen(i):
    a = m.node("v_wall", i); b = m.node("mass", i)
    return np.hypot(a[0]-b[0], a[1]-b[1])
i = int(np.argmin([vlen(k) for k in range(m.n)]))
print("most-compressed frame:", i, " vlen=%.2f m" % (vlen(i)*C.PX_PER_M))
game.hoop_rig.idx = i
game.pend.t = 0.0
game.ball.attach_to(game.pend)
game.draw()
pygame.image.save(screen, "_dev/renders3/l7_compressed.png")
print("saved")
