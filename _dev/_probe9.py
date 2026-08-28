"""Probe level-9 vertical stack: world + screen positions of each node."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame
pygame.init()
W, H = 921, 691
screen = pygame.display.set_mode((W, H))
from game.engine import Game, builtin_run
from game.config import PX_PER_M

run = builtin_run()
g = Game(W, H, screen, run=run)
g.level = 9
g._build_level()
m = g.motion
rr = g.hoop_rig
i = 0  # t=0
print(f"window {W}x{H}px, PX_PER_M={PX_PER_M}, world {W/PX_PER_M:.2f}x{H/PX_PER_M:.2f} m")
print(f"root {rr.root}")
for name in ["ceiling", "mass2", "mass1", "floor"]:
    off = rr.motion.off[name][i]
    world = rr.root + off
    px = (world[0]*PX_PER_M, world[1]*PX_PER_M)
    print(f"  {name:8s} world={world[0]:+.2f},{world[1]:+.2f}  screen=({px[0]:.0f},{px[1]:.0f})  "
          f"{'ON' if 0<=px[0]<W and 0<=px[1]<H else 'OFF'}-screen")
# driven amplitude
drv = rr.motion.off[rr.motion.mr.driven]
print(f"driven={rr.motion.mr.driven}  amp_m={m.target_amp}")
