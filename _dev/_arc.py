"""Playability for the NEW architecture: the pendulum launcher is FIXED (same every
level), the hoop MOVES on the physics system. Find where the ball's arc goes, and
where to place the hoop base center so the hoop's motion envelope overlaps the arc.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame
pygame.init()
from game.bodies import Pendulum, Ball
from game.config import PX_PER_M
from game.engine import PEND_PIVOT_FRAC, PEND_L, PEND_A, PEND_PHI

W, H = 921, 691
px = PX_PER_M; g = 9.81; dt = 1/60
ww, wh = W/px, H/px
pivot = (PEND_PIVOT_FRAC[0]*W/px, PEND_PIVOT_FRAC[1]*H/px)
print(f"world {ww:.2f}x{wh:.2f} m ; pivot {pivot} ; L={PEND_L} A={PEND_A}")

pend = Pendulum(pivot, PEND_L, PEND_A, g, PEND_PHI)
# ball trajectory envelope over all release phases
xs=[]; ys=[]
trajs=[]
for k in range(720):
    pend.t = k*dt
    pos = pend.bob_pos(); vel = pend.bob_vel()
    trajs.append((pos.copy(), vel.copy()))
    p=pos.copy(); v=vel.copy()
    for _ in range(480):
        v=v+np.array([0,g])*dt; p=p+v*dt
        if p[0]>ww+1 or p[1]>wh+1 or p[0]<-1: break
        xs.append(p[0]); ys.append(p[1])

xs=np.array(xs); ys=np.array(ys)
print(f"ball arc reach: x in [{xs.min():.2f},{xs.max():.2f}]  y in [{ys.min():.2f},{ys.max():.2f}]")
# where is the arc at x = 0.65*ww (candidate hoop x)?
target_x = 0.65*ww
near = ys[(xs>target_x-0.05)&(xs<target_x+0.05)]
print(f"arc y at x={target_x:.2f}: {near.min() if len(near) else 'none':.2f}-{near.max() if len(near) else 'none':.2f}" if len(near) else f"arc does not reach x={target_x:.2f}")
# the lowest (most dropped) point on the arc at x=target_x is the natural basket
if len(near):
    print(f"suggested hoop base center ~ ({target_x:.2f}, {near.mean():.2f})  i.e. frac ({target_x/ww:.2f}, {near.mean()/wh:.2f})")
