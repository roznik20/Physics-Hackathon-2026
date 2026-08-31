import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame; pygame.init()
import game.config as C
from game.engine import Game, builtin_run
import numpy as np
W,H=1440,900
screen=pygame.display.set_mode((W,H))
game=Game(W,H,screen,run=builtin_run())
game.level=7; game._build_level()
m=game.motion; px=game.px
def plen(i,a,b):
    p=m.node(a,i); q=m.node(b,i)
    return float(np.hypot(p[0]-q[0],p[1]-q[1])*px)
for name,a,b in [("vert","v_wall","mass"),("horiz","h_wall","mass")]:
    lens=[plen(i,a,b) for i in range(m.n)]
    i_min=int(np.argmin(lens)); i_max=int(np.argmax(lens))
    # coil params my _spring uses at min length
    d=lens[i_min]
    coils=max(3,int(d/14)); amp=8*max(0.35,min(1.0,d/100.0))
    print(f"{name}: rest={plen(0,a,b):.0f}px min={lens[i_min]:.0f}px (f={i_min}) max={lens[i_max]:.0f}px (f={i_max})  -> at min: coils={coils} amp={amp:.1f}px  spacing={d*0.8/(coils+1):.1f}px")
