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
# sample 6 frames across the motion
for k,frac in enumerate([0.0,0.25,0.5,0.75,0.9,1.0]):
    i=int(frac*(m.n-1)); i=max(1,i)
    game.hoop_rig.idx=i; game.pend.t=0.0; game.ball.attach_to(game.pend)
    game.draw()
    pygame.image.save(screen,f"_dev/renders3/l7_anim_{k}.png")
print("saved 6 frames")
