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
m=game.motion; px=game.px; i=3264
def sc(name,ii):
    off=m.node(name,ii); return (game.hoop_rig.root[0]+off[0])*px, (game.hoop_rig.root[1]+off[1])*px
vwall=sc("v_wall",i); mass=sc("mass",i)
print("v_wall px",tuple(round(v) for v in vwall))
print("mass   px",tuple(round(v) for v in mass))
# crop bounds around vertical spring
x0=int(min(vwall[0],mass[0]))-30; x1=int(max(vwall[0],mass[0]))+30
y0=int(min(vwall[1],mass[1]))-30; y1=int(max(vwall[1],mass[1]))+30
print("crop region:",x0,y0,x1,y1)
img=pygame.image.load("_dev/renders3/l7_compressed.png")
from PIL import Image
im=Image.open("_dev/renders3/l7_compressed.png").convert("RGB")
im.crop((x0,y0,x1,y1)).save("_dev/renders3/l7_spring_zoom.png")
print("saved zoom")
