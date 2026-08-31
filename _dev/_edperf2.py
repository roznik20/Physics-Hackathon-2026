import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame; pygame.init()
import game.config as C
from main import window_size
from game.menu import MapEditor
W,H = window_size()
screen = pygame.display.set_mode((W,H))
ed = MapEditor(W, H, screen, name="run_1")
# simulate a 3-second drag: 180 frames, hoop moving each frame
t0=time.time(); n=0
for k in range(180):
    ed.cfg.hoop = (0.5 + 0.001*k, 0.6 + 0.0005*k)
    ed.tick(1/60)
    if ed._cfg_key()==ed._prev: n+=1
    ed.draw()
t1=time.time()
print("3s drag: %.3fs for 180 frames = %.3fs/frame = %.1fx realtime; %d live-rebuilds" % (t1-t0,(t1-t0)/180,(t1-t0)/180/(1/60),n))
print("FINAL cfg matches preview?", ed._cfg_key()==ed._prev)
