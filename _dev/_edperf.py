import os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame; pygame.init()
import game.config as C
from main import window_size
from game.menu import MapEditor
W,H = window_size()
print("real window:", W, H)
screen = pygame.display.set_mode((W,H))
# the editor created by main.py for a brand-new map (default config)
ed = MapEditor(W, H, screen, name="run_1")
print("default cfg:", ed.cfg.system, ed.cfg.launcher, ed.cfg.hoop)
# time a single build
t0=time.time(); ed._build_preview(); t1=time.time()
print("one _build_preview: %.3fs" % (t1-t0))
# now simulate a drag: 60 mousemove events each changing the hoop -> each triggers a rebuild
t0=time.time()
n_rebuilds=0
ox = ed.panel_w
for k in range(60):
    fx = 0.5 + 0.001*k; fy = 0.6
    ed.cfg.hoop = (fx, fy)
    ed.tick(1/60)  # this should detect cfg change and rebuild
    if ed._cfg_key()==ed._prev:  # rebuilt this tick
        n_rebuilds+=1
t1=time.time()
print("60 drag frames: %.3fs total, %.3fs/frame, rebuilds=%d" % (t1-t0,(t1-t0)/60,n_rebuilds))
print("=> at 60fps this is %.1x realtime" % (max((t1-t0)/60,0)/ (1/60)))
