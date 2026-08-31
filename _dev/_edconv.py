import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame; pygame.init()
from main import window_size
from game.menu import MapEditor
W,H = window_size()
screen = pygame.display.set_mode((W,H))
ed = MapEditor(W, H, screen, name="run_1")
# drag then stop
for k in range(40):
    ed.cfg.hoop = (0.5 + 0.001*k, 0.6)
    ed.tick(1/60)
# now idle for a few frames — preview should catch up to the final cfg
caught = False
for k in range(120):
    ed.tick(1/60)
    if ed._cfg_key()==ed._prev:
        caught=True; break
print("preview converged to final cfg after stop:", caught, "(after", k, "idle frames)")
