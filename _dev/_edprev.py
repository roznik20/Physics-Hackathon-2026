"""Render the MapEditor with its WYSIWYG live preview (a few frames so it animates)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame
from game.menu import MapEditor
from game.config import PX_PER_M
pygame.init()
pygame.font.init()
W, H = 900, 640
screen = pygame.display.set_mode((W, H))
ed = MapEditor(W, H, screen)
os.makedirs("_dev/renders3", exist_ok=True)
# advance a few frames so the apparatus is mid-swing
for i in range(40):
    ed.tick(1/60)
    ed.draw()
pygame.image.save(screen, "_dev/renders3/editor_live.png")
# also a mid-swing frame for a different system
ed.cfg.system = "double_pendulum"
ed._sys_index = 5
print("before tick: cfg.system=", ed.cfg.system, "preview=", ed._hoop_rig.motion.mr.label)
ed.tick(1/60); ed.tick(1/60)
print("after tick:  cfg.system=", ed.cfg.system, "preview=", ed._hoop_rig.motion.mr.label)
for i in range(20):
    ed.tick(1/60); ed.draw()
print("final preview=", ed._hoop_rig.motion.mr.label)
pygame.image.save(screen, "_dev/renders3/editor_live_dp.png")
print("editor live-preview renders saved")
