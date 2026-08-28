"""Headless render of the NEW architecture (hoop on physics system + pendulum launcher)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame
pygame.init()
W, H = 921, 691
screen = pygame.display.set_mode((W, H))
from game.engine import Game, builtin_run
from game.ui import draw_hud
import numpy as np

os.makedirs("_dev/renders2", exist_ok=True)
run = builtin_run()
for li, cfg in enumerate(run, start=1):
    game = Game(W, H, screen, run=run)
    game.level = li
    game._build_level()
    # let the hoop motion + pendulum advance a bit
    for _ in range(25):
        game.update(1 / 60)
    # release the ball at some swing instant and fly it
    game.release()
    for _ in range(30):
        game.update(1 / 60)
    game.draw()
    draw_hud(screen, game, W, H)
    safe = cfg.system[:14].replace(" ", "_")
    pygame.image.save(screen, f"_dev/renders2/level{li:02d}_{safe}.png")
print("saved 10 renders to _dev/renders2")
