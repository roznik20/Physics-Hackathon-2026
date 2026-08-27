"""Headless per-level render test for the NEW engine: build Game, draw each
level (ball mid-flight), save screenshots to _new/."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pathlib
import pygame
from game.engine import Game

pygame.init()
pygame.display.set_mode((1, 1))
W, H = 1280, 800
screen = pygame.display.set_mode((W, H))
game = Game(W, H, screen)
pathlib.Path("_new").mkdir(exist_ok=True)

for lvl in range(1, 11):
    game.level = lvl
    game._build_level()
    # let it run a few frames, then release and advance
    for _ in range(60):
        game.update(1 / 60)
    game.update(1 / 60)
    game.ball.release_from(game.launcher)
    for _ in range(22):
        game.update(1 / 60)
    game.draw()
    from game.ui import draw_hud
    draw_hud(screen, game, W, H)
    p = f"_new/level{lvl}_{game.system_short.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    pygame.image.save(screen, p)
    print("saved", p, "system:", game.system_short, "root:", [round(x,2) for x in game.launcher_root],
          "hoop:", [round(x,2) for x in game.hoop.c])

pygame.quit()
print("DONE")
