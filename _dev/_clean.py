"""Clean render (ball attached, no flight/flash) to verify the new architecture."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame
pygame.init()
W, H = 921, 691
screen = pygame.display.set_mode((W, H))
from game.engine import Game, builtin_run
from game.ui import draw_hud

os.makedirs("_dev/renders2", exist_ok=True)
run = builtin_run()
for li in [1, 3, 6, 7, 9]:
    game = Game(W, H, screen, run=run)
    game.level = li
    game._build_level()
    for _ in range(40):
        game.update(1 / 60)      # advance hoop motion + pendulum, ball stays attached
    game.draw()
    draw_hud(screen, game, W, H)
    pygame.image.save(screen, f"_dev/renders2/clean{li:02d}.png")
print("done")
