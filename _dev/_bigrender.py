import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame; pygame.init()
from game.engine import Game, builtin_run
from game.ui import draw_hud
W,H = 1440,900
screen = pygame.display.set_mode((W,H))
run = builtin_run()
for li in [6,9]:
    game = Game(W,H,screen,run=run)
    game.level=li; game._build_level()
    for _ in range(40): game.update(1/60)
    game.draw(); draw_hud(screen,game,W,H)
    pygame.image.save(screen, f"_dev/renders3/big{li:02d}.png")
print("done")
