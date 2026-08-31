"""Deterministic scoring proof: force the ball through the rim and check score++ and level advance."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame
pygame.init()
W, H = 1024, 620
screen = pygame.display.set_mode((W, H))
from game.engine import Game, builtin_run
from game.maps import MapLevel

game = Game(W, H, screen, run=builtin_run())
hoop = game.hoop
# put the ball just left of the rim, moving right at rim height -> should score
game.ball.attached = False
game.ball.pos = pygame.math.Vector2(hoop.c[0] - 0.3, hoop.c[1])
game.ball.vel = pygame.math.Vector2(1.2, 0.0)
s0, lv0 = game.score, game.level
for _ in range(120):
    game.update(1 / 60)
    if game.score > s0:
        break
print("score:", s0, "->", game.score, " level:", lv0, "->", game.level)
assert game.score == s0 + 1, "did not score"
assert game.level == lv0 + 1, "level did not advance"
print("SCORING OK (advanced to", game.system_label, ")")
