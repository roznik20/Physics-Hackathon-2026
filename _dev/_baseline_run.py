"""Baseline: run the CURRENT game headless, force through levels, save screenshots."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame, time, pathlib
import main as M

pygame.init()
W, H = 1280, 800
screen = pygame.display.set_mode((W, H))
game = M.Game(W, H)
game.clock = pygame.time.Clock()
pathlib.Path("_baseline").mkdir(exist_ok=True)

def shot(tag):
    game.draw(screen)
    p = f"_baseline/{tag}.png"
    pygame.image.save(screen, p)
    print("saved", p, "level", game.level, "motion", game.motion_name)

# level 1
for _ in range(40):
    game.update(1/60)
shot("level1")

for lvl in range(2, 12):
    game.level = lvl
    game._apply_level_motion(lvl, reset_ball=True)
    # release ball so it flies, advance a bit to show trajectory + apparatus
    game.pend.t = 0.3
    game.ball.release_from(game.pend)
    for _ in range(18):
        game.update(1/60)
    shot(f"level{lvl}")

pygame.quit()
print("DONE")
