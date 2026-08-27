"""Integration: drive the REAL main.py state machine via synthetic events."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame
import main as M

pygame.init()
W, H = M.window_size()
screen = pygame.display.set_mode((W, H))
print("window:", W, H)

# Recreate main's state objects the same way main() does
menu = M.Menu(W, H, screen)
gallery = M.MapGallery(W, H, screen)
game = M.Game(W, H, screen, run=M.builtin_run())
from game.ui import draw_hud

def click(x, y):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (x, y)})

# 1) menu -> click Play (its center)
menu.draw()
r = menu.handle(click(*menu.play.rect.center))
print("menu Play ->", r)
assert r == "GAME"

# 2) game: step + draw a few frames, release ball, then M -> menu
for _ in range(30):
    game.update(1/60); game.draw(); draw_hud(screen, game, W, H)
game.release()
for _ in range(40):
    game.update(1/60); game.draw(); draw_hud(screen, game, W, H)
r = game.handle(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_m}))
print("game M ->", r, "| level", game.level, "score", game.score)
assert r == "MENU"

# 3) menu -> Map Gallery -> Back
menu.draw()
r = menu.handle(click(*menu.maps.rect.center))
print("menu Maps ->", r)
assert r == "MAPS"
gallery.draw()
r = gallery.handle(click(*gallery.back.rect.center))
print("gallery Back ->", r)
assert r == "MENU"

print("INTEGRATION OK")
