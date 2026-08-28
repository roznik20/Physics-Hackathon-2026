"""Full verification after re-architecture: menu look, integration transitions,
and a clean per-level render."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame
pygame.init()
import main as M
from game.menu import Menu, MapGallery
from game.ui import draw_hud

W, H = M.window_size()
screen = pygame.display.set_mode((W, H))
os.makedirs("_dev/renders3", exist_ok=True)

# menu look
menu = M.Menu(W, H, screen)
menu.draw()
pygame.image.save(screen, "_dev/renders3/menu_lehoop.png")

# integration transitions
def click(x, y):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (x, y)})
r = menu.handle(click(*menu.play.rect.center)); assert r == "GAME", r
game = M.Game(W, H, screen, run=M.builtin_run())
for _ in range(40):
    game.update(1/60); game.draw(); draw_hud(screen, game, W, H)
r = game.handle(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_m})); assert r == "MENU", r
r = menu.handle(click(*menu.maps.rect.center)); assert r == "MAPS", r
gallery = M.MapGallery(W, H, screen); gallery.draw()
r = gallery.handle(click(*gallery.back.rect.center)); assert r == "MENU", r
print("menu look + integration OK")

# clean per-level render (ball attached, no flash)
from game.engine import builtin_run
run = builtin_run()
for li, cfg in enumerate(run, start=1):
    g = M.Game(W, H, screen, run=run); g.level = li; g._build_level()
    for _ in range(45):
        g.update(1/60)
    g.draw(); draw_hud(screen, g, W, H)
    pygame.image.save(screen, f"_dev/renders3/l{li:02d}.png")
print("saved 10 clean renders to _dev/renders3")
