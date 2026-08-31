"""Headless smoke test of the full app: menu, play+score, gallery, editor."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame
pygame.init()
W, H = 1024, 620
screen = pygame.display.set_mode((W, H))

from game.menu import Menu, MapGallery, About, MapEditor
from game.engine import Game, builtin_run
from game.ui import draw_hud
from game.maps import save_run, load_run, MapLevel
from game.config import MAP_DIR

os.makedirs("_smoke", exist_ok=True)

# --- menu
menu = Menu(W, H, screen)
menu.draw()
pygame.image.save(screen, "_smoke/menu.png")
print("menu ok")

# --- play built-in run, force a score by dropping ball straight into hoop
game = Game(W, H, screen, run=builtin_run())
for lv in range(1, 4):
    game.draw(); draw_hud(screen, game, W, H)
    pygame.image.save(screen, f"_smoke/play_l{lv}.png")
    # release and let it fly a bit
    game.ball.release_from(game.pend)
    for _ in range(120):
        game.update(1 / 60)
    print(f"  level {lv}: score={game.score} now-level={game.level} system={game.system_label!r}")
print("play ok")

# --- gallery (with one saved map)
save_run("smoketest", [MapLevel(name="a", system="double_pendulum", amp_m=0.8)], MAP_DIR)
gallery = MapGallery(W, H, screen)
gallery.draw()
pygame.image.save(screen, "_smoke/gallery.png")
print("gallery ok, maps:", gallery.maps)

# --- editor
ed = MapEditor(W, H, screen, name="edtest")
ed.cfg.system = "pendulum_cart"
ed.draw()
pygame.image.save(screen, "_smoke/editor.png")
print("editor ok")

# --- about
about = About(W, H, screen)
about.draw()
pygame.image.save(screen, "_smoke/about.png")
print("about ok")

# round-trip a map (clean up the fixture afterwards so re-runs don't dirty the tree)
lvl = load_run("smoketest", MAP_DIR)[0]
assert lvl.system == "double_pendulum", lvl.system
(MAP_DIR / "smoketest.json").unlink()
print("map round-trip ok:", lvl.name, lvl.system)
print("ALL SMOKE OK")
