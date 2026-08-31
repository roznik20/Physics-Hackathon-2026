import os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame
pygame.init()
import pygame.font
import game.config as C
from game.menu import MapEditor
from game.maps import MapLevel, save_run, load_run, list_maps, validate
from game.engine import Game, builtin_run

W,H = 1440,900
screen = pygame.display.set_mode((W,H))
font = pygame.font.SysFont("consolas", 18)
bigfont = pygame.font.SysFont("consolas", 28, bold=True)

def E(label, fn):
    try:
        fn()
        print("OK   ", label)
        return True
    except Exception:
        print("FAIL ", label)
        traceback.print_exc()
        return False

cfg = MapLevel(name="testmap", system="simple_pendulum", launcher=(0.20,0.10), hoop=(0.55,0.60), amp_m=0.7, gravity=9.81, ball_radius_m=0.12)
ed = MapEditor(W, H, screen, cfg, "testmap")

E("initial draw", ed.draw)
E("animate 60 ticks", lambda: [ed.tick(1/60) or ed.draw() for _ in range(60)])

# click each system in the list (the main "edit" action)
def click_sys(i):
    r, nm = ed.sys_rects[i]
    ed.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(r.x+5, r.y+5)))
    ed.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(r.x+5, r.y+5)))
    ed.draw()
for i in range(10):
    E(f"click system {i} ({ed.sys_rects[i][1]})", lambda i=i: click_sys(i))

# drag the hoop
def drag_hoop():
    ox, top, pw, ph, sc = ed._preview_rect()
    hx, hy = ed._frac_to_scene(*ed.cfg.hoop)
    ed.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(hx, hy)))
    for dx in range(-40, 60, 10):
        ed.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=(hx+dx, hy+dx//2)))
    ed.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(hx+50, hy)))
    ed.draw()
E("drag hoop", drag_hoop)

# drag the launcher
def drag_launcher():
    lx, ly = ed._frac_to_scene(*ed.cfg.launcher)
    ed.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(lx, ly)))
    for dx in range(-30, 40, 10):
        ed.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=(lx+dx, ly+dx//3)))
    ed.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(lx+30, ly)))
    ed.draw()
E("drag launcher", drag_launcher)

# move a slider
def move_slider():
    r = ed.slider_amp
    ed.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(r.x + int(0.5*r.w), r.centery)))
    ed.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=(r.x + int(0.8*r.w), r.centery)))
    ed.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(r.x + int(0.8*r.w), r.centery)))
    ed.draw()
E("move amp slider", move_slider)

# save
E("save run", lambda: ed.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=ed.save.rect.center)))
E("draw after save", ed.draw)

# test (returns to game with this cfg)
E("test click", lambda: ed.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=ed.test.rect.center)))

# now PLAY the saved map through the real Game (the "game breaks" step)
def play_saved():
    runs = list_maps(MAP_DIR:=C.MAP_DIR)
    print("  maps on disk:", runs)
    run = load_run("testmap", C.MAP_DIR)
    g = Game(W, H, screen, run=run)
    for _ in range(120):
        g.update(1/60); g.draw()
E("play saved map (120 frames)", play_saved)

print("\nDONE")
