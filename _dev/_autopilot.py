"""End-to-end autoplayer: drive the REAL Game through all 10 levels. Per level,
try releasing at successive frames; when the ball scores, the engine advances to
the next level. Assert we reach level 11 (all 10 cleared). Also record how many
attempts (release timing) each level took.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame
pygame.init()
W, H = 1440, 900   # LARGE window — the size where levels used to be unwinnable
screen = pygame.display.set_mode((W, H))
from game.engine import Game, builtin_run

run = builtin_run()
game = Game(W, H, screen, run=run)
dt = 1/60

start_level = 1
for target in range(1, 11):
    # make sure we're on the target level
    attempts = 0
    cleared = False
    for rel_phase in range(0, 720, 2):
        attempts += 1
        # reset to a fresh shot on this level
        if game.level != target:
            break  # already advanced past
        game.level = target
        game._build_level()
        # advance the launcher/hoop to a given phase, then release
        for _ in range(rel_phase):
            game.pend.step(dt)
            game.hoop_rig.step()
            game.hoop_rig.update_hoop()
        game.release()
        for _ in range(720):   # let the ball fly
            game.update(dt)
            if game.level > target:
                cleared = True
                break
        if cleared:
            break
    print(f"level {target:2d} {run[target-1].system:<22} cleared={cleared}  (tried {attempts} releases)")
    if not cleared:
        print("  !! did not clear within one full release cycle")

print("\nFINAL: level =", game.level, " score =", game.score)
print("ALL 10 CLEARED" if game.level >= 11 else "NOT ALL CLEARED")
