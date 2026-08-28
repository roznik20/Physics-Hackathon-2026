"""Measure the hoop sprite: below the rim (y>183), how wide is the non-transparent
content per row? The net is wide, the pole is a narrow bar; the wide->narrow
transition marks where to crop the pole off."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame
pygame.display.set_mode((10, 10))
from basketball_sprites.hoop_spawnv1 import build_hoop_surface

s = build_hoop_surface("assets/hoopnobgd.png", tolerance=60, crop_right_half=True)
w, h = s.get_width(), s.get_height()
print(f"cropped surface {w}x{h}")

# per-row: count non-transparent pixels and their x-span
rows = []
for y in range(h):
    xs = [x for x in range(w) if s.get_at((x, y)).a > 20]
    if xs:
        rows.append((y, len(xs), min(xs), max(xs)))

print("y    count  xmin  xmax")
for y, c, x0, x1 in rows:
    if y >= 170:  # from just above rim to bottom
        print(f"{y:4d}  {c:4d}  {x0:4d} {x1:4d}")
