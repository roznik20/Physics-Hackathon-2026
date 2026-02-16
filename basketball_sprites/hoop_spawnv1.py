# hoop_spawnv1.py
# PURPOSE: load + recolor hoop PNG once, then draw it so a chosen pixel ("rim_anchor")
# aligns to the rim center (in SCREEN PIXELS) each frame.
from __future__ import annotations

from pathlib import Path
import pygame


def color_close(c1, c2, tolerance=30) -> bool:
    """Return True if RGB colors c1 and c2 are within tolerance."""
    return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2) ** 0.5 < tolerance


def build_hoop_surface(
    image_path: str = "assets/hoopnobgd.png",
    *,
    tolerance: int = 60,
    crop_right_half: bool = True,
) -> pygame.Surface:
    """
    Build the cropped + recolored hoop surface ONCE.

    IMPORTANT: image_path is resolved relative to THIS FILE, so it works no matter
    what your working directory is.
    """
    base_dir = Path(__file__).resolve().parent
    p = Path(image_path)
    if not p.is_absolute():
        p = base_dir / p

    hoop_image = pygame.image.load(str(p)).convert_alpha()
    width, height = hoop_image.get_size()

    # Crop (keep right half by default, matching the teammate art logic)
    if crop_right_half:
        crop_rect = pygame.Rect(width // 2, 0, width // 2, height)
        surf = pygame.Surface((width // 2, height), pygame.SRCALPHA)
        surf.blit(hoop_image, (0, 0), crop_rect)
    else:
        surf = hoop_image.copy()

    # Color replacement mapping
    color_map = {
        (198, 27, 7): (255, 46, 214),
        (238, 63, 44): (255, 105, 226),
        (170, 187, 87): (125, 79, 116),
        (217, 226, 235): (255, 105, 226),
        (255, 126, 7): (255, 125, 243),
        (221, 94, 0): (201, 97, 192),
        (255, 255, 255): (255, 125, 243),
    }

    surf.lock()
    for x in range(surf.get_width()):
        for y in range(surf.get_height()):
            pixel = surf.get_at((x, y))
            rgb = pixel[:3]
            for old_rgb, new_rgb in color_map.items():
                if color_close(rgb, old_rgb, tolerance):
                    surf.set_at((x, y), (*new_rgb, pixel[3]))
                    break
    surf.unlock()

    return surf


class HoopSprite:
    """
    Draw hoop art so that a chosen pixel INSIDE the image (rim_anchor_px)
    sits exactly on the rim center position (screen pixels) each frame.

    - If rim_anchor_px is wrong, motion will be correct but visually offset.
    - Use the click-calibration mode in game_main60.py to find the correct anchor.
    """

    def __init__(
        self,
        *,
        image_path: str = "assets/hoopnobgd.png",
        tolerance: int = 60,
        rim_anchor_px: tuple[int, int] | None = None,
    ):
        self.surface = build_hoop_surface(image_path, tolerance=tolerance)

        if rim_anchor_px is None:
            rim_anchor_px = (self.surface.get_width() // 2, self.surface.get_height() // 2)

        self.rim_anchor = pygame.Vector2(rim_anchor_px)

    def draw(self, screen: pygame.Surface, rim_center_xy: tuple[float, float], *, debug: bool = False) -> None:
        rcx, rcy = rim_center_xy
        top_left = (round(rcx - self.rim_anchor.x), round(rcy - self.rim_anchor.y))
        screen.blit(self.surface, top_left)

        if debug:
            pygame.draw.circle(screen, (0, 255, 0), (round(rcx), round(rcy)), 4)
