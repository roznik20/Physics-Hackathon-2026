from __future__ import annotations
from pathlib import Path
import pygame


def color_close(c1, c2, tolerance=30) -> bool:
    return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2) ** 0.5 < tolerance


def _make_placeholder_hoop() -> tuple[pygame.Surface, tuple[int, int]]:
    """
    No PNG? No problem. Make a simple hoop drawing.
    Returns (surface, rim_center_px_inside_surface).
    """
    w, h = 240, 160
    surf = pygame.Surface((w, h), pygame.SRCALPHA)

    rim_center = (110, 80)

    # backboard
    pygame.draw.rect(surf, (230, 230, 230), (160, 20, 60, 60), border_radius=6)
    pygame.draw.rect(surf, (50, 50, 50), (160, 20, 60, 60), width=2, border_radius=6)

    # rim (ring)
    pygame.draw.circle(surf, (255, 120, 0), rim_center, 28, 6)

    # net-ish lines
    for i in range(7):
        x0 = rim_center[0] - 22 + i * 7
        pygame.draw.line(surf, (200, 200, 255), (x0, rim_center[1] + 24), (x0 + 4, 145), 2)

    return surf, rim_center


def build_hoop_surface_from_file(image_path: str, *, tolerance: int = 60) -> pygame.Surface:
    """
    Load + crop + recolor hoop from a PNG.
    Path is resolved relative to this file, not the terminal.
    """
    base_dir = Path(__file__).resolve().parent
    p = Path(image_path)
    if not p.is_absolute():
        p = base_dir / p

    hoop_image = pygame.image.load(str(p)).convert_alpha()

    width, height = hoop_image.get_size()
    crop_rect = pygame.Rect(width // 2, 0, width // 2, height)

    cropped = pygame.Surface((width // 2, height), pygame.SRCALPHA)
    cropped.blit(hoop_image, (0, 0), crop_rect)

    color_map = {
        (198, 27, 7): (255, 46, 214),
        (238, 63, 44): (255, 105, 226),
        (170, 187, 87): (125, 79, 116),
        (217, 226, 235): (255, 105, 226),
        (255, 126, 7): (255, 125, 243),
        (221, 94, 0): (201, 97, 192),
        (255, 255, 255): (255, 125, 243),
    }

    cropped.lock()
    for x in range(cropped.get_width()):
        for y in range(cropped.get_height()):
            pixel = cropped.get_at((x, y))
            rgb = pixel[:3]
            for old_rgb, new_rgb in color_map.items():
                if color_close(rgb, old_rgb, tolerance):
                    cropped.set_at((x, y), (*new_rgb, pixel[3]))
                    break
    cropped.unlock()

    return cropped


class HoopSprite:
    def __init__(
        self,
        *,
        image_path: str = "assets/hoopnobgd.png",
        tolerance: int = 60,
        rim_anchor_px: tuple[int, int] | None = None,
    ):
        # Try real art; fallback to placeholder
        try:
            self.surface = build_hoop_surface_from_file(image_path, tolerance=tolerance)
            suggested_anchor = (self.surface.get_width() // 2, self.surface.get_height() // 2)
        except FileNotFoundError:
            self.surface, suggested_anchor = _make_placeholder_hoop()

        if rim_anchor_px is None:
            rim_anchor_px = suggested_anchor

        self.rim_anchor = pygame.Vector2(rim_anchor_px)

    def draw(self, screen: pygame.Surface, rim_center_xy: tuple[float, float], *, debug: bool = False) -> None:
        rcx, rcy = rim_center_xy
        top_left = (round(rcx - self.rim_anchor.x), round(rcy - self.rim_anchor.y))
        screen.blit(self.surface, top_left)

        if debug:
            pygame.draw.circle(screen, (0, 255, 0), (round(rcx), round(rcy)), 4)