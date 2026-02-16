from __future__ import annotations
from basketball_sprites.hoop_spawnv1 import HoopSprite 


"""
basket_assembly.py

PURPOSE (team contract):
- Physics engine outputs ONE canonical position each frame: hoop center (x, y) in screen coords.
- THIS module:
    1) receives that center position per frame (update_from_physics)
    2) clamps to keep the whole hoop assembly on-screen
    3) computes consistent positions/rects for rim/backboard/net using OFFSETS
- Other team members should treat `BasketAssembly.center` (and the provided rects/anchors)
  as the single source of truth for where to draw things and where to check collisions.

IMPORTANT RULE:
- Do NOT add extra motion/velocity here. Physics already decided the position.
- Only derive visuals/collision shapes from center + offsets.
"""

from dataclasses import dataclass, field
import math
import pygame


# =============================================================================
# EDIT ZONE (SAFE): BasketConfig
# -----------------------------------------------------------------------------
# Teammates SHOULD edit this section to tweak "looks":
# - dimensions, offsets, clamp padding
# They should NOT put per-frame motion logic here.
# =============================================================================
@dataclass
class BasketConfig:
    # ----------------------------
    # Rim geometry (collision + drawing)
    # ----------------------------
    rim_radius: int = 18
    rim_thickness: int = 4

    # ----------------------------
    # Backboard geometry (collision + drawing)
    # ----------------------------
    backboard_w: int = 90
    backboard_h: int = 55

    # ----------------------------
    # Net geometry (collision + drawing)
    # ----------------------------
    net_w: int = 40
    net_h: int = 45

    # ----------------------------
    # OFFSETS relative to hoop center (x right, y down).
    # These are the MAIN knobs for aligning sprites.
    # Example: backboard is usually slightly left/up from rim center.
    # ----------------------------
    backboard_offset: pygame.Vector2 = field(default_factory=lambda: pygame.Vector2(-55, -20))
    net_offset: pygame.Vector2 = field(default_factory=lambda: pygame.Vector2(0, 28))

    # Extra padding used by the clamp logic so sprites don't clip the screen edge.
    clamp_padding: int = 2


class BasketAssembly:
    """
    BasketAssembly = "single source of truth" object.

    What it owns:
    - center (hoop center): canonical position updated once per frame
    - derived collision shapes:
        - backboard_rect
        - net_rect
        - rim circle (center + radius)
    - helper methods for teammates:
        - get_anchors(): named points to attach sprites/particles
        - get_collision_shapes(): shapes for scoring/collisions

    What it DOES NOT own:
    - background rendering
    - score/miss animations
    - the physics engine
    """

    # =========================================================================
    # DO NOT TOUCH (unless you know what you're doing):
    # - update_from_physics
    # - clamping logic
    # - applying transforms from center to subparts
    # =========================================================================
    def __init__(self, center_xy=(0, 0), config: BasketConfig | None = None):
        self.cfg = config or BasketConfig()
        self.center = pygame.Vector2(center_xy)

        # Rects are useful for collisions AND for sprite placement (draw sprite at rect.topleft).
        self.backboard_rect = pygame.Rect(0, 0, self.cfg.backboard_w, self.cfg.backboard_h)
        self.net_rect = pygame.Rect(0, 0, self.cfg.net_w, self.cfg.net_h)

        # Precompute local bounds for clamping (relative to center).
        self._local_bounds = self._compute_local_bounds()

        # Initialize placement.
        self._apply_component_transforms()

    # -------------------------------------------------------------------------
    # SAFE: teammates can call this if they want to tweak offsets dynamically
    # (e.g., in a settings screen or while iterating on visuals).
    # -------------------------------------------------------------------------
    def set_offsets(
        self,
        *,
        backboard_offset: tuple[float, float] | None = None,
        net_offset: tuple[float, float] | None = None,
    ) -> None:
        if backboard_offset is not None:
            self.cfg.backboard_offset = pygame.Vector2(backboard_offset)
        if net_offset is not None:
            self.cfg.net_offset = pygame.Vector2(net_offset)

        # Changing offsets affects clamping bounds and placement.
        self._local_bounds = self._compute_local_bounds()
        self._apply_component_transforms()

    def update_from_physics(self, x: float, y: float, screen_size: tuple[int, int]) -> None:
        """
        CALL ONCE PER FRAME.

        Inputs:
            x, y: hoop center position from physics engine (screen coords)
            screen_size: (width, height) used to clamp

        Steps:
            1) set canonical center
            2) clamp so ENTIRE assembly stays on-screen
            3) update derived rects/anchors
        """
        self.center.update(x, y)
        self._clamp_center_to_screen(screen_size)
        self._apply_component_transforms()

    # -------------------------------------------------------------------------
    # TEAM API: use these instead of "guessing" positions in multiple places.
    # -------------------------------------------------------------------------
    def get_center(self) -> tuple[float, float]:
        return float(self.center.x), float(self.center.y)

    def get_anchors(self) -> dict[str, tuple[int, int]]:
        """
        Named attachment points. Good for:
        - drawing sprites
        - spawning particles on score
        - aligning net/backboard art

        You can add more anchors here as your art becomes more detailed.
        """
        return {
            "rim_center": (round(self.center.x), round(self.center.y)),
            "backboard_center": self.backboard_rect.center,
            "net_center": self.net_rect.center,
            # Useful reference: bottom of net (for "ball passed through" checks)
            "net_bottom": (self.net_rect.centerx, self.net_rect.bottom),
            # Useful reference: top of net
            "net_top": (self.net_rect.centerx, self.net_rect.top),
        }

    def get_collision_shapes(self) -> dict[str, object]:
        """
        Collision/scoring helpers.
        Teammates should use these instead of hardcoding geometry elsewhere.

        Rim is returned as (center_x, center_y, radius).
        """
        cx, cy = self.get_center()
        return {
            "rim_circle": (cx, cy, float(self.cfg.rim_radius)),
            "backboard_rect": self.backboard_rect.copy(),
            "net_rect": self.net_rect.copy(),
        }

    # -------------------------------------------------------------------------
    # INTERNAL: derives backboard/net placement from center + offsets.
    # -------------------------------------------------------------------------
    def _apply_component_transforms(self) -> None:
        bb_center = self.center + self.cfg.backboard_offset
        net_center = self.center + self.cfg.net_offset

        self.backboard_rect.size = (self.cfg.backboard_w, self.cfg.backboard_h)
        self.net_rect.size = (self.cfg.net_w, self.cfg.net_h)

        self.backboard_rect.center = (round(bb_center.x), round(bb_center.y))
        self.net_rect.center = (round(net_center.x), round(net_center.y))

    # -------------------------------------------------------------------------
    # INTERNAL: compute union bounding box (rim + backboard + net) in local coords.
    # Used for clamping.
    # -------------------------------------------------------------------------
    def _compute_local_bounds(self) -> pygame.Rect:
        r = self.cfg.rim_radius
        rim_rect = pygame.Rect(-r, -r, 2 * r, 2 * r)

        bb = pygame.Rect(0, 0, self.cfg.backboard_w, self.cfg.backboard_h)
        bb.center = (round(self.cfg.backboard_offset.x), round(self.cfg.backboard_offset.y))

        net = pygame.Rect(0, 0, self.cfg.net_w, self.cfg.net_h)
        net.center = (round(self.cfg.net_offset.x), round(self.cfg.net_offset.y))

        union = rim_rect.union(bb).union(net)
        pad = self.cfg.clamp_padding
        union.inflate_ip(pad * 2, pad * 2)
        return union

    # -------------------------------------------------------------------------
    # INTERNAL: clamp center so that (center + local_bounds) stays inside screen.
    # -------------------------------------------------------------------------
    def _clamp_center_to_screen(self, screen_size: tuple[int, int]) -> None:
        sw, sh = screen_size
        lb = self._local_bounds

        min_cx = -lb.left
        max_cx = sw - lb.right
        min_cy = -lb.top
        max_cy = sh - lb.bottom

        self.center.x = max(min_cx, min(self.center.x, max_cx))
        self.center.y = max(min_cy, min(self.center.y, max_cy))

    # =========================================================================
    # EDIT ZONE (SAFE): Rendering
    # -----------------------------------------------------------------------------
    # Teammates SHOULD replace draw_debug with real sprites/animation.
    # They should NOT change motion logic above.
    #
    # Suggested replacement:
    #   - draw backboard sprite at self.backboard_rect.topleft
    #   - draw net sprite at self.net_rect.topleft
    #   - draw rim sprite centered at self.center (or build a rim_rect)
    # =========================================================================
    def draw_debug(self, surf: pygame.Surface) -> None:
        pygame.draw.rect(surf, (230, 230, 230), self.backboard_rect, border_radius=6)
        pygame.draw.rect(surf, (40, 40, 40), self.backboard_rect, width=2, border_radius=6)

        pygame.draw.rect(surf, (220, 220, 255), self.net_rect, border_radius=6)
        pygame.draw.rect(surf, (40, 40, 40), self.net_rect, width=2, border_radius=6)

        pygame.draw.circle(surf, (255, 120, 0), (round(self.center.x), round(self.center.y)), self.cfg.rim_radius)
        pygame.draw.circle(
            surf, (40, 40, 40), (round(self.center.x), round(self.center.y)), self.cfg.rim_radius, width=2
        )


# =============================================================================
# TEST HELPERS (SAFE): fake physics motion + standalone demo
# -----------------------------------------------------------------------------
# This is only for sanity-checking your integration.
# It is NOT required for the final game.
# =============================================================================
def sanity_physics_position(t: float, screen_w: int, screen_h: int, margin: int = 80) -> tuple[float, float]:
    ax = (screen_w - 2 * margin) / 2
    ay = (screen_h - 2 * margin) / 2
    cx = screen_w / 2
    cy = screen_h / 2
    x = cx + ax * math.sin(2 * math.pi * 0.20 * t)
    y = cy + ay * math.sin(2 * math.pi * 0.13 * t + 1.1)
    return x, y

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((900, 550))
    clock = pygame.time.Clock()

    FPS = 60  # cap/target FPS

    # Choose where the oscillation origin (0,0) sits on the screen:
    osc_center = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)

    basket = BasketAssembly(center_xy=(osc_center.x, osc_center.y))

    hoop_sprite = HoopSprite(
        image_path="assets/hoopnobgd.png",
        tolerance=60,
        rim_anchor_px=None,  # start with center-of-image; calibrate if needed
    )

    t = 0.0
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # seconds since last frame
        t += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- physics engine output (RELATIVE to osc_center) ---
        # Replace this with your real engine's (dx, dy).
        # Right now, we fake it by converting your sanity function into relative motion:
        x_abs, y_abs = sanity_physics_position(t, screen.get_width(), screen.get_height(), margin=80)
        dx = x_abs - screen.get_width() / 2
        dy = y_abs - screen.get_height() / 2

        rim_x = osc_center.x + dx
        rim_y = osc_center.y + dy

        # Move the basket assembly so its rim center follows (rim_x, rim_y)
        basket.update_from_physics(rim_x, rim_y, screen.get_size())

        # --- draw ---
        screen.fill((18, 18, 24))
        basket.draw_debug(screen)

        rim_center = basket.get_anchors()["rim_center"]
        hoop_sprite.draw(screen, rim_center, debug=True)

        # Optional: actual FPS readout (not capped value)
        # print(clock.get_fps())

        pygame.display.flip()

    pygame.quit()