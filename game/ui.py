"""Presentation: HUD (in-game overlay) and simple panel helpers.

Kept separate from the engine so the simulation stays testable headless.
"""
from __future__ import annotations


import pygame

from .config import C


def _panel(screen, rect: pygame.Rect, alpha=235):
    s = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
    s.fill((*C["panel"], alpha))
    pygame.draw.rect(s, (*C["panel_edge"], 255), s.get_rect(), 2, border_radius=10)
    screen.blit(s, rect.topleft)
    pygame.draw.rect(screen, C["panel_edge"], rect, 2, border_radius=10)


def draw_hud(screen, game, W, H):
    font = game.font
    big = game.bigfont

    # top bar
    top = pygame.Rect(12, 10, W - 24, 46)
    _panel(screen, top, alpha=210)
    # system label
    sys_txt = font.render(f"System: {game.system_label}", True, C["ink"])
    screen.blit(sys_txt, (24, top.centery - sys_txt.get_height() // 2))
    # score + level
    right = big.render(f"Score {game.score}    Level {game.level}", True, C["ink"])
    screen.blit(right, (W - right.get_width() - 24, top.centery - right.get_height() // 2))

    # bottom-left: controls
    controls = font.render("SPACE release    R reset    P pause    M menu", True, C["ink_soft"])
    screen.blit(controls, (18, H - controls.get_height() - 14))

    # bottom-right: hoop motion amplitude
    amp = getattr(game.motion, "target_amp", 0.0)
    amp_txt = font.render(f"hoop motion {amp:.2f} m", True, C["ink_soft"])
    screen.blit(amp_txt, (W - amp_txt.get_width() - 18, H - amp_txt.get_height() - 14))

    # pause overlay
    if game.paused:
        veil = pygame.Surface((W, H), pygame.SRCALPHA)
        veil.fill((20, 20, 30, 130))
        screen.blit(veil, (0, 0))
        p = big.render("PAUSED", True, C["panel"])
        screen.blit(p, p.get_rect(center=(W // 2, H // 2 - 20)))
        c = font.render("press P to resume", True, C["panel"])
        screen.blit(c, c.get_rect(center=(W // 2, H // 2 + 16)))


# ---------------------------------------------------------------------------
# Button helper
# ---------------------------------------------------------------------------
class Button:
    def __init__(self, rect: pygame.Rect, label: str, font: pygame.font.Font,
                 fg=(255, 255, 255), bg=C["accent"], hover=None):
        self.rect = rect
        self.label = label
        self.font = font
        self.fg = fg
        self.bg = bg
        self.hover = hover or (bg[0] + 20, bg[1] + 20, bg[2] + 20)

    def draw(self, screen):
        hov = self.rect.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(screen, self.hover if hov else self.bg, self.rect, border_radius=8)
        t = self.font.render(self.label, True, self.fg)
        screen.blit(t, t.get_rect(center=self.rect.center))

    def clicked(self, pos) -> bool:
        return self.rect.collidepoint(pos)


def centered_button(W, H, y, label, font, w=220, h=54, **kw):
    return Button(pygame.Rect(W // 2 - w // 2, y, w, h), label, font, **kw)
