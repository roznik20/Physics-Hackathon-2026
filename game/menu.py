"""Menu, map gallery, and the map editor (presentation layer).

Each class owns its buttons and returns a transition string (e.g. ``"GAME"``,
``"MENU"``, ``None``) on the frame it should hand control back to ``main``.
"""
from __future__ import annotations

from typing import List, Optional

import pygame

from .config import C, MAP_DIR, PX_PER_M
from .maps import MapLevel, list_maps, load_run, save_run, validate
from .ui import Button, _panel
from .engine import builtin_run
from physics.apparatus import SYSTEMS


def _label_font(screen, size=18, bold=False):
    return pygame.font.SysFont("consolas", size, bold=bold)


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------
class Menu:
    def __init__(self, W, H, screen):
        self.W, self.H, self.screen = W, H, screen
        self.font = pygame.font.SysFont("consolas", 18)
        self.title = pygame.font.SysFont("consolas", 46, bold=True)
        self.sub = pygame.font.SysFont("consolas", 15)
        self.btn_font = pygame.font.SysFont("consolas", 22, bold=True)

        cx = W // 2
        self.play = Button(pygame.Rect(cx - 130, H // 2 - 10, 260, 58), "Play", self.btn_font)
        self.maps = Button(pygame.Rect(cx - 130, H // 2 + 58, 260, 58), "Map Gallery", self.btn_font)
        self.about = Button(pygame.Rect(cx - 130, H // 2 + 126, 260, 58), "How to Play", self.btn_font)
        self.quit = Button(pygame.Rect(cx - 130, H // 2 + 194, 260, 46), "Quit", self.btn_font,
                          bg=(96, 100, 112), hover=(116, 120, 132))

    def draw(self):
        s = self.screen
        # gradient backdrop
        s.fill(C["sky_top"])
        for y in range(self.H):
            t = y / max(1, self.H - 1)
            col = tuple(int(C["sky_top"][i] + (C["sky_bot"][i] - C["sky_top"][i]) * t) for i in range(3))
            pygame.draw.line(s, col, (0, y), (self.W, y))
        # title
        t = self.title.render("Physics Hoop", True, C["ink"])
        s.blit(t, t.get_rect(center=(self.W // 2, self.H // 2 - 120)))
        sub = self.sub.render("Lagrangian-powered launcher  ·  hit the moving goal", True, C["ink_soft"])
        s.blit(sub, sub.get_rect(center=(self.W // 2, self.H // 2 - 82)))

        for b in (self.play, self.maps, self.about, self.quit):
            b.draw(s)

    def handle(self, event) -> Optional[str]:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.play.clicked(event.pos):
                return "GAME"
            if self.maps.clicked(event.pos):
                return "MAPS"
            if self.about.clicked(event.pos):
                return "ABOUT"
            if self.quit.clicked(event.pos):
                return "QUIT"
        return None


# ---------------------------------------------------------------------------
# About / How to play
# ---------------------------------------------------------------------------
class About:
    def __init__(self, W, H, screen):
        self.W, self.H, self.screen = W, H, screen
        self.font = pygame.font.SysFont("consolas", 17)
        self.big = pygame.font.SysFont("consolas", 30, bold=True)
        self.back = Button(pygame.Rect(W // 2 - 100, H - 78, 200, 50), "Back",
                           pygame.font.SysFont("consolas", 20, bold=True))

    def draw(self):
        s = self.screen
        s.fill(C["sky_top"])
        title = self.big.render("How to Play", True, C["ink"])
        s.blit(title, (30, 26))
        lines = [
            "The ball hangs on the free end of a real Lagrangian system on the",
            "left (pendulum, double pendulum, spring-mass, cart, ...). Each level",
            "uses a different system. The hoop on the right is a fixed, mounted",
            "goal. Time your release so the ball's velocity sends it through the",
            "",
            "  SPACE   release the ball at the current instant",
            "  R       re-attach the ball (reset this shot)",
            "  P       pause        M  back to menu",
            "",
            "Score by passing the ball through the rim. Every point advances the",
            "run to the next level. Build your own runs in the Map Gallery.",
        ]
        y = 92
        for ln in lines:
            if ln:
                s.blit(self.font.render(ln, True, C["ink_soft"]), (30, y))
            y += 26
        self.back.draw(s)

    def handle(self, event) -> Optional[str]:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.back.clicked(event.pos):
            return "MENU"
        return None


# ---------------------------------------------------------------------------
# Map gallery
# ---------------------------------------------------------------------------
class MapGallery:
    def __init__(self, W, H, screen):
        self.W, self.H, self.screen = W, H, screen
        self.font = pygame.font.SysFont("consolas", 18)
        self.big = pygame.font.SysFont("consolas", 26, bold=True)
        self.btn = pygame.font.SysFont("consolas", 18, bold=True)
        self.refresh()
        self.new = Button(pygame.Rect(20, self.H - 64, 180, 44), "+ New map", self.btn)
        self.back = Button(pygame.Rect(self.W - 200, self.H - 64, 180, 44), "Back", self.btn,
                          bg=(96, 100, 112), hover=(116, 120, 132))

    def refresh(self):
        self.maps = list_maps(MAP_DIR)
        # playable buttons
        self.buttons: List[Button] = []
        x0, y0 = 20, 70
        colw, rowh = (self.W - 40) // 2, 92
        for i, name in enumerate(self.maps):
            r = pygame.Rect(x0 + (i % 2) * (colw + 16), y0 + (i // 2) * (rowh + 12), colw, rowh)
            self.buttons.append(Button(r, name, self.btn, bg=(64, 120, 232), hover=(84, 140, 252)))

    def draw(self):
        s = self.screen
        s.fill(C["sky_top"])
        s.blit(self.big.render("Map Gallery", True, C["ink"]), (20, 24))
        hint = self.font.render("Click a run to play it. Build a new one with '+ New map'.", True, C["ink_soft"])
        s.blit(hint, (22, 58))
        if not self.maps:
            empty = self.font.render("(no saved runs yet)", True, C["ink_soft"])
            s.blit(empty, (26, 120))
        for b in self.buttons:
            b.draw(s)
        self.new.draw(s)
        self.back.draw(s)

    def handle(self, event) -> Optional[str]:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, b in enumerate(self.buttons):
                if b.clicked(event.pos):
                    return ("PLAYMAP", self.maps[i])
            if self.new.clicked(event.pos):
                return "NEWMAP"
            if self.back.clicked(event.pos):
                return "MENU"
        return None


# ---------------------------------------------------------------------------
# Map editor: place launcher + hoop, pick system / amplitude / gravity / ball size
# ---------------------------------------------------------------------------
class MapEditor:
    def __init__(self, W, H, screen, initial: Optional[MapLevel] = None,
                 name: str = "new_map"):
        self.W, self.H, self.screen = W, H, screen
        self.font = pygame.font.SysFont("consolas", 16)
        self.big = pygame.font.SysFont("consolas", 24, bold=True)
        self.name = name
        self.cfg = initial or MapLevel()
        # editing: 'none' | 'launcher' | 'hoop'
        self.editing = "launcher"
        self.hover = None
        self.saved = False
        # left control panel
        self.panel_w = 268
        self.sys_names = [s.label for s in SYSTEMS]
        self.sys_rects = []
        y = 74
        for nm in self.sys_names:
            r = pygame.Rect(20, y, self.panel_w - 40, 24)
            self.sys_rects.append((r, nm))
            y += 26
        # sliders
        self.slider_amp = pygame.Rect(20, y + 14, self.panel_w - 40, 20)
        self.slider_grav = pygame.Rect(20, y + 52, self.panel_w - 40, 20)
        self.slider_ball = pygame.Rect(20, y + 90, self.panel_w - 40, 20)
        self.sliders = {"amp": self.slider_amp, "grav": self.slider_grav, "ball": self.slider_ball}
        self._sys_index = self._system_index()

        self.save = Button(pygame.Rect(20, self.H - 60, 120, 40), "Save", self.font, bg=(52, 176, 110), hover=(72, 196, 130))
        self.test = Button(pygame.Rect(150, self.H - 60, 90, 40), "Test", self.font, bg=(64, 120, 232), hover=(84, 140, 252))
        self.back = Button(pygame.Rect(self.W - 120, self.H - 60, 100, 40), "Back", self.font,
                          bg=(96, 100, 112), hover=(116, 120, 132))

    def _system_index(self):
        ids = [s.id for s in SYSTEMS]
        return ids.index(self.cfg.system) if self.cfg.system in ids else 0

    def _frac_to_scene(self, fx, fy):
        ox = self.panel_w
        return (int(ox + fx * (self.W - ox)), int(fy * self.H))

    def _scene_to_frac(self, px, py):
        ox = self.panel_w
        return (px - ox) / max(1, (self.W - ox)), py / max(1, self.H)

    def _slider_value(self, rect, v, lo, hi):
        t = (v - lo) / (hi - lo)
        return int(rect.left + t * rect.w)

    def draw(self):
        s = self.screen
        s.fill(C["sky_top"])
        # scene area
        ox = self.panel_w
        pygame.draw.rect(s, C["court"], (ox, 0, self.W - ox, self.H))
        # faint grid so positions are easy to read
        for gx in range(ox, self.W, 48):
            pygame.draw.line(s, (233, 214, 192), (gx, 0), (gx, self.H), 1)
        for gy in range(0, self.H, 48):
            pygame.draw.line(s, (233, 214, 192), (ox, gy), (self.W, gy), 1)
        pygame.draw.line(s, C["court_line"], (ox, 0), (ox, self.H), 2)
        hint = self.font.render("drag LAUNCHER / HOOP to place them", True, (150, 132, 112))
        s.blit(hint, (ox + 10, 8))

        # launcher + hoop markers
        lp = self._frac_to_scene(*self.cfg.launcher)
        hp = self._frac_to_scene(*self.cfg.hoop)
        pygame.draw.circle(s, C["accent"], lp, 16, 3)
        pygame.draw.circle(s, C["accent"], lp, 5)
        pygame.draw.rect(s, C["ok"], pygame.Rect(hp[0] - 20, hp[1] - 20, 40, 40), 3)
        pygame.draw.line(s, C["ok"], (hp[0], hp[1]), (hp[0], hp[1] + 60), 4)
        lt = self.font.render("LAUNCHER", True, C["ink"])
        s.blit(lt, (lp[0] - lt.get_width() // 2, lp[1] - 40))
        ht = self.font.render("HOOP", True, C["ink"])
        s.blit(ht, (hp[0] - ht.get_width() // 2, hp[1] - 44))

        # control panel
        _panel(s, pygame.Rect(0, 0, self.panel_w, self.H), alpha=245)
        s.blit(self.big.render("Edit map", True, C["ink"]), (16, 18))
        s.blit(self.font.render("System:", True, C["ink_soft"]), (20, 52))
        for i, (r, nm) in enumerate(self.sys_rects):
            on = (i == self._sys_index)
            if on:
                pygame.draw.rect(s, C["accent"], r, border_radius=4)
            s.blit(self.font.render(nm, True, C["panel"] if on else C["ink_soft"]), (r.x + 8, r.y + 3))

        # sliders
        self._slider(self.slider_amp, "Amplitude", self.cfg.amp_m, 0.2, 2.2, self._slider_value(self.slider_amp, self.cfg.amp_m, 0.2, 2.2))
        self._slider(self.slider_grav, "Gravity", self.cfg.gravity, 1.0, 20.0, self._slider_value(self.slider_grav, self.cfg.gravity, 1.0, 20.0))
        self._slider(self.slider_ball, "Ball size", self.cfg.ball_radius_m, 0.05, 0.30, self._slider_value(self.slider_ball, self.cfg.ball_radius_m, 0.05, 0.30))

        self.save.draw(s)
        self.test.draw(s)
        self.back.draw(s)

        # validation
        probs = validate(self.cfg)
        if probs:
            s.blit(self.font.render("! " + probs[0], True, C["warn"]), (20, self.H - 84))

    def _slider(self, rect, label, value, lo, hi, knob_x):
        s = self.screen
        s.blit(self.font.render(f"{label}: {value:.2f}", True, C["ink"]), (20, rect.y - 18))
        pygame.draw.rect(s, C["panel_edge"], rect, border_radius=10)
        pygame.draw.rect(s, C["accent2"], pygame.Rect(rect.x, rect.centery - 3, knob_x - rect.x, 6), border_radius=3)
        pygame.draw.circle(s, C["accent"], (knob_x, rect.centery), 10)

    def handle(self, event) -> Optional[str]:
        ox = self.panel_w
        in_scene = event.pos[0] > ox
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # system list
            for i, (r, nm) in enumerate(self.sys_rects):
                if r.collidepoint(event.pos):
                    self._sys_index = i
                    self.cfg.system = SYSTEMS[i].id
                    return None
            for key, rect in self.sliders.items():
                if rect.collidepoint(event.pos):
                    self._drag_slider = key
                    self._set_slider(key, event.pos[0])
                    return None
            if self.save.clicked(event.pos):
                save_run(self.name, [self.cfg], MAP_DIR)
                return "SAVED"
            if self.test.clicked(event.pos):
                return ("TEST", [self.cfg])
            if self.back.clicked(event.pos):
                return "MENU"
            # drag targets in scene
            if in_scene:
                lp = self._frac_to_scene(*self.cfg.launcher)
                hp = self._frac_to_scene(*self.cfg.hoop)
                if abs(event.pos[0] - lp[0]) < 24 and abs(event.pos[1] - lp[1]) < 24:
                    self.editing = "launcher"
                elif abs(event.pos[0] - hp[0]) < 28 and abs(event.pos[1] - hp[1]) < 28:
                    self.editing = "hoop"

        elif event.type == pygame.MOUSEMOTION:
            if getattr(self, "_drag_slider", None):
                self._set_slider(self._drag_slider, event.pos[0])
            elif self.editing == "launcher" and in_scene and pygame.mouse.get_pressed()[0]:
                fx, fy = self._scene_to_frac(event.pos)
                self.cfg.launcher = (max(0.02, min(0.98, fx)), max(0.06, min(0.94, fy)))
            elif self.editing == "hoop" and in_scene and pygame.mouse.get_pressed()[0]:
                fx, fy = self._scene_to_frac(event.pos)
                self.cfg.hoop = (max(0.02, min(0.98, fx)), max(0.06, min(0.94, fy)))

        elif event.type == pygame.MOUSEBUTTONUP:
            if getattr(self, "_drag_slider", None):
                self._drag_slider = None
        return None

    def _set_slider(self, key, x):
        rect = self.sliders[key]
        t = max(0.0, min(1.0, (x - rect.x) / rect.w))
        if key == "amp":
            self.cfg.amp_m = 0.2 + t * (2.2 - 0.2)
        elif key == "grav":
            self.cfg.gravity = 1.0 + t * (20.0 - 1.0)
        elif key == "ball":
            self.cfg.ball_radius_m = 0.05 + t * (0.30 - 0.05)
