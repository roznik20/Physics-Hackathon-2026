"""Menu, map gallery, and the map editor (presentation layer).

Each class owns its buttons and returns a transition string (e.g. ``"GAME"``,
``"MENU"``, ``None``) on the frame it should hand control back to ``main``.
"""
from __future__ import annotations

from typing import List, Optional

import pygame

from .config import C, MAP_DIR, PX_PER_M, MOTION_T_MAX
from .maps import MapLevel, list_maps, save_run, validate
from .ui import Button, _panel
from .engine import PEND_L, PEND_A, PEND_PHI
from .bodies import Ball, HoopRig, Pendulum, choose_hoop_base
from .motion import Motion
from .render import draw_scene
from physics.apparatus import SYSTEMS, run as run_system, system_by_id
from basketball_sprites.hoop_spawnv1 import HoopSprite


def _label_font(screen, size=18, bold=False):
    return pygame.font.SysFont("consolas", size, bold=bold)


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------
class Menu:
    def __init__(self, W, H, screen):
        self.W, self.H, self.screen = W, H, screen
        self.font = pygame.font.SysFont("consolas", 18)
        self.title_font = pygame.font.SysFont(None, 72)
        self.sub = pygame.font.SysFont("consolas", 17)
        self.btn_font = pygame.font.SysFont("consolas", 24, bold=True)

        # the original backdrop: bliss photo + LeBron, both scaled to fill
        self.bliss = pygame.image.load("assets/bliss.jpg").convert()
        self.lebron = pygame.image.load("assets/lebron.png").convert_alpha()
        self.bliss = self._cover(self.bliss, W, H)
        self.lebron = self._cover(self.lebron, W, H)
        self.title = self.title_font.render("LeHoop and Ball Game", True, (10, 10, 10))

        cx = W // 2
        top = H // 2 - 20
        self.play = Button(pygame.Rect(cx - 150, top, 300, 60), "Play", self.btn_font,
                          bg=(255, 153, 204), hover=(218, 112, 214))
        self.maps = Button(pygame.Rect(cx - 150, top + 72, 300, 60), "Map Gallery", self.btn_font,
                          bg=(255, 153, 204), hover=(218, 112, 214))
        self.about = Button(pygame.Rect(cx - 150, top + 144, 300, 60), "How to Play", self.btn_font,
                          bg=(255, 153, 204), hover=(218, 112, 214))
        self.quit = Button(pygame.Rect(cx - 150, top + 216, 300, 48), "Quit", self.btn_font,
                          bg=(200, 120, 170), hover=(170, 96, 150))

    @staticmethod
    def _cover(img, W, H):
        w, h = img.get_size()
        scale = max(W / w, H / h)
        return pygame.transform.scale(img, (int(w * scale), int(h * scale)))

    def draw(self):
        s = self.screen
        s.blit(self.bliss, (0, 0))
        s.blit(self.lebron, (0, 0))
        s.blit(self.title, self.title.get_rect(center=(self.W // 2, 70)))
        sub = self.sub.render("Time your release off the pendulum  ·  sink the moving goal", True, (20, 20, 20))
        s.blit(sub, sub.get_rect(center=(self.W // 2, 116)))
        # a soft panel behind the buttons for readability
        panel = pygame.Surface((360, 320), pygame.SRCALPHA)
        panel.fill((255, 255, 255, 150))
        s.blit(panel, panel.get_rect(center=(self.W // 2, self.H // 2 + 40)))
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
            "The ball hangs on a simple PENDULUM on the left. On the right, the",
            "HOOP rides a real Lagrangian/Newton system (pendulum, double",
            "pendulum, spring-mass, cart, ...) that swings it around as a",
            "moving target. Each level uses a different system, with bigger",
            "motion as you climb.",
            "",
            "  SPACE   release the ball off the pendulum at that instant",
            "  R       re-attach the ball (reset this shot)",
            "  P       pause        M  back to menu",
            "",
            "Time your release so the ball's arc meets the rim where it will",
            "be. Score to advance the run. Build your own in the Map Gallery.",
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

        # ---- WYSIWYG live preview (rendered exactly as the game will show it) ----
        self.preview = pygame.Surface((self.W - self.panel_w, self.H))
        self.preview_scale = 1.0
        self._prev = None          # last config hash the preview was built for
        self._prev_t = 0.0
        self._prev_rebuild = 0.0
        # use the same window-scaled px as the engine so the preview matches
        from .config import DESIGN_WORLD_W_M, DESIGN_WORLD_H_M
        self.px = max(PX_PER_M, W / DESIGN_WORLD_W_M, H / DESIGN_WORLD_H_M)
        self._hoop_sprite = HoopSprite(image_path="assets/hoopnobgd.png", tolerance=60,
                                       rim_anchor_px=(94, 183), crop_bottom_px=248)
        self._ball_img = pygame.image.load("assets/ball.png").convert_alpha()
        self._hoop_rig = None
        self._hoop = None
        self._pend = None
        self._ball = None
        self._build_preview()

    def _cfg_key(self):
        c = self.cfg
        return (c.system, round(c.launcher[0], 3), round(c.launcher[1], 3),
                round(c.hoop[0], 3), round(c.hoop[1], 3), round(c.amp_m, 3),
                round(c.gravity, 2), round(c.ball_radius_m, 3))

    def _build_preview(self):
        """Build the live preview objects from the current config, mirroring the
        engine's `_build_level` exactly (so the preview matches the game)."""
        cfg = self.cfg
        spec = system_by_id(cfg.system)
        mr = run_system(spec, t_max=MOTION_T_MAX, fps=60)
        motion = Motion(mr, cfg.amp_m, self.W, self.H,
                        launcher_frac=cfg.launcher, hoop_frac=cfg.hoop,
                        px=self.px)
        hoop_root = choose_hoop_base(motion, self.W, self.H, cfg.hoop, px=self.px)
        self._hoop_rig = HoopRig(motion, hoop_root)
        self._hoop = self._hoop_rig.hoop
        pivot = (cfg.launcher[0] * self.W / self.px, cfg.launcher[1] * self.H / self.px)
        self._pend = Pendulum(pivot, PEND_L, PEND_A, cfg.gravity, PEND_PHI)
        self._ball = Ball(radius_m=cfg.ball_radius_m)
        self._ball.attach_to(self._pend)
        self._prev = self._cfg_key()

    def tick(self, dt: float):
        """Advance the preview so the apparatus + pendulum animate in place."""
        self._prev_t += dt
        if self._pend and self._ball:
            self._pend.step(dt)
            self._ball.attach_to(self._pend)
        if self._hoop_rig:
            self._hoop_rig.step()
            self._hoop_rig.update_hoop()
        # The preview rebuild is a full scipy integration (~0.1 s), so rebuild
        # at most a few times a second — only when the config has changed since
        # the last rebuild. Without this, dragging a marker re-integrates every
        # frame and the editor freezes.
        now = self._prev_t
        if self._cfg_key() != self._prev and now - self._prev_rebuild > 0.18:
            self._build_preview()
            self._prev_rebuild = now

    def _system_index(self):
        ids = [s.id for s in SYSTEMS]
        return ids.index(self.cfg.system) if self.cfg.system in ids else 0

    def _preview_rect(self):
        """Where the scaled live preview sits in the editor (letterboxed)."""
        ox = self.panel_w
        sc = (self.W - ox) / max(1, self.W)
        pw = self.W - ox
        ph = int(self.H * sc)
        top = (self.H - ph) // 2
        return ox, top, pw, ph, sc

    def _frac_to_scene(self, fx, fy):
        """Window fraction -> editor scene pixel (matches where the game draws it)."""
        ox, top, pw, ph, sc = self._preview_rect()
        return (int(ox + fx * pw), int(top + fy * ph))

    def _scene_to_frac(self, px, py):
        """Editor scene pixel -> window fraction (inverse of _frac_to_scene)."""
        ox, top, pw, ph, sc = self._preview_rect()
        return (px - ox) / max(1, pw), (py - top) / max(1, ph)

    def _slider_value(self, rect, v, lo, hi):
        t = (v - lo) / (hi - lo)
        return int(rect.left + t * rect.w)

    def _render_preview(self):
        """Draw the live game scene at full window resolution, then scale it down
        into the preview area. Returns the blit origin (ox, top)."""
        ox, top, pw, ph, sc = self._preview_rect()
        full = pygame.Surface((self.W, self.H))
        d = max(6, int(self.cfg.ball_radius_m * self.px * 2))
        ball_img = pygame.transform.smoothscale(self._ball_img, (d, d))
        draw_scene(full, self._hoop_rig, self._hoop, self._pend, self._ball,
                   ball_img, self._hoop_sprite, self.px, self.W, self.H,
                   self._prev_t, 0.78)
        self.preview = pygame.transform.scale(full, (pw, ph))
        return ox, top

    def draw(self):
        s = self.screen
        s.fill(C["sky_top"])
        ox = self.panel_w

        # WYSIWYG live preview of the game scene (scaled into the right area)
        if self._hoop_rig is not None:
            ox, top = self._render_preview()
            s.blit(self.preview, (ox, top))
            pygame.draw.rect(s, C["court_line"], (ox, top, self.W - ox, self.H), 2)
        else:
            pygame.draw.rect(s, C["court"], (ox, 0, self.W - ox, self.H))
        hint = self.font.render("live preview — drag LAUNCHER / HOOP", True, (150, 132, 112))
        s.blit(hint, (ox + 10, 8))

        # placement markers on top of the live preview
        lp = self._frac_to_scene(*self.cfg.launcher)
        hp = self._frac_to_scene(*self.cfg.hoop)
        pygame.draw.circle(s, C["accent"], lp, 14, 3)
        lt = self.font.render("LAUNCHER", True, C["ink"])
        s.blit(lt, (lp[0] - lt.get_width() // 2, lp[1] - 34))
        pygame.draw.rect(s, C["ok"], pygame.Rect(hp[0] - 16, hp[1] - 16, 32, 32), 3)
        ht = self.font.render("HOOP", True, C["ink"])
        s.blit(ht, (hp[0] - ht.get_width() // 2, hp[1] - 34))

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
            # drag targets in the scene (window-frac, matching the game)
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
