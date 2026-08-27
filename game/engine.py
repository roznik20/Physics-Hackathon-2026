"""The Game engine: state machine for a *run* of levels, scoring, input, and the
per-frame update/draw. UI chrome (HUD, menu, editor) lives in the presentation
layer; this class owns the simulation and exposes clean state for it.

A *level* is a :class:`game.maps.MapLevel` (system + positions + amplitude +
gravity + ball radius). The engine plays an ordered list of such levels — the
built-in ladder is just one such run.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pygame

from physics.apparatus import SYSTEMS, run as run_system, system_by_id
from physics.common import MotionResult

from .bodies import Ball, Hoop, Launcher
from .config import (G, HOOP_FRAC, LAUNCHER_FRAC, MOTION_T_MAX, PX_PER_M, C,
                     BALL_RADIUS_M, MAP_DIR)
from .motion import Motion, launcher_amp_for_level
from .render import (draw_apparatus, draw_background, draw_ball, draw_hoop,
                     draw_trail, world_to_screen)
from .maps import MapLevel, load_run, list_maps
from basketball_sprites.hoop_spawnv1 import HoopSprite


def builtin_run() -> List[MapLevel]:
    """The default 10-level ladder: one physics system per level, increasing
    launcher amplitude. This is a normal run — it can be replayed/edited."""
    return [
        MapLevel(name=f"{s.label}", system=s.id,
                 launcher=LAUNCHER_FRAC, hoop=HOOP_FRAC,
                 amp_m=launcher_amp_for_level(i + 1))
        for i, s in enumerate(SYSTEMS)
    ]


class Game:
    def __init__(self, W: int, H: int, screen: pygame.Surface,
                 run: Optional[List[MapLevel]] = None, run_name: str = "ladder"):
        self.W = W
        self.H = H
        self.screen = screen
        self.px = PX_PER_M
        self.run = run if run else builtin_run()
        self.run_name = run_name

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.bigfont = pygame.font.SysFont("consolas", 28, bold=True)

        # ball image (resized per-level if the map changes ball radius)
        self.ball_img = self._ball_image(BALL_RADIUS_M)

        # hoop sprite (calibrated)
        self.hoop_sprite = HoopSprite(image_path="assets/hoopnobgd.png", tolerance=60,
                                      rim_anchor_px=(94, 183))
        self.court_y_frac = 0.78

        # state
        self.score = 0
        self.level = 1          # 1-based index into self.run (wraps)
        self.paused = False
        self.trail = []
        self.TRAIL_MAX = 26
        self.time = 0.0
        self._accumulator = 0.0
        self._miss_triggered = False

        self._build_level()

    # ------------------------------------------------------------------ level
    def _ball_image(self, radius_m: float):
        d = max(8, int(radius_m * PX_PER_M * 2))
        return pygame.transform.smoothscale(
            pygame.image.load("assets/ball.png").convert_alpha(), (d, d))

    def current(self) -> MapLevel:
        i = (self.level - 1) % len(self.run)
        return self.run[i]

    def _build_level(self):
        cfg = self.current()
        spec = system_by_id(cfg.system)
        mr: MotionResult = run_system(spec, t_max=MOTION_T_MAX, fps=60)
        self.motion = Motion(mr, cfg.amp_m, self.W, self.H,
                             launcher_frac=cfg.launcher, hoop_frac=cfg.hoop)
        self.system_label = spec.label
        self.system_short = spec.short
        self.gravity = cfg.gravity
        self.ball_radius_m = cfg.ball_radius_m
        self.ball_img = self._ball_image(cfg.ball_radius_m)

        # launcher root (anchored at the launcher screen position)
        self.launcher_root = self.motion.root_position(
            self.W, self.H, launcher_frac=cfg.launcher)
        self.launcher = Launcher(self.motion, self.launcher_root)

        # hoop (fixed, at the hoop screen position)
        self.hoop = Hoop((cfg.hoop[0] * self.W / self.px, cfg.hoop[1] * self.H / self.px))

        # ball attached to the launcher
        self.ball = Ball(radius_m=cfg.ball_radius_m)
        self.ball.attach_to(self.launcher)
        self.trail.clear()
        self._miss_triggered = False

    # ------------------------------------------------------------------- step
    def update(self, frame_dt: float):
        if self.paused:
            return
        self.time += frame_dt
        self.launcher.step()

        self._accumulator += frame_dt
        dt = 1.0 / 60.0
        while self._accumulator >= dt:
            if self.ball.attached:
                self.ball.pos = self.launcher.bob_pos()
            else:
                self.ball.step(dt, g=self.gravity)
                self.trail.append(world_to_screen(tuple(self.ball.pos), self.px))
                if len(self.trail) > self.TRAIL_MAX:
                    self.trail.pop(0)

                if self.hoop.scored(self.ball):
                    self.score += 1
                    self.level += 1
                    self.on_score()
                    self._build_level()
                    break

                # miss / out-of-bounds
                ww, wh = self.W / self.px, self.H / self.px
                if (not self._miss_triggered and
                        (self.ball.pos[1] > wh * 0.98 or
                         self.ball.pos[0] < -0.3 or self.ball.pos[0] > ww + 0.3)):
                    self._miss_triggered = True
                if (self.ball.pos[0] < -1 or self.ball.pos[0] > ww + 1
                        or self.ball.pos[1] > wh + 1):
                    self._reset_ball()

            self._accumulator -= dt

    def _reset_ball(self):
        self.launcher.reset()
        self.ball.attach_to(self.launcher)
        self.trail.clear()
        self._miss_triggered = False

    def release(self):
        if self.ball.attached:
            self.ball.release_from(self.launcher)
            self.trail.clear()

    def on_score(self):
        pass  # hook for sound/flash (presentation layer reads self.score)

    # ------------------------------------------------------------------ input
    def handle(self, event) -> Optional[str]:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and self.ball.attached:
                self.release()
            elif event.key == pygame.K_r:
                self._reset_ball()
            elif event.key in (pygame.K_p, pygame.K_ESCAPE):
                self.paused = not self.paused
            elif event.key == pygame.K_m:
                return "MENU"
        return "GAME"

    # ------------------------------------------------------------------- draw
    def draw(self):
        screen = self.screen
        draw_background(screen, self.W, self.H, self.time, self.court_y_frac)
        draw_hoop(screen, self.hoop, self.hoop_sprite, self.px, self.court_y_frac, self.H)
        nodes = draw_apparatus(screen, self.launcher, self.px,
                               ball_attached=self.ball.attached)
        if not self.ball.attached:
            draw_trail(screen, self.trail)
        draw_ball(screen, self.ball, self.ball_img, self.px, self.court_y_frac, self.H)
        if self.ball.attached:
            bp = nodes.get("bob")
            if bp:
                pygame.draw.circle(screen, C["accent"], bp, 6, 2)
