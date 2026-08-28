"""The Game engine: a *run* of levels, scoring, input, update/draw.

Per level:
  * a **pendulum launcher** (left) holds the ball;
  * a **physics-system apparatus** (right) carries the **hoop** on its driven
    node — the moving target.
Score = pass the ball through the rim at the right instant. A score advances the
run.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pygame

from physics.apparatus import SYSTEMS, run as run_system, system_by_id

from .bodies import Ball, HoopRig, Pendulum, choose_hoop_base
from .config import (G, HOOP_FRAC, MOTION_T_MAX, PX_PER_M, C, BALL_RADIUS_M,
                     MAP_DIR)
from .motion import Motion, launcher_amp_for_level
from .render import (draw_apparatus, draw_background, draw_ball, draw_hoop,
                     draw_pendulum, draw_trail)
from .maps import MapLevel, load_run, list_maps
from basketball_sprites.hoop_spawnv1 import HoopSprite

# Pendulum launcher geometry (the thing the player times). The pivot is hung
# from the UPPER-LEFT so the bob swings high and, on release, the ball arcs
# DOWN toward the hoop on the right (a proper basketball shot, not a flat throw).
PEND_PIVOT_FRAC = (0.20, 0.16)   # fraction of the window (upper-left)
PEND_L = 1.35                    # m (rod length)
PEND_A = 0.90                    # swing amplitude (rad)
PEND_PHI = 0.3                   # phase


def builtin_run() -> List[MapLevel]:
    """The default 10-level ladder: one physics system per level (driving the
    hoop), increasing hoop motion amplitude. A normal run — replayable/editable."""
    return [
        MapLevel(name=f"{s.label}", system=s.id,
                 launcher=PEND_PIVOT_FRAC, hoop=HOOP_FRAC,
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

        # score / miss flashes (the original's essential overlays)
        self.green_fn = pygame.image.load("assets/green_fn.png").convert_alpha()
        self.curry_moonshot = pygame.image.load("assets/curry_moonshot.png").convert_alpha()
        self.flash_duration = 1000  # ms
        self.im_green = False
        self.im_moonshot = False
        self.flash_start_green = 0
        self.flash_start_moonshot = 0

        self.ball_img = self._ball_image(BALL_RADIUS_M)
        self.hoop_sprite = HoopSprite(image_path="assets/hoopnobgd.png", tolerance=60,
                                      rim_anchor_px=(94, 183))
        self.court_y_frac = 0.78

        self.score = 0
        self.level = 1
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
        mr = run_system(spec, t_max=MOTION_T_MAX, fps=60)
        self.motion = Motion(mr, cfg.amp_m, self.W, self.H,
                             launcher_frac=cfg.launcher, hoop_frac=cfg.hoop)
        self.system_label = spec.label
        self.system_short = spec.short
        self.gravity = cfg.gravity
        self.ball_radius_m = cfg.ball_radius_m
        self.ball_img = self._ball_image(cfg.ball_radius_m)

        # hoop rides the physics system, on the right
        hoop_root = choose_hoop_base(self.motion, self.W, self.H, cfg.hoop)
        self.hoop_rig = HoopRig(self.motion, hoop_root)
        self.hoop = self.hoop_rig.hoop

        # ball hangs on a simple pendulum launcher, on the left
        pivot = (cfg.launcher[0] * self.W / self.px, cfg.launcher[1] * self.H / self.px)
        self.pend = Pendulum(pivot, PEND_L, PEND_A, self.gravity, PEND_PHI)
        self.ball = Ball(radius_m=cfg.ball_radius_m)
        self.ball.attach_to(self.pend)
        self.trail.clear()
        self._miss_triggered = False

    # ------------------------------------------------------------------- step
    def update(self, frame_dt: float):
        if self.paused:
            return
        self.time += frame_dt
        self.pend.step(frame_dt)
        self.hoop_rig.step()
        self.hoop_rig.update_hoop()

        self._accumulator += frame_dt
        dt = 1.0 / 60.0
        while self._accumulator >= dt:
            if self.ball.attached:
                self.ball.pos = self.pend.bob_pos()
            else:
                self.ball.step(dt, g=self.gravity)
                self.trail.append(tuple(self.ball.pos))
                if len(self.trail) > self.TRAIL_MAX:
                    self.trail.pop(0)

                if self.hoop.scored(self.ball):
                    self.score += 1
                    self.level += 1
                    self.im_green = True
                    self.flash_start_green = pygame.time.get_ticks()
                    self.on_score()
                    self._build_level()
                    break

                # miss flash (once per throw)
                ww, wh = self.W / self.px, self.H / self.px
                if (not self._miss_triggered and
                        (self.ball.pos[1] > wh * 0.90 or
                         self.ball.pos[0] < -0.2 or self.ball.pos[0] > ww + 0.2)):
                    self.im_moonshot = True
                    self.flash_start_moonshot = pygame.time.get_ticks()
                    self._miss_triggered = True

                # hard out-of-bounds -> re-attach
                ww, wh = self.W / self.px, self.H / self.px
                if (self.ball.pos[0] < -1 or self.ball.pos[0] > ww + 1
                        or self.ball.pos[1] > wh + 1):
                    self._reset_ball()

            self._accumulator -= dt

    def _reset_ball(self):
        self.pend.reset()
        self.ball.attach_to(self.pend)
        self.trail.clear()
        self._miss_triggered = False

    def release(self):
        if self.ball.attached:
            self.ball.release_from(self.pend)
            self.trail.clear()

    def on_score(self):
        pass  # hook for sound

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

        # the physics-system apparatus + the hoop on its driven node
        draw_apparatus(screen, self.hoop_rig, self.px,
                       ball_attached=False, driven_is_hoop=True)
        draw_hoop(screen, self.hoop, self.hoop_sprite, self.px, self.court_y_frac, self.H)

        # the pendulum launcher (left)
        draw_pendulum(screen, self.pend, self.px, self.ball.attached, self.court_y_frac, self.H)

        if not self.ball.attached:
            draw_trail(screen, self.trail)
        draw_ball(screen, self.ball, self.ball_img, self.px, self.court_y_frac, self.H)

        # score / miss flashes
        if self.im_green:
            self._flash(screen, self.green_fn, self.flash_start_green, "im_green")
        if self.im_moonshot:
            self._flash(screen, self.curry_moonshot, self.flash_start_moonshot, "im_moonshot")

    def _flash(self, screen, img, start_ms, flag):
        elapsed = pygame.time.get_ticks() - start_ms
        alpha = max(255 - 255 * (elapsed / self.flash_duration), 0)
        copy = img.copy()
        copy.set_alpha(int(alpha))
        # scale the overlay to the window
        if copy.get_size() != (self.W, self.H):
            copy = pygame.transform.smoothscale(copy, (self.W, self.H))
        screen.blit(copy, (0, 0))
        if alpha <= 0:
            setattr(self, flag, False)
