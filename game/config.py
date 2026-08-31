"""Global configuration. All lengths are in METERS; the world is y-DOWN and
screen = world * PX_PER_M with the origin at the top-left."""
from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ASSET_DIR = BASE_DIR / "assets"
MAP_DIR = BASE_DIR / "maps"

FPS = 60
DT = 1.0 / FPS

PX_PER_M = 220.0          # meters -> pixels

# Design-world dimensions (meters) — the game is laid out for a world this size.
# On larger windows, PX_PER_M is scaled up so the world stays the same size in
# meters (the ball's throw is in meters, so a bigger world = further hoop =
# unwinnable).  See Game.__init__ for the dynamic computation.
DESIGN_WORLD_W_M = 921.0 / PX_PER_M   # 4.19 m
DESIGN_WORLD_H_M = 691.0 / PX_PER_M   # 3.14 m

G = 9.81

# Window sizing
WINDOW_SCALE = 0.90

# Launcher (ball) radius in meters
BALL_RADIUS_M = 0.12

# Hoop / scoring
HOOP_RIM_RADIUS_M = 0.20   # scoring radius (meters)
HOOP_SCORE_BAND_M = 0.12   # vertical band around rim center that counts

# Launcher amplitude (peak displacement of the driven node) in meters, by level
BASE_LAUNCHER_AMP_M = 0.55
AMP_PER_LEVEL_M = 0.06
MAX_LAUNCHER_AMP_M = 1.30

# Where the launcher (pendulum pivot) and the hoop base center sit, as fractions
# of the screen. The pendulum hangs upper-left; the hoop is on the right, at the
# height the ball's descending arc passes through it (verified reachable on all
# 10 systems). The hoop's motion (driven node) is clamped on-screen around this.
LAUNCHER_FRAC = (0.20, 0.10)
HOOP_FRAC = (0.55, 0.60)
HOOP_POLE_HEIGHT_M = 2.4   # pole length below the rim (meters)

# Motion precompute window (seconds) per level
MOTION_T_MAX = 90.0

# Palette (kept in one place)
C = {
    "sky_top": (233, 244, 255),
    "sky_bot": (206, 232, 255),
    "court": (243, 226, 204),
    "court_line": (205, 184, 162),
    "ink": (28, 28, 34),
    "ink_soft": (96, 100, 112),
    "panel": (255, 255, 255),
    "panel_edge": (220, 222, 232),
    "accent": (224, 74, 150),
    "accent2": (64, 120, 232),
    "ok": (52, 176, 110),
    "warn": (224, 168, 52),
    "rod": (92, 96, 110),
    "spring": (120, 84, 168),
    "mass": (58, 60, 72),
    "anchor": (70, 72, 84),
    "ball": (224, 110, 52),
}
