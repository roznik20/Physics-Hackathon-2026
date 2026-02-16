# game_main60_v3merge.py
# Merge runner: main_v3 gameplay + 60 FPS + moving hoop driven by physics motion arrays
#
# Requirements (repo-local):
#   - vladbasketv1.py (BasketAssembly)
#   - hoop_spawnv1.py (HoopSprite)
#   - physics/simple_pendulum.py (simulate_pendulum)
#   - assets/hoopnobgd.png, assets/green_fn.png, assets/curry_moonshot.png
#
# Controls:
#   SPACE = release ball
#   R     = reset ball + hoop motion (keeps score/level)
#   C     = calibrate hoop rim anchor (click rim center; saved to assets/hoop_anchor.json)
#   ESC   = exit calibration mode
#
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Tuple, Optional, Callable, Dict, List, Any

import ast
import inspect
from dataclasses import dataclass

import pygame
import numpy as np
import background as bg

from vladbasketv1 import BasketAssembly
from hoop_spawnv1 import HoopSprite

# =============================================================================
# Global configuration (make everything consistent at 60 FPS)
# =============================================================================
FPS = 60
DT = 1.0 / FPS

# Rendering / scale
WINDOW_SCALE = 0.90  # 90% of monitor, windowed
PX_PER_M = 220  # meters -> pixels (same as v2/v3 feel)

# Hoop scoring circle in world meters (cheap arcade scoring)
HOOP_RADIUS_M = 0.20

# Physics constants
G = 9.81

# Level motion: per-level motion model selection
LEVEL_MOTION_T_MAX = 120.0  # seconds of precomputed motion per level (clamps at end)
BASE_TARGET_AMP_M = 0.55  # target max amplitude of hoop motion at level 1 (meters)
AMP_PER_LEVEL_M = 0.06  # added target amplitude per level (meters)
MAX_TARGET_AMP_M = 1.15

# "Good" hoop center from main_v2/v3 (keeps the game scorable)
ORIG_HOOP_CENTER_M = (3.8, 2.15)

# Ball launcher (main_v3)
PEND_L = 1.35
PEND_A = 0.9
PEND_PHI = 0.3
PEND_PIVOT_M = (1.2, -1.0)  # y-down world

# Margins (pixels) for keeping hoop assembly visible
MARGIN_LEFT = 160
MARGIN_RIGHT = 160
MARGIN_TOP = 120
MARGIN_BOTTOM = 200

# =============================================================================
# Paths / assets
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "assets"
ANCHOR_JSON = ASSET_DIR / "hoop_anchor.json"


def asset_path(name: str) -> Path:
    """
    Always prefer ./assets/<name>. If not found, fall back to repo root ./<name>.
    This prevents file-not-found when teammates move assets around.
    """
    p = ASSET_DIR / name
    if p.exists():
        return p
    p2 = BASE_DIR / name
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Missing asset '{name}'. Looked in: {ASSET_DIR} and {BASE_DIR}")


# =============================================================================
# Helpers
# =============================================================================
def m2px(v: float) -> float:
    return v * PX_PER_M


def world_to_screen(pos_m: Tuple[float, float]) -> Tuple[int, int]:
    # World origin at top-left; y increases downward
    return (int(m2px(pos_m[0])), int(m2px(pos_m[1])))


def lerp(a, b, t):
    return a + (b - a) * t


def draw_vertical_gradient(screen, W, H, top_color, bottom_color):
    for y in range(H):
        t = y / (H - 1) if H > 1 else 0.0
        c = (
            int(lerp(top_color[0], bottom_color[0], t)),
            int(lerp(top_color[1], bottom_color[1], t)),
            int(lerp(top_color[2], bottom_color[2], t)),
        )
        pygame.draw.line(screen, c, (0, y), (W, y))


def aa_circle(screen, color, center, radius):
    try:
        import pygame.gfxdraw as gfx
        gfx.filled_circle(screen, center[0], center[1], radius, color)
        gfx.aacircle(screen, center[0], center[1], radius, color)
    except Exception:
        pygame.draw.circle(screen, color, center, radius)


def aa_line(screen, color, p1, p2, width=2):
    if width <= 1:
        pygame.draw.aaline(screen, color, p1, p2)
    else:
        pygame.draw.line(screen, color, p1, p2, width)


def draw_text(screen, font, text, x, y, color=(25, 25, 25)):
    img = font.render(text, True, color)
    screen.blit(img, (x, y))


# =============================================================================
# Hoop rig visualization (anchors + strings/springs) to make the motion "feel physical"
# =============================================================================
@dataclass
class HoopRig:
    kind: str  # "rod", "spring", "springs2d", "spring_horizontal", "double_pendulum", "none"
    anchors_m: List[Tuple[float, float]]
    label: str
    coils: int = 12
    amp_px: int = 8


def _clamp_anchor_m(p: Tuple[float, float], W: int, H: int, px_per_m: float, margin_px: int = 18) -> Tuple[
    float, float]:
    world_w = W / px_per_m
    world_h = H / px_per_m
    m = margin_px / px_per_m
    return (max(m, min(world_w - m, p[0])), max(m, min(world_h - m, p[1])))


def _max_amp_m(dx_m: np.ndarray, dy_m: np.ndarray) -> float:
    try:
        if dx_m is None or dy_m is None or len(dx_m) == 0:
            return 0.0
        return float(np.max(np.hypot(dx_m, dy_m)))
    except Exception:
        return 0.0


def build_hoop_rig(motion_name: str, base_center_m: Tuple[float, float],
                   dx_m: np.ndarray, dy_m: np.ndarray,
                   W: int, H: int, px_per_m: float) -> HoopRig:
    """Pick fixed anchor point(s) + connector style based on the current motion model."""
    name = (motion_name or "").lower()
    amp = _max_amp_m(dx_m, dy_m)

    # A visually reasonable "rig size" in meters (bigger than the motion envelope)
    L = max(0.65, amp + 0.55)

    if "double pendulum" in name:
        pivot = _clamp_anchor_m((base_center_m[0], base_center_m[1] - L), W, H, px_per_m)
        return HoopRig(kind="double_pendulum", anchors_m=[pivot], label="Double pendulum (chaos)")
    if "three" in name:
        pivot = _clamp_anchor_m((base_center_m[0], base_center_m[1] - L), W, H, px_per_m)
        return HoopRig(kind="rod", anchors_m=[pivot], label="Three pendulum chain")
    if "cart" in name:
        pivot = _clamp_anchor_m((base_center_m[0], base_center_m[1] - L), W, H, px_per_m)
        return HoopRig(kind="rod", anchors_m=[pivot], label="Pendulum cart")
    if "spring pendulum" in name:
        pivot = _clamp_anchor_m((base_center_m[0], base_center_m[1] - L), W, H, px_per_m)
        return HoopRig(kind="spring", anchors_m=[pivot], label="Spring pendulum")
    if "2d springs" in name or "2d spring" in name:
        left = _clamp_anchor_m((base_center_m[0] - (L + 0.35), base_center_m[1]), W, H, px_per_m)
        top = _clamp_anchor_m((base_center_m[0], base_center_m[1] - (L + 0.25)), W, H, px_per_m)
        return HoopRig(kind="springs2d", anchors_m=[left, top], label="2D springs")
    if "damped" in name or "driven" in name:
        left = _clamp_anchor_m((base_center_m[0] - (L + 0.55), base_center_m[1]), W, H, px_per_m)
        return HoopRig(kind="spring_horizontal", anchors_m=[left], label="Driven damped spring")
    if "pendulum" in name:
        pivot = _clamp_anchor_m((base_center_m[0], base_center_m[1] - L), W, H, px_per_m)
        return HoopRig(kind="rod", anchors_m=[pivot], label="Simple pendulum")
    if "station" in name:
        pivot = _clamp_anchor_m((base_center_m[0], base_center_m[1] - 0.75), W, H, px_per_m)
        return HoopRig(kind="rod", anchors_m=[pivot], label="Fixed mount")

    # Fallback: a single pivot/rod
    pivot = _clamp_anchor_m((base_center_m[0], base_center_m[1] - L), W, H, px_per_m)
    return HoopRig(kind="rod", anchors_m=[pivot], label=motion_name or "Motion")


def draw_spring(screen: pygame.Surface, p1: Tuple[int, int], p2: Tuple[int, int], *,
                color=(90, 90, 90), width: int = 3, coils: int = 12, amp_px: int = 8) -> None:
    """Draw a simple zig-zag spring between two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 1.0:
        return

    ux, uy = dx / dist, dy / dist
    # Perpendicular
    px, py = -uy, ux

    pts = [(x1, y1)]
    # leave small straight segments at both ends
    end_pad = 0.10 * dist
    start = end_pad
    end = dist - end_pad
    if end <= start:
        pygame.draw.line(screen, color, p1, p2, width)
        return

    for i in range(1, coils + 1):
        t = start + (end - start) * (i / (coils + 1))
        sgn = -1 if (i % 2 == 0) else 1
        pts.append((x1 + ux * t + px * amp_px * sgn, y1 + uy * t + py * amp_px * sgn))
    pts.append((x2, y2))

    pygame.draw.lines(screen, color, False, [(int(a), int(b)) for a, b in pts], width)


def draw_hoop_rig(screen: pygame.Surface, rig: HoopRig,
                  rim_center_px: Tuple[int, int],
                  world_to_screen_fn: Callable[[Tuple[float, float]], Tuple[int, int]],
                  font: pygame.font.Font) -> None:
    """
    Render anchors/connectors ONLY for:
        - 2D springs
        - Driven damped spring (horizontal spring)
        - Simple pendulum
        - Spring pendulum
    All other systems are handled elsewhere.
    """

    if rig is None or rig.kind == "none":
        return

    # Only allow specific rig kinds
    not_allowed = {"double", "three", "horizontal", "cart", "vertical"}

    name = rig.label.lower()

    if any(word in name for word in not_allowed):
        return

    # Subtle behind-hoop styling
    anchor_col = (35, 35, 35)
    rod_col = (80, 80, 80)
    spring_col = (70, 70, 90)

    # Label near first anchor
    if rig.anchors_m:
        a0_px = world_to_screen_fn(rig.anchors_m[0])
        draw_text(
            screen,
            font,
            f"Rig: {rig.label}",
            a0_px[0] + 14,
            a0_px[1] - 22,
            color=(20, 20, 20),
        )

    # Draw anchors + connectors
    for a_m in rig.anchors_m:
        a_px = world_to_screen_fn(a_m)

        # anchor point
        aa_circle(screen, anchor_col, a_px, 6)

        # springs
        if rig.kind in ("spring", "springs2d", "spring_horizontal"):
            draw_spring(
                screen,
                a_px,
                rim_center_px,
                color=spring_col,
                width=3,
                coils=rig.coils,
                amp_px=rig.amp_px,
            )
        else:
            # rod (simple pendulum)
            aa_line(screen, rod_col, a_px, rim_center_px, width=5)

        # horizontal wall for driven damped spring
        if rig.kind == "spring_horizontal":
            pygame.draw.rect(screen, anchor_col,
                             (a_px[0] - 10, a_px[1] - 18, 6, 36))
            pygame.draw.rect(screen, anchor_col,
                             (a_px[0] - 16, a_px[1] - 18, 6, 36))


def load_anchor_from_json() -> Optional[Tuple[int, int]]:
    try:
        if ANCHOR_JSON.exists():
            data = json.loads(ANCHOR_JSON.read_text(encoding="utf-8"))
            ax, ay = data.get("rim_anchor_px", [None, None])
            if ax is None or ay is None:
                return None
            return (int(ax), int(ay))
    except Exception:
        return None
    return None


def save_anchor_to_json(anchor: Tuple[int, int]) -> None:
    try:
        ANCHOR_JSON.write_text(json.dumps({"rim_anchor_px": [int(anchor[0]), int(anchor[1])]}, indent=2),
                               encoding="utf-8")
    except Exception as e:
        print("WARNING: failed to save hoop anchor JSON:", e)


# =============================================================================
# Physics objects (world meters, y-down)
# =============================================================================
class Pendulum:
    """Small-angle analytic pendulum: theta = A cos(omega t + phi)."""

    def __init__(self, pivot_m=(1.5, 1.2), L=1.2, A=0.75, g=9.81, phi=0.0):
        self.pivot = pivot_m
        self.L = L
        self.A = A
        self.g = g
        self.phi = phi
        self.t = 0.0

    def omega(self):
        return math.sqrt(max(self.g, 0.0) / max(self.L, 1e-6))

    def theta(self):
        return self.A * math.cos(self.omega() * self.t + self.phi)

    def theta_dot(self):
        return -self.A * self.omega() * math.sin(self.omega() * self.t + self.phi)

    def ball_pos(self):
        th = self.theta()
        x = self.pivot[0] + self.L * math.sin(th)
        y = self.pivot[1] + self.L * math.cos(th)  # y-down world
        return (x, y)

    def ball_vel(self):
        th = self.theta()
        thd = self.theta_dot()
        vx = self.L * thd * math.cos(th)
        vy = -self.L * thd * math.sin(th)  # y-down world
        return (vx, vy)

    def step(self, dt):
        self.t += dt


class Ball:
    def __init__(self, pos_m=(0, 0), vel_mps=(0, 0), radius_m=0.12):
        self.pos = pos_m
        self.vel = vel_mps
        self.r = radius_m
        self.attached = True

    def attach_to(self, pendulum: Pendulum):
        self.attached = True
        self.pos = pendulum.ball_pos()
        self.vel = (0.0, 0.0)

    def release_from(self, pendulum: Pendulum):
        self.attached = False
        self.pos = pendulum.ball_pos()
        self.vel = pendulum.ball_vel()

    def step(self, dt, g=9.81, wind_ax=0.0):
        if self.attached:
            return
        ax, ay = wind_ax, g  # y-down gravity is +g
        self.vel = (self.vel[0] + ax * dt, self.vel[1] + ay * dt)
        self.pos = (self.pos[0] + self.vel[0] * dt, self.pos[1] + self.vel[1] * dt)


class Hoop:
    """Score zone as a circle (meters). Drawing is handled by PNG sprite."""

    def __init__(self, center_m=(2.0, 2.1), radius_m=HOOP_RADIUS_M):
        self.c = center_m
        self.r = radius_m

    def set_center(self, center_m: Tuple[float, float]):
        self.c = center_m

    def scored(self, ball: Ball) -> bool:
        dx = ball.pos[0] - self.c[0]
        dy = ball.pos[1] - self.c[1]
        return (dx * dx + dy * dy) <= (self.r * self.r)


# =============================================================================
# Hoop motion arrays (physics offsets) — LEVEL SYSTEM
# =============================================================================
PHYSICS_DIR = BASE_DIR / "physics"

# The physics team returns positions in **meters** (y-up in their math). Our game uses **meters** with y-down.
# We treat every model as y-up, center it about its mean, then flip y to y-down.

_FUNC_CACHE: Dict[tuple[str, str], Callable[..., Any]] = {}


def _safe_load_function(py_path: Path, func_name: str) -> Callable[..., Any]:
    """Load a single function from a physics module *without executing* its demo/plot code."""
    key = (str(py_path.resolve()), func_name)
    if key in _FUNC_CACHE:
        return _FUNC_CACHE[key]

    if not py_path.exists():
        raise FileNotFoundError(f"Physics module not found: {py_path}")

    src = py_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(py_path))

    banned_prefixes = ("matplotlib", "sympy", "pygame")

    new_body = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            kept = []
            for alias in node.names:
                if alias.name.startswith(banned_prefixes):
                    continue
                kept.append(alias)
            if kept:
                node.names = kept
                new_body.append(node)

        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").startswith(banned_prefixes):
                continue
            # also drop specific heavy submodules
            if node.module in ("matplotlib.animation",):
                continue
            new_body.append(node)

        elif isinstance(node, ast.FunctionDef) and node.name == func_name:
            new_body.append(node)

    mod = ast.Module(body=new_body, type_ignores=[])
    ast.fix_missing_locations(mod)
    code = compile(mod, filename=str(py_path), mode="exec")
    ns: Dict[str, Any] = {}
    exec(code, ns)

    if func_name not in ns or not callable(ns[func_name]):
        raise AttributeError(f"Function '{func_name}' not found in {py_path.name}")

    _FUNC_CACHE[key] = ns[func_name]
    return ns[func_name]


@dataclass
class MotionSpec:
    name: str
    filename: str
    func_name: str
    # adapter receives the raw return tuple/list and must produce (t, x, y)
    adapter: Callable[[Any], tuple[Any, Any, Any]]
    default_kwargs: Dict[str, Any]


def _adapt_txy(out):
    t, x, y = out
    return t, x, y


def _adapt_forced(out):
    t, x, x_wall = out
    return t, x, x_wall


def _adapt_2D(out):
    t_eval, x, y, vertical_spring_x, vertical_spring_y, horizontal_spring_x, horizontal_spring_y = out
    return t_eval, x, y


def _adapt_tx(out):
    t, x = out
    # y will be zero; caller will handle
    return t, x, None


def _adapt_double_pend(out):
    t, x1, y1, x2, y2 = out
    return t, x2, y2


def _adapt_spring_pend(out):
    t, x, y = out
    return t, x, y


def _adapt_damped_spring(out):
    # returns (t, x_mass, x_wall)
    t, x, _x_wall = out
    return t, x, None


def _adapt_three_pend(out):
    t, x0, x1, x2 = out[:4]
    return t, x2, None


def _adapt_cart(out):
    t, x_cart, y_cart, x_pend, y_pend = out
    return t, x_pend, y_pend


def _adapt_vertical(out):
    t, x1, x2, total_length = out
    return t, None, x2


class MotionManager:
    """Cycles through different physics-based hoop motions as levels increase."""

    def __init__(self, physics_dir: Path, fps: int, t_max: float):
        self.physics_dir = physics_dir
        self.fps = int(fps)
        self.t_max = float(t_max)

        self.specs: List[MotionSpec] = [
            MotionSpec(
                name="Horizontal spring",
                filename="horizontal_spring.py",
                func_name="simulate_pendulum",
                adapter=_adapt_forced,
                default_kwargs={},
            ),
            MotionSpec(
                name="Three pendulum",
                filename="horizontal_three_pend.py",
                func_name="simulate_pendulum",
                adapter=_adapt_three_pend,
                default_kwargs={},
            ),
            MotionSpec(
                name="Pendulum cart",
                filename="pendulum_cart.py",
                func_name="simulate_pendulum",
                adapter=_adapt_cart,
                default_kwargs={},
            ),
            MotionSpec(
                name="Vertical double spring",
                filename="verticle_double_spring.py",
                func_name="simulate_vertical_2mass",
                adapter=_adapt_vertical,
                default_kwargs={},
            ),

            MotionSpec(
                name="Stationary",
                filename="stationiary.py",
                func_name="simulate_pendulum",
                adapter=_adapt_txy,
                default_kwargs={},
            ),
            MotionSpec(
                name="Simple pendulum",
                filename="simple_pendulum.py",
                func_name="simulate_pendulum",
                adapter=_adapt_txy,
                default_kwargs={"L_val": .5, "g_val": .5*G, "theta0": 1.0, "omega0": 0},
            ),
            MotionSpec(
                name="2D springs",
                filename="springs_2d.py",
                func_name="simulate_pendulum",
                adapter=_adapt_2D,
                default_kwargs={"L_extention_x": 0.35, "L_extention_y": 0.25, "m": 2, "k_x": 6, "k_y": 4},
            ),
            MotionSpec(
                name="Driven damped spring (1D)",
                filename="damped_spring.py",
                func_name="simulate_pendulum",
                adapter=_adapt_damped_spring,
                default_kwargs={"b": 0.25, "omega": 2.2, "d": 0.40, "m": 2, "x0": -0.4, "x_dis": 0.25, "v_ini": 1,
                                "k": 10},
            ),
            MotionSpec(
                name="Spring pendulum",
                filename="spring_pendulum.py",
                func_name="simulate_pendulum",
                adapter=_adapt_spring_pend,
                default_kwargs={"m": 1.0, "k": 25.0, "L0": 0.9, "g": G, "r0": 1.0, "theta0": 1.0, "rdot0": 0.0,
                                "thetadot0": 2.0},
            ),
            MotionSpec(
                name="Double pendulum (chaos)",
                filename="double_pendulum.py",
                func_name="simulate_pendulum",
                adapter=_adapt_double_pend,
                default_kwargs={"m1": 2.0, "m2": 1.0, "r1": 0.75, "r2": 0.70, "theta1_0": 1.2, "omega1_0": 0.0,
                                "theta2_0": -1.4, "omega2_0": 0.0, "g": G},
            ),
        ]

        # Cache of centered (but not amplitude-scaled) base motions: name -> (t, dx0, dy0)
        self._base_cache: Dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def spec_for_level(self, level: int) -> MotionSpec:
        idx = (max(1, int(level)) - 1) % len(self.specs)
        return self.specs[idx]

    def _call_sim(self, spec: MotionSpec):
        fn = _safe_load_function(self.physics_dir / spec.filename, spec.func_name)
        sig = inspect.signature(fn)

        kwargs = dict(spec.default_kwargs)
        if "fps" in sig.parameters:
            kwargs["fps"] = self.fps
        if "t_max" in sig.parameters:
            kwargs["t_max"] = self.t_max

        # Some signatures use different names (rare). Just ignore unknown kwargs.
        call_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(**call_kwargs)

    def get_level_motion(self, level: int):
        spec = self.spec_for_level(level)

        if spec.name not in self._base_cache:
            out = self._call_sim(spec)

            # ---------- SYSTEM-SPECIFIC EXTRA DATA ----------
            extra = {}

            if spec.name.lower().startswith("double"):
                t, x1, y1, x2, y2 = out
                extra["mid"] = (np.asarray(x1), np.asarray(y1))
                x = x2
                y = y2

            elif spec.name.lower().startswith("horizontal spring"):
                t, x, wall = out
                extra["wall"] = np.asarray(wall)
                y = np.zeros_like(x)

            elif spec.name.lower().startswith("three"):
                t, x0, x1, x2 = out[:4]
                extra["boxes"] = (np.asarray(x0), np.asarray(x1))
                x = x2
                y = np.zeros_like(x)

            elif spec.name.lower().startswith("pendulum cart"):
                t, x_cart, y_cart, x_pend, y_pend = out
                extra["cart"] = (np.asarray(x_cart), np.asarray(y_cart))
                x = x_pend
                y = y_pend

            elif spec.name.lower().startswith("vertical"):
                t, x1, x2, total_length = out
                extra["bottom"] = np.asarray(x1)

                # Vertical motion only
                x = np.zeros_like(x2)
                y = np.asarray(x2)

            else:
                t, x, y = spec.adapter(out)

            t = np.asarray(t, float)
            x = np.asarray(x, float)
            y = np.asarray(y, float) if y is not None else np.zeros_like(x)

            dx_up = x - float(np.mean(x))
            dy_up = y - float(np.mean(y))

            dx0 = dx_up
            dy0 = -dy_up

            self._base_cache[spec.name] = (t, dx0, dy0, extra)

        t, dx0, dy0, extra = self._base_cache[spec.name]

        target = min(BASE_TARGET_AMP_M + AMP_PER_LEVEL_M * (max(1, int(level)) - 1), MAX_TARGET_AMP_M)
        max_abs = float(max(np.max(np.abs(dx0)), np.max(np.abs(dy0)), 1e-9))
        scale = target / max_abs

        dx = dx0 * scale
        dy = dy0 * scale

        if "mid" in extra:  # double pendulum
            mx, my = extra["mid"]
            extra["mid"] = (mx * scale, my * scale)

        return t, dx, dy, spec.name, extra


# Fallback stationary motion if something explodes
_DEFAULT_MOTION_MGR: Optional[MotionManager] = None


def load_motion_arrays_for_level(level: int, fps: int):
    global _DEFAULT_MOTION_MGR
    if _DEFAULT_MOTION_MGR is None:
        _DEFAULT_MOTION_MGR = MotionManager(PHYSICS_DIR, fps=fps, t_max=LEVEL_MOTION_T_MAX)

    return _DEFAULT_MOTION_MGR.get_level_motion(level)

    # try:
    return _DEFAULT_MOTION_MGR.get_level_motion(level)
    # except Exception as e:
    #     print(f"WARNING: failed to generate motion arrays for level {level}: {e}")
    #     # 1 second of zeros (but clamp will stop)
    #     t = np.linspace(0.0, 1.0, int(1.0 * fps))
    #     z = np.zeros_like(t)
    #     return t, z, z, "(fallback)"


def choose_base_center_m(W: int, H: int, dx_m: np.ndarray, dy_m: np.ndarray) -> Tuple[float, float]:
    """
    Choose a hoop base center (world meters) so the entire motion stays on-screen.
    Start near ORIG_HOOP_CENTER_M (keeps game scorable), then clamp into allowed region.
    """
    dx_px = dx_m * PX_PER_M
    dy_px = dy_m * PX_PER_M

    min_dx, max_dx = float(dx_px.min()), float(dx_px.max())
    min_dy, max_dy = float(dy_px.min()), float(dy_px.max())

    lo_x = MARGIN_LEFT - min_dx
    hi_x = (W - MARGIN_RIGHT) - max_dx
    lo_y = MARGIN_TOP - min_dy
    hi_y = (H - MARGIN_BOTTOM) - max_dy

    base0_px = world_to_screen(ORIG_HOOP_CENTER_M)
    if lo_x <= hi_x:
        base_x_px = float(np.clip(base0_px[0], lo_x, hi_x))
    else:
        base_x_px = float(W / 2)

    if lo_y <= hi_y:
        base_y_px = float(np.clip(base0_px[1], lo_y, hi_y))
    else:
        base_y_px = float(H / 2)

    return (base_x_px / PX_PER_M, base_y_px / PX_PER_M)


# =============================================================================
# States
# =============================================================================
class Menu:
    def __init__(self, W: int, H: int):
        self.screen_width = W
        self.screen_height = H

        self.button_width = 200
        self.button_height = 80

        self.button_rect = pygame.Rect(
            self.screen_width // 2 - self.button_width // 2,
            self.screen_height // 2 - self.button_height // 2,
            self.button_width,
            self.button_height
        )

        self.font = pygame.font.SysFont(None, 48)
        self.title_font = pygame.font.SysFont(None, 100)
        self.title_color = (10, 10, 10)

        self.bg_color = (30, 30, 30)
        self.button_color = (255, 153, 204)
        self.button_hover_color = (218, 112, 214)
        self.text_color = (255, 255, 255)

        # bg images
        self.bliss = pygame.image.load("assets/bliss.jpg")
        self.lebron = pygame.image.load("assets/lebron.png")

        self.bliss_scaled, self.bliss_pos = self.scale_and_center_image(self.bliss)
        self.lebron_scaled, self.lebron_pos = self.scale_and_center_image(self.lebron)

        # render the text
        self.title_surf = self.title_font.render("LeHoop and Ball Game", True, self.title_color)

        # get centered position
        self.title_rect = self.title_surf.get_rect(center=(self.screen_width // 2, 80))  # y=80 pixels from top

    def scale_and_center_image(self, image):
        # get original size
        img_w, img_h = image.get_size()

        # compute scale factors
        scale_w = self.screen_width / img_w
        scale_h = self.screen_height / img_h
        scale = max(scale_w, scale_h)  # ensures no empty space

        # compute new size and scale image
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        image = pygame.transform.scale(image, (new_w, new_h))

        # compute centered position
        x = (self.screen_width - new_w) // 2
        y = (self.screen_height - new_h) // 2

        return image, (x, y)

    def handle_events(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.button_rect.collidepoint(event.pos):
                return "GAME"
        return "MENU"

    def update(self, dt):
        pass

    def draw(self, screen):
        screen.blit(self.bliss_scaled, self.bliss_pos)
        screen.blit(self.lebron_scaled, self.lebron_pos)
        screen.blit(self.title_surf, self.title_rect)

        mouse_pos = pygame.mouse.get_pos()
        color = self.button_hover_color if self.button_rect.collidepoint(mouse_pos) else self.button_color
        pygame.draw.rect(screen, color, self.button_rect)

        text = self.font.render("PLAY", True, (255, 255, 255))
        text_rect = text.get_rect(center=self.button_rect.center)
        screen.blit(text, text_rect)


class Game:
    def __init__(self, W: int, H: int):
        self.W = W
        self.H = H
        self.FPS = FPS
        self.DT = DT
        self.px_per_m = PX_PER_M

        # Colors
        self.white = (245, 245, 245)
        self.black = (25, 25, 25)
        self.red = (220, 60, 60)
        self.blue = (50, 90, 220)
        self.gray = (180, 180, 180)
        self.green = (60, 190, 90)
        self.skytop = (235, 245, 255)
        self.skybot = (210, 230, 255)
        self.court = (245, 230, 210)
        self.courtline = (210, 190, 170)

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.bigfont = pygame.font.SysFont("consolas", 28, bold=True)

        self.g = G
        self.L = PEND_L
        self.A = PEND_A
        self.wind_ax = 0.0

        # Assets
        self.green_fn = pygame.image.load(str(asset_path("green_fn.png"))).convert_alpha()
        self.curry_moonshot = pygame.image.load(str(asset_path("curry_moonshot.png"))).convert_alpha()

        # Score/level
        self.score = 0
        self.level = 1

        # Trail
        self.trail = []
        self.TRAIL_MAX = 28

        self.win = False

        # Timing
        self.accumulator = 0.0
        # Instantiate motion arrays for current level (world meters offsets)
        (
            self.motion_t,
            self.motion_dx_m,
            self.motion_dy_m,
            self.motion_name,
            self.motion_extra,
        ) = load_motion_arrays_for_level(self.level, self.FPS)
        self.motion_i = 0
        self.n_motion = len(self.motion_t)

        # Choose base hoop center so motion stays on-screen
        self.base_center_m = choose_base_center_m(self.W, self.H, self.motion_dx_m, self.motion_dy_m)
        delta_m = (self.base_center_m[0] - ORIG_HOOP_CENTER_M[0], self.base_center_m[1] - ORIG_HOOP_CENTER_M[1])

        # Objects
        self.pend = Pendulum(
            pivot_m=(PEND_PIVOT_M[0] + delta_m[0], PEND_PIVOT_M[1] + delta_m[1]),
            L=self.L,
            A=self.A,
            g=self.g,
            phi=PEND_PHI
        )
        self.ball = Ball(radius_m=0.12)
        self.ball.attach_to(self.pend)

        self.hoop = Hoop(center_m=self.base_center_m, radius_m=HOOP_RADIUS_M)
        self.hoop_center_m = self.base_center_m

        # Basket assembly (pixel geometry + clamping)
        self.basket = BasketAssembly(center_xy=world_to_screen(self.base_center_m))
        self.hoop_rig = build_hoop_rig(self.motion_name, self.base_center_m, self.motion_dx_m, self.motion_dy_m, self.W,
                                       self.H, self.px_per_m)

        # Hoop sprite (PNG)
        rim_anchor = load_anchor_from_json()
        self.hoop_sprite = HoopSprite(
            image_path=str(asset_path("hoopnobgd.png")),
            tolerance=60,
            rim_anchor_px=rim_anchor,
        )

        # Image flash (overlay)
        self.im_green = False
        self.im_moonshot = False
        self.flash_start_time = 0
        self.flash_duration = 1000  # ms

        # Calibration
        self.calibrating = False
        self.cal_top_left = None

        # Background variables

        self.background = bg.Background(self.W, self.H)

    # --- coordinate helpers ---
    def m2px(self, v):
        return v * self.px_per_m

    def world_to_screen(self, pos_m):
        return (int(self.m2px(pos_m[0])), int(self.m2px(pos_m[1])))

    def _apply_level_motion(self, new_level: int, *, reset_ball: bool = True) -> None:
        """Switch to the motion model for `new_level`, reset motion index, and keep things on-screen.

        This is called after a SCORE to change the hoop motion model (levels).
        """
        (t, dx_m, dy_m, name, extra) = load_motion_arrays_for_level(new_level, self.FPS)
        self.motion_t, self.motion_dx_m, self.motion_dy_m = t, dx_m, dy_m
        self.motion_name = name
        self.motion_i = 0
        self.n_motion = len(self.motion_t)
        self.motion_extra = extra

        # Recompute base center and apply consistent shift so shots remain scorable
        self.base_center_m = choose_base_center_m(self.W, self.H, self.motion_dx_m, self.motion_dy_m)
        delta_m = (self.base_center_m[0] - ORIG_HOOP_CENTER_M[0], self.base_center_m[1] - ORIG_HOOP_CENTER_M[1])

        # Update pendulum pivot to keep relative geometry consistent
        self.pend.pivot = (PEND_PIVOT_M[0] + delta_m[0], PEND_PIVOT_M[1] + delta_m[1])
        self.pend.t = 0.0

        # Update hoop + basket
        self.hoop_center_m = self.base_center_m
        self.hoop.set_center(self.hoop_center_m)
        self.basket = BasketAssembly(center_xy=world_to_screen(self.base_center_m))
        self.hoop_rig = build_hoop_rig(self.motion_name, self.base_center_m, self.motion_dx_m, self.motion_dy_m, self.W,
                                       self.H, self.px_per_m)

        if reset_ball:
            self.ball.attach_to(self.pend)
            self.trail.clear()

    # --- events ---
    def handle_events(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                # Reset ball + motion (NOT score/level)
                self.pend.t = 0.0
                self.ball.attach_to(self.pend)
                self.trail.clear()
                self.motion_i = 0

            if event.key == pygame.K_SPACE:
                if (not self.calibrating) and self.ball.attached:
                    self.ball.release_from(self.pend)
                    self.trail.clear()

            if event.key == pygame.K_c:
                self.calibrating = not self.calibrating
                if self.calibrating:
                    self.cal_top_left = (
                        self.W // 2 - self.hoop_sprite.surface.get_width() // 2,
                        self.H // 2 - self.hoop_sprite.surface.get_height() // 2
                    )

            if event.key == pygame.K_ESCAPE and self.calibrating:
                self.calibrating = False

            if event.key == pygame.K_ESCAPE and (not self.calibrating):
                return "MENU"

            if event.key == pygame.K_s:
                self.score += 1
                self.level += 1
                self.flash_start_time = pygame.time.get_ticks()
                # Switch hoop motion model for the NEW level, and reset ball + motion
                self._apply_level_motion(self.level, reset_ball=True)

        if event.type == pygame.MOUSEBUTTONDOWN and self.calibrating and self.cal_top_left is not None:
            mx, my = event.pos
            ax = mx - self.cal_top_left[0]
            ay = my - self.cal_top_left[1]
            if 0 <= ax < self.hoop_sprite.surface.get_width() and 0 <= ay < self.hoop_sprite.surface.get_height():
                self.hoop_sprite.rim_anchor.update(ax, ay)
                save_anchor_to_json((int(ax), int(ay)))
                print(f"RIM ANCHOR PICKED: rim_anchor_px=({ax}, {ay}) saved to {ANCHOR_JSON}")
                self.calibrating = False

        return "GAME"

    # --- update ---
    def update(self, frame_dt: float):
        # Advance hoop motion ONCE PER FRAME (clamp at end)
        if (not self.calibrating) and (self.motion_i < self.n_motion - 1):
            self.motion_i += 1

        self.hoop_center_m = (
            self.base_center_m[0] + float(self.motion_dx_m[self.motion_i]),
            self.base_center_m[1] + float(self.motion_dy_m[self.motion_i]),
        )
        self.hoop.set_center(self.hoop_center_m)

        if self.calibrating:
            return

        # Fixed-step physics at 60 FPS
        self.background.update()
        self.accumulator += frame_dt
        while self.accumulator >= self.DT:
            self.pend.step(self.DT)

            if self.ball.attached:
                self.ball.pos = self.pend.ball_pos()
            else:
                self.ball.step(self.DT, g=self.g, wind_ax=self.wind_ax)

                # score check (meters)
                if self.hoop.scored(self.ball):
                    self.score += 1
                    self.level += 1
                    self.im_green = True
                    self.flash_start_time = pygame.time.get_ticks()

                    # Switch hoop motion model for the NEW level, and reset ball + motion
                    self._apply_level_motion(self.level, reset_ball=True)

                elif self.hoop.scored(self.ball):
                    self.score += 1
                    # self.win removed for infinite levels

                else:
                    # miss flash (soft)
                    world_w = self.W / self.px_per_m
                    world_h = self.H / self.px_per_m
                    if self.ball.pos[1] > world_h * 0.90 or self.ball.pos[0] < -0.2 or self.ball.pos[0] > world_w + 0.2:
                        self.im_moonshot = True
                        self.flash_start_time = pygame.time.get_ticks()

                # hard out-of-bounds reset
                world_w = self.W / self.px_per_m
                world_h = self.H / self.px_per_m
                if self.ball.pos[0] < -1 or self.ball.pos[0] > world_w + 1 or self.ball.pos[1] > world_h + 1:
                    self.pend.t = 0.0
                    self.ball.attach_to(self.pend)
                    self.trail.clear()

            self.accumulator -= self.DT

    def draw_physics_system(game, screen, rim_center_px):
        name = game.motion_name.lower()
        i = game.motion_i
        extra = game.motion_extra

        def to_px(x, y=0):
            return game.world_to_screen((game.base_center_m[0] + x, game.base_center_m[1] + y))

        # ---------- DOUBLE PENDULUM ----------
        if "double pendulum" in name and "mid" in extra:
            mid_x, mid_y = extra["mid"]

            # Height offset in WORLD units (not pixels)
            world_shift = (game.H * 0.13) / game.px_per_m

            # Base anchor (physics origin) shifted upward once
            pivot_world = (
                game.base_center_m[0],
                game.base_center_m[1] - world_shift
            )

            pivot_px = game.world_to_screen(pivot_world)

            # Middle pendulum relative to that same anchor
            mid_world = (
                pivot_world[0] + mid_x[i],
                pivot_world[1] - mid_y[i]  # physics y-up → screen y-down
            )

            mid_px = game.world_to_screen(mid_world)

            # Draw rods
            aa_line(screen, (0, 0, 0), pivot_px, mid_px, 4)
            aa_line(screen, (0, 0, 0), mid_px, rim_center_px, 4)

            # Draw joints
            aa_circle(screen, (0, 0, 0), pivot_px, 6)
            aa_circle(screen, (0, 0, 0), mid_px, 6)

            return

        # ---------- HORIZONTAL SPRING ----------
        if "horizontal spring" in name and "wall" in extra:
            wall_x = extra["wall"][i]
            wall_px = to_px(wall_x)

            pygame.draw.rect(screen, (40, 40, 40), (wall_px[0] - 10, wall_px[1] - 40, 20, 80))
            draw_spring(screen, wall_px, rim_center_px)
            return

        # ---------- THREE PENDULUM ----------
        if "three" in name and "boxes" in extra:
            b1, b2 = extra["boxes"]
            p1 = to_px(b1[i])
            p2 = to_px(b2[i])

            wall = (p1[0] - 120, p1[1])

            pygame.draw.rect(screen, (40, 40, 40), (wall[0] - 10, wall[1] - 40, 20, 80))

            draw_spring(screen, wall, p1)
            pygame.draw.rect(screen, (0, 0, 0), (p1[0] - 10, p1[1] - 10, 20, 20))

            draw_spring(screen, p1, rim_center_px)

            draw_spring(screen, rim_center_px, p2)
            pygame.draw.rect(screen, (0, 0, 0), (p2[0] - 10, p2[1] - 10, 20, 20))
            return

        # ---------- PENDULUM CART ----------
        if "cart" in name and "cart" in extra:
            cx, cy = extra["cart"]
            cart_px = to_px(cx[i], -cy[i])

            pygame.draw.rect(screen, (0, 0, 0), (cart_px[0] - 20, cart_px[1] - 10, 40, 20))
            aa_line(screen, (0, 0, 0), cart_px, rim_center_px, 4)
            return

        # ---------- VERTICAL DOUBLE SPRING ----------
        if "vertical" in name and "bottom" in extra:
            b = extra["bottom"]
            bottom_px = to_px(0, -b[i])

            top = (rim_center_px[0], rim_center_px[1] - 150)

            pygame.draw.rect(screen, (40, 40, 40), (top[0] - 10, top[1] - 40, 20, 80))

            draw_spring(screen, top, rim_center_px)
            draw_spring(screen, rim_center_px, bottom_px)

            pygame.draw.rect(screen, (0, 0, 0), (bottom_px[0] - 10, bottom_px[1] - 10, 20, 20))
            return

    # --- draw ---
    def draw(self, screen):
        draw_vertical_gradient(screen, self.W, self.H, self.skytop, self.skybot)
        self.background.draw(screen)

        # court strip at bottom
        court_y = int(self.H * 0.78)
        pygame.draw.rect(screen, self.court, (0, court_y, self.W, self.H - court_y))
        pygame.draw.line(screen, self.courtline, (0, court_y), (self.W, court_y), 4)

        # pivot + rod
        pivot_px = self.world_to_screen(self.pend.pivot)
        ball_px = self.world_to_screen(self.ball.pos)
        aa_circle(screen, self.black, pivot_px, 6)
        aa_line(screen, (120, 120, 120), pivot_px, ball_px, width=5)

        # trail after release
        if not self.ball.attached and not self.calibrating:
            self.trail.append(ball_px)
            if len(self.trail) > self.TRAIL_MAX:
                self.trail.pop(0)
            for i in range(1, len(self.trail)):
                pygame.draw.line(screen, (140, 160, 200), self.trail[i - 1], self.trail[i], 3)

        # ball shadow
        shadow_strength = max(0.15, min(0.65, (ball_px[1] / max(self.H, 1))))
        shadow_w = int(self.m2px(self.ball.r) * (1.6 + 0.8 * shadow_strength))
        shadow_h = int(self.m2px(self.ball.r) * (0.55 + 0.3 * shadow_strength))
        shadow_surf = pygame.Surface((shadow_w * 2, shadow_h * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, int(120 * shadow_strength)), shadow_surf.get_rect())
        screen.blit(shadow_surf, (ball_px[0] - shadow_w, court_y - shadow_h))

        # draw ball
        aa_circle(screen, self.blue, ball_px, int(self.m2px(self.ball.r)))
        aa_circle(screen, (240, 240, 255), (ball_px[0] - 6, ball_px[1] - 6), max(2, int(self.m2px(self.ball.r) * 0.35)))

        # Hoop drawing via BasketAssembly + HoopSprite
        rim_px = self.world_to_screen(self.hoop_center_m)
        self.basket.update_from_physics(rim_px[0], rim_px[1], (self.W, self.H))
        rim_center_px = self.basket.get_anchors()["rim_center"]

        # draw physical rig (anchors + strings/springs) so hoop motion is understandable
        if not self.calibrating:
            self.draw_physics_system(screen, rim_center_px)
            draw_hoop_rig(screen, self.hoop_rig, rim_center_px, self.world_to_screen, self.font)

        if self.calibrating and self.cal_top_left is not None:
            tlx, tly = self.cal_top_left
            screen.blit(self.hoop_sprite.surface, (tlx, tly))
            pygame.draw.rect(screen, (40, 40, 40),
                             (tlx, tly, self.hoop_sprite.surface.get_width(), self.hoop_sprite.surface.get_height()), 2)
            draw_text(screen, self.bigfont, "CALIBRATION: click the RIM CENTER", 20, 20)
            draw_text(screen, self.font, f"Saving to {ANCHOR_JSON.name}", 20, 55)
            draw_text(screen, self.font, "ESC = cancel", 20, 75)
        else:
            self.hoop_sprite.draw(screen, rim_center_px, debug=False)

        # HUD
        draw_text(screen, self.font, f"FPS target={FPS} actual={self.clock.get_fps():.1f}", 20, 18)
        draw_text(screen, self.font, f"wind_ax={self.wind_ax:.2f} m/s^2", 20, 40)
        draw_text(screen, self.font, "SPACE=release   R=reset motion+ball   C=calibrate   ESC=menu", 20, 62)
        draw_text(screen, self.font, f"ball: x={self.ball.pos[0]:.2f} m  y={self.ball.pos[1]:.2f} m", 20, 84)
        draw_text(screen, self.font, f"hoop: x={self.hoop_center_m[0]:.2f} m  y={self.hoop_center_m[1]:.2f} m", 20, 106)
        draw_text(screen, self.font, f"motion model: {self.motion_name}", 20, 128)

        title = self.bigfont.render(f"Score: {self.score}   Level: {self.level}", True, (30, 30, 30))
        screen.blit(title, (self.W - title.get_width() - 20, 16))

        # overlays
        if self.im_green:
            elapsed = pygame.time.get_ticks() - self.flash_start_time
            alpha = max(255 - 255 * (elapsed / self.flash_duration), 0)
            image_copy = self.green_fn.copy()
            image_copy.set_alpha(int(alpha))
            screen.blit(image_copy, (0, 0))
            if alpha <= 0:
                self.im_green = False

        if self.im_moonshot:
            elapsed = pygame.time.get_ticks() - self.flash_start_time
            alpha = max(255 - 255 * (elapsed / self.flash_duration), 0)
            image_copy = self.curry_moonshot.copy()
            image_copy.set_alpha(int(alpha))
            screen.blit(image_copy, (0, 0))
            if alpha <= 0:
                self.im_moonshot = False


# =============================================================================
# Main loop (menu + game)
# =============================================================================
def main():
    pygame.init()

    info = pygame.display.Info()
    W = max(800, int(info.current_w * WINDOW_SCALE))
    H = max(600, int(info.current_h * WINDOW_SCALE))

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Pendulum Basketball (p_v3 + moving hoop + LEVEL motions @ 60 FPS)")

    menu = Menu(W, H)
    game = Game(W, H)

    state = "MENU"
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if state == "MENU":
                next_state = menu.handle_events(event)
                if next_state == "GAME":
                    state = "GAME"

            elif state == "GAME":
                next_state = game.handle_events(event)
                if next_state == "MENU":
                    state = "MENU"

        if not running:
            break

        if state == "MENU":
            menu.update(0.0)
            menu.draw(screen)
            # cap menu too
            game.clock.tick(FPS)

        else:
            frame_dt = game.clock.tick(FPS) / 1000.0
            game.update(frame_dt)
            game.draw(screen)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
