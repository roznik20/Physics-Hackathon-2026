"""Rendering. Procedural background + faithful apparatus + hoop + ball.

All functions take the game's screen size (W, H) and draw in screen pixels.
World->screen is simply ``pos_m * PX_PER_M`` (origin top-left, y-down).
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pygame

from .config import C, PX_PER_M

# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _px(v: float) -> int:
    return int(round(v))


def world_to_screen(pos_m: Tuple[float, float], px_per_m: float = PX_PER_M) -> Tuple[int, int]:
    return (_px(pos_m[0] * px_per_m), _px(pos_m[1] * px_per_m))


def _aa_circle(surf, color, cx, cy, r):
    if r < 1:
        pygame.draw.circle(surf, color, (int(cx), int(cy)), max(1, int(r)))
        return
    try:
        import pygame.gfxdraw as gfx
        gfx.filled_circle(surf, int(cx), int(cy), int(r), color)
    except Exception:
        pygame.draw.circle(surf, color, (int(cx), int(cy)), int(r))


def _line(surf, color, a, b, width=2):
    if width <= 1:
        pygame.draw.aaline(surf, color, a, b)
    else:
        pygame.draw.line(surf, color, a, b, width)


def _spring(surf, color, p1, p2, width=3, amp=8):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 4:
        _line(surf, color, p1, p2, width)
        return
    ux, uy = dx / dist, dy / dist
    nx, ny = -uy, ux
    # Physical spring: constant coil density (coils ∝ length) and coil width
    # shrinks when compressed (a real helical spring gets narrower, not
    # self-intersecting, when squeezed).
    coils = max(3, int(dist / 14))
    amp_actual = amp * max(0.35, min(1.0, dist / 100.0))
    end_pad = 0.10 * dist
    start, end = end_pad, dist - end_pad
    pts = [(x1, y1)]
    if end <= start:
        pts.append((x2, y2))
    else:
        for i in range(1, coils + 1):
            t = start + (end - start) * (i / (coils + 1))
            sgn = -1 if (i % 2 == 0) else 1
            pts.append((x1 + ux * t + nx * amp_actual * sgn,
                        y1 + uy * t + ny * amp_actual * sgn))
        pts.append((x2, y2))
    pygame.draw.lines(surf, color, False, [(int(a), int(b)) for a, b in pts], width)


def _hatch(surf, color, cx, cy, half, vertical=True, depth=12):
    """A hatched fixed support (wall / ceiling / floor)."""
    if vertical:
        rect = (cx - 4, cy - half, 8, half * 2)
        pygame.draw.rect(surf, color, rect)
        for hy in range(-half + 6, half, 12):
            pygame.draw.line(surf, color, (cx, cy + hy), (cx + depth, cy + hy - 10), 2)
    else:
        rect = (cx - half, cy - 4, half * 2, 8)
        pygame.draw.rect(surf, color, rect)
        for hx in range(-half + 6, half, 12):
            pygame.draw.line(surf, color, (cx + hx, cy), (cx + hx - 10, cy + depth), 2)


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------

def draw_background(screen, W, H, t=0.0, court_y_frac=0.78):
    # vertical sky gradient
    top, bot = C["sky_top"], C["sky_bot"]
    for y in range(H):
        f = y / max(1, H - 1)
        col = (int(top[0] + (bot[0] - top[0]) * f),
               int(top[1] + (bot[1] - top[1]) * f),
               int(top[2] + (bot[2] - top[2]) * f))
        pygame.draw.line(screen, col, (0, y), (W, y))
    # a couple of soft cloud bands
    for k, (cy, cr, spd) in enumerate([(0.22, 70, 4), (0.34, 50, 6), (0.5, 40, 8)]):
        cx = ((t * spd) % (W + 300)) - 150
        cloud = (255, 255, 255)
        for dx in (-cr, 0, cr):
            _aa_circle(screen, cloud, cx + dx, cy * H, cr * 0.7)
    # court floor
    court_y = int(H * court_y_frac)
    pygame.draw.rect(screen, C["court"], (0, court_y, W, H - court_y))
    pygame.draw.line(screen, C["court_line"], (0, court_y), (W, court_y), 3)
    # a faint center-court arc on the floor
    pygame.draw.arc(screen, C["court_line"],
                    (W * 0.5 - 160, court_y + 40, 320, 120), math.pi, 2 * math.pi, 2)


# ---------------------------------------------------------------------------
# Apparatus (the launcher)
# ---------------------------------------------------------------------------

_NODE_STYLE = {
    "bob": "ball", "mid": "mass", "cart": "cart", "pivot": "ceiling",
    "mass": "mass", "mass1": "mass", "mass2": "mass", "mass3": "mass",
    "wall": "wall", "h_wall": "wall", "v_wall": "ceiling",
    "ceiling": "ceiling", "floor": "floor", "rail_l": "wall", "rail_r": "wall",
}


def _node_style(name: str) -> str:
    return _NODE_STYLE.get(name, "mass")


def draw_apparatus(screen, rig, px_per_m: float = PX_PER_M,
                   ball_attached: bool = True, driven_is_hoop: bool = False) -> Dict[str, Tuple[int, int]]:
    """Draw every node and connector of the physics apparatus in screen coords.

    ``rig`` exposes ``.motion``, ``.idx`` and ``.root`` (the launcher *or* the
    hoop rig). When ``driven_is_hoop`` the driven (free) node is the hoop rim, so
    no ghost/ball marker is drawn there (the hoop sprite is drawn separately on
    top). Returns a dict of node-name -> screen-px.
    """
    motion = rig.motion
    i = int(np.clip(rig.idx, 0, motion.n - 1))
    root = rig.root
    px = px_per_m

    screen_nodes: Dict[str, Tuple[int, int]] = {}
    for name in motion.mr.points:
        off = motion.off[name][i]
        wp = (root[0] + off[0], root[1] + off[1])
        screen_nodes[name] = world_to_screen(wp, px)

    # connectors first (behind nodes)
    for a, b, kind in motion.connectors:
        if a in screen_nodes and b in screen_nodes:
            pa, pb = screen_nodes[a], screen_nodes[b]
            if kind == "spring":
                _spring(screen, C["spring"], pa, pb)
            else:  # rod
                _line(screen, C["rod"], pa, pb, width=6)

    # rail (pendulum cart): a horizontal track between the two rail nodes
    if "rail_l" in screen_nodes and "rail_r" in screen_nodes:
        a, b = screen_nodes["rail_l"], screen_nodes["rail_r"]
        pygame.draw.line(screen, C["anchor"], a, b, 5)
        pygame.draw.line(screen, C["panel_edge"], a, b, 1)

    # nodes
    for name in motion.mr.points:
        style = _node_style(name)
        p = screen_nodes[name]
        if style == "ball":
            is_driven = (name == motion.mr.driven)
            if driven_is_hoop and is_driven:
                continue  # the hoop sprite is drawn on top of this node
            if not ball_attached:
                # ghost ring at the free end while the ball is in flight
                pygame.draw.circle(screen, C["accent"], p, 9, 2)
        elif style == "joint":
            _aa_circle(screen, C["mass"], p[0], p[1], 7)
            _aa_circle(screen, C["panel"], p[0], p[1], 3)
        elif style == "mass":
            _aa_circle(screen, C["mass"], p[0], p[1], 11)
            _aa_circle(screen, (150, 154, 170), p[0] - 3, p[1] - 3, 3)
        elif style == "wall":
            _hatch(screen, C["anchor"], p[0], p[1], 34, vertical=True)
        elif style == "ceiling":
            _hatch(screen, C["anchor"], p[0], p[1], 44, vertical=False)
        elif style == "floor":
            _hatch(screen, C["anchor"], p[0], p[1], 44, vertical=False)
        elif style == "cart":
            _draw_cart(screen, p)

    return screen_nodes


def _draw_cart(screen, p):
    w, h = 46, 20
    x, y = p
    body = pygame.Rect(x - w // 2, y - h // 2, w, h)
    pygame.draw.rect(screen, C["mass"], body, border_radius=5)
    pygame.draw.rect(screen, (150, 154, 170), body, 2, border_radius=5)
    for wx in (-13, 13):
        _aa_circle(screen, C["anchor"], x + wx, y + h // 2 + 4, 6)
        _aa_circle(screen, (150, 154, 170), x + wx, y + h // 2 + 4, 2)


# ---------------------------------------------------------------------------
# Hoop
# ---------------------------------------------------------------------------

def draw_hoop(screen, hoop, sprite, px_per_m: float = PX_PER_M, court_y_frac=0.78, H=800):
    """Draw the fixed hoop sprite so its rim (the calibrated anchor) sits exactly
    on hoop.c, with the pole reaching the court floor."""
    px = px_per_m
    cx, cy = world_to_screen(tuple(hoop.c), px)
    surf = sprite.surface
    ax, ay = sprite.rim_anchor.x, sprite.rim_anchor.y
    top_left = (cx - ax, cy - ay)
    screen.blit(surf, top_left)


# ---------------------------------------------------------------------------
# Ball
# ---------------------------------------------------------------------------

def draw_ball(screen, ball, ball_img, px_per_m: float = PX_PER_M, court_y_frac=0.78, H=800):
    px = px_per_m
    bx, by = world_to_screen(tuple(ball.pos), px)
    r = _px(ball.r * px)
    # shadow on the court
    court_y = int(H * court_y_frac)
    if by < court_y:
        f = max(0.15, min(0.7, by / max(1, court_y)))
        sw = int(r * (1.4 + 0.8 * f))
        sh = int(r * 0.5)
        sh_surf = pygame.Surface((sw * 2, sh * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(sh_surf, (0, 0, 0, int(110 * f)), sh_surf.get_rect())
        screen.blit(sh_surf, (bx - sw, court_y - sh))
    screen.blit(ball_img, (bx - r, by - r))
    return (bx, by)


def draw_trail(screen, trail: List[Tuple[int, int]]):
    for idx in range(1, len(trail)):
        a, b = trail[idx - 1], trail[idx]
        f = idx / max(1, len(trail))
        col = (int(150 + 90 * f), int(120 + 40 * f), int(120))
        pygame.draw.line(screen, col, a, b, max(1, int(3 * f)))


def draw_pendulum(screen, pend, px_per_m: float = PX_PER_M,
                  ball_attached: bool = True, court_y_frac=0.78, H=800):
    """The launcher: a small hatched ceiling mount + rod to the bob.

    The bob is the ball (drawn separately) when attached; otherwise a ghost ring
    marks the release point.
    """
    px = px_per_m
    pivot = world_to_screen(tuple(pend.pivot), px)
    bob = world_to_screen(tuple(pend.bob_pos()), px)

    # ceiling mount above the pivot
    _hatch(screen, C["anchor"], pivot[0], pivot[1] - 12, 34, vertical=False)
    # rod
    _line(screen, C["rod"], pivot, bob, width=6)
    # pivot hub
    _aa_circle(screen, C["mass"], pivot[0], pivot[1], 6)
    _aa_circle(screen, C["panel"], pivot[0], pivot[1], 2)

    if not ball_attached:
        # ghost release point
        pygame.draw.circle(screen, C["accent"], bob, 9, 2)


def draw_scene(screen, hoop_rig, hoop, pend, ball, ball_img, hoop_sprite,
               px, W, H, time=0.0, court_y_frac=0.78):
    """Render the full game scene (everything the game draws except the HUD,
    the score/miss flashes and the flight trail) into ``screen``. Shared by the
    engine and the map editor's WYSIWYG preview so the preview is exactly what
    the game will show.
    """
    draw_background(screen, W, H, time, court_y_frac)
    # the physics-system apparatus + the hoop on its driven node
    draw_apparatus(screen, hoop_rig, px, ball_attached=False, driven_is_hoop=True)
    draw_hoop(screen, hoop, hoop_sprite, px, court_y_frac, H)
    # the pendulum launcher (left)
    draw_pendulum(screen, pend, px, ball.attached, court_y_frac, H)
    # the ball (attached or in flight)
    draw_ball(screen, ball, ball_img, px, court_y_frac, H)
