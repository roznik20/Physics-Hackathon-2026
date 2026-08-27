# Architecture

Three clean layers. The math never imports pygame; the UI never integrates an
ODE.

```
physics/          pure ODEs → MotionResult (named nodes + connectors)   [no pygame]
     │  simulate() → (t, x1, y1, …)
     ▼
physics/apparatus.py   builders: raw arrays → MotionResult
     │  MotionResult{points, fixed, driven, connectors, anchors}
     ▼
game/             pygame application
     motion.py   MotionResult → world‑frame Motion (scale, y‑flip, root)
     bodies.py   Ball, Launcher, Hoop
     render.py   faithful apparatus + hoop + background → pixels
     engine.py   Game: per‑run state, scoring, input, update/draw
     menu/ui     Menu, MapGallery, MapEditor, HUD, buttons
     maps.py     MapLevel + run save/load + validation
main.py           state machine: MENU ⇄ MAPS ⇄ NEWMAP ⇄ GAME ⇄ ABOUT
```

## Coordinates

- **Physics** (solvers): meters, **y‑UP**, origin at the system's natural anchor
  (pivot/wall/floor).
- **World** (game): meters, **y‑DOWN**, origin at the screen top‑left. Screen
  pixels = world meters × `PX_PER_M` (`220`), same axes.
- The conversion is the *only* place the y‑flip happens: `Motion` builds, for
  every node, `off = [ (x-mean)·scale, -(y-mean)·scale ]`.

## One frame, end to end

1. `Game.update(dt)` advances `Launcher.idx` by one sim step (the apparatus is
   pre‑computed, so the "simulation" is table‑lookup, not per‑frame ODE).
2. A fixed‑step accumulator (60 Hz) steps the **ball** if it's in flight:
   `vel += g·dt; pos += vel·dt`, and appends to the trail.
3. Scoring: if the ball's position is within the rim radius **and** in the rim's
   vertical band, `score++`, `level++`, `_build_level()`.
4. Miss/OOB: if the ball leaves the field, it re‑attaches to the launcher.
5. `Game.draw()` → `render.draw_background`, `draw_hoop` (calibrated sprite),
   `draw_apparatus` (nodes + connectors for the current `Motion`), `draw_ball`,
   then `ui.draw_hud` (system label, score/level, controls, pause veil).

## The launcher mechanic (the core idea)

The ball hangs on the **driven node** (the free end of the system). Each frame it
sits at `Motion.driven_world(idx, root)`. On `SPACE`, `release_from` copies the
node's **position and velocity** into the ball and detaches it. So the ball
departs exactly as the physical apparatus would throw it — release a pendulum
bob, a spring‑mass tip, a cart‑mounted bob, a 2‑D spring mass, etc. This is what
makes *every* system a meaningful launcher, not just the pendulums.

## Runs and levels

- A **level** is a `MapLevel` (`game/maps.py`): `system`, `launcher` & `hoop`
  screen fractions, `amp_m`, `ball_radius_m`, `gravity`.
- A **run** is an ordered `list[MapLevel]`. `Game` plays them in sequence,
  wrapping. The default `builtin_run()` is the 10‑system ladder with increasing
  amplitude. A saved map in `maps/` is just a run of one or more levels.

## Adding a new system

1. `physics/<name>.py`: a `simulate(...)` returning `(t, *coords)` in y‑up
   meters (copy an existing file as a template).
2. `physics/apparatus.py`:
   - add a `build_<name>(out) -> MotionResult` that names the nodes, sets
     `fixed`, `driven` (the free end the ball rides), and `connectors`.
   - add a `System(...)` entry to `SYSTEMS`.
3. `game/render.py`: if it needs a node style not already in `_NODE_STYLE`
   (`ball`, `mass`, `cart`, `wall`, `ceiling`, `floor`, `joint`), add it to
   `_draw_node`.
4. It appears in the menu ladder and the map editor automatically.

Verify with `python _test_physics.py` (builds every `MotionResult` and checks
finiteness, node counts, and connector validity).

## Files I changed vs. the original hackathon

- **`main.py`** — replaced (the 1,372‑line monolith is preserved in
  `legacy/main_original.py`; the new entry point is ~130 lines).
- **`physics/*.py`** — rewritten to drop matplotlib/sympy from the hot path and
  expose a uniform `simulate()` contract; equations unchanged.
- **`basketball_sprites/*.py`** — small path‑robustness fixes only (load `assets/`
  from the repo root; `max(1, …)` on a zero‑radius circle; a try/except import
  for direct‑script runs). No art or logic changed.
- **`game/`, `physics/common.py`, `physics/apparatus.py`, `docs/`, `maps/`** — new.
