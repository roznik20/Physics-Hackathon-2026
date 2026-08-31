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
    bodies.py   Pendulum (the launcher), Ball, Hoop, HoopRig (hoop on the system)
    render.py   apparatus + hoop + pendulum + background → pixels
    engine.py   Game: per‑run state, scoring, flashes, input, update/draw
    menu/ui     Menu (LeHoop), MapGallery, MapEditor, HUD, buttons
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

1. `Game.update(dt)` advances the **pendulum launcher** (closed‑form θ(t), no
   ODE) and the **hoop rig** (`HoopRig.idx` → one pre‑computed `Motion` step; the
   apparatus is a table, not a per‑frame ODE).
2. A fixed‑step accumulator (60 Hz) steps the **ball** if it's in flight:
   `vel += g·dt; pos += vel·dt`, and appends to the trail.
3. Scoring: if the ball is within the rim radius **and** in the rim's vertical
   band, `score++`, the **green `fn`** flash fires, and the run advances to the
   next level (`_build_level()`).
4. Miss/OOB: if the ball leaves the field, the **moonshot** flash fires and the
   ball re‑attaches to the pendulum.
5. `Game.draw()` → `render.draw_background`, `draw_pendulum` (launcher + ball),
   `draw_apparatus` (nodes + connectors of the `HoopRig`, with the driven node
   drawn as the hoop), `draw_trail`, `draw_ball`, then `ui.draw_hud`.

## The core mechanic

Two moving things, one release:

- **The ball hangs on a simple pendulum** on the left (`Pendulum`). Its angle
  is the analytic solution `θ(t)=θ₀cos(ωt+φ)`, so the release *timing* is the
  only thing the player controls. On `SPACE` the ball departs with the pendulum
  bob's position **and velocity** (the derivative of the analytic solution).
- **The hoop rides the driven node** of a real Lagrangian/Newton system on the
  right (`HoopRig` wraps a pre‑computed `Motion`; the hoop's world position is
  `root + off[driven, idx]`). As the system swings, the hoop moves — the target
  is where the ball's arc meets it.

So a level is *winnable by timing the release so the ball's arc crosses the rim
where the hoop will be*. Every one of the ten systems is used (none reduced to
"just a pendulum"), and each has a different motion to time against.

## Runs and levels

- A **level** is a `MapLevel` (`game/maps.py`): `system`, `launcher` & `hoop`
  screen fractions (pendulum pivot / hoop base center), `amp_m`,
  `ball_radius_m`, `gravity`.
- A **run** is an ordered `list[MapLevel]`. `Game` plays them in sequence,
  wrapping. The default `builtin_run()` is the 10‑system ladder with increasing
  amplitude. A saved map in `maps/` is just a run of one or more levels.

## Adding a new system

1. `physics/<name>.py`: a `simulate(...)` returning `(t, *coords)` in y‑up
   meters (copy an existing file as a template).
2. `physics/apparatus.py`:
   - add a `build_<name>(out) -> MotionResult` that names the nodes, sets
     `fixed`, `driven` (the free end the **hoop** rides), and `connectors`.
   - add a `System(...)` entry to `SYSTEMS`.
3. `game/render.py`: if it needs a node style not already in `_NODE_STYLE`
   (`ball`, `mass`, `cart`, `wall`, `ceiling`, `floor`, `joint`), add it to
   `_draw_node`.
4. It appears in the menu ladder and the map editor automatically.

## Verification

The three headless checks below are the project's test suite (they use
`SDL_VIDEODRIVER=dummy`, so no display is needed):

- `python _dev/_test_physics.py` — builds every `MotionResult` and checks
  finiteness, node counts, and connector validity.
- `python _dev/_autopilot.py` — drives the real `Game` through all 10 levels by
  timing the release; prints how many release attempts each level took.
- `python _dev/_smoke.py` — menu → play → gallery → editor → about, saving
  screenshots to `_smoke/`.

Run them after changing any of `physics/`, `game/`, or `main.py`.

## Files I changed vs. the original hackathon

- **`main.py`** — replaced (the 1,372‑line monolith is preserved in
  `legacy/main_original.py`; the new entry point is ~130 lines).
- **`physics/*.py`** — rewritten to drop matplotlib/sympy from the hot path and
  expose a uniform `simulate()` contract; equations unchanged.
- **`basketball_sprites/*.py`** — small path‑robustness fixes only (load `assets/`
  from the repo root; `max(1, …)` on a zero‑radius circle; a try/except import
  for direct‑script runs). No art or logic changed.
- **`game/`, `physics/common.py`, `physics/apparatus.py`, `docs/`, `maps/`** — new.
- **Restored from the original:** the *LeHoop and Ball Game* title, the LeBron
  backdrop, the green `fn` score flash, and the `curry_moonshot` miss flash.
