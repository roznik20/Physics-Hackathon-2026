# Physics Hoop

A 2‑D basketball game where the ball is held on the **free end of a real
Lagrangian system** and you time your release to fling it through a fixed goal.
The physics is the point: every level is driven by a different genuine
many‑body system solved from its equations of motion (not hand‑tweened motion
paths).

```
            launcher (a Lagrangian system)          fixed, mounted goal
   ceiling/  ┌───────────────┐  release    ┌───────────────┐
   wall ──── │  ...─●─...─●─🏀 │ ──────────▶ │  ╔═╗          │
             └───────────────┘   velocity   │  ╚═╝  rim     │
                                            └───────────────┘
```

Built from the **Physics Hackathon 2026** prototype. The math from the original
was kept (Lagrangian / Newton‑equation ODEs); the game layer was rebuilt around
it so that **every** system is represented faithfully and the UI is a real
application (menu, maps, editor, HUD).

---

## Run it

```bash
cd Physics-Hackathon-2026
python main.py
```

Requirements: Python 3.9+ with `pygame`, `numpy`, `scipy` (see `requirements.txt`).

### Controls
| Key | Action |
|-----|--------|
| `SPACE` | release the ball at the current instant (it departs with the apparatus's velocity) |
| `R` | re‑attach the ball (reset this shot, keep score/level) |
| `P` / `ESC` | pause / resume |
| `M` | back to the main menu |

### Scoring
Pass the ball through the rim. A score advances the **run** to its next level
(the next physics system). Missed balls auto‑re‑attach after they leave the
field.

---

## The ten launcher systems

Each level uses a different system as the launcher. All are solved with
`scipy.integrate.solve_ivp` from their actual equations of motion — see
[`docs/physics.md`](docs/physics.md) for the derivation of each.

| # | System | Module |
|---|--------|--------|
| 1 | Horizontal spring–mass | `physics/horizontal_spring.py` |
| 2 | Spring pendulum (elastic pendulum) | `physics/spring_pendulum.py` |
| 3 | Pendulum cart (coupled, free cart) | `physics/pendulum_cart.py` |
| 4 | 3‑mass horizontal spring chain | `physics/horizontal_three_pend.py` |
| 5 | Driven, damped spring | `physics/damped_spring.py` |
| 6 | Double pendulum (chaotic) | `physics/double_pendulum.py` |
| 7 | 2‑D springs (independent x & y) | `physics/springs_2d.py` |
| 8 | Simple pendulum | `physics/simple_pendulum.py` |
| 9 | Vertical 2‑mass spring stack | `physics/verticle_double_spring.py` |
| 10 | Stationary (fixed hoop, no drive) | `physics/stationiary.py` |

The launcher amplitude grows with level so later systems throw harder.

---

## Custom maps

Build your own runs from the **Map Gallery** (or by hand — each run is a small
JSON file in `maps/`). In the editor you pick the system, then **drag** the
`LAUNCHER` and `HOOP` markers to place them and set amplitude, gravity and ball
size with the sliders. `Test` plays the level; `Save` writes it to `maps/`.

The full schema is in [`docs/maps.md`](docs/maps.md).

---

## Project layout

```
main.py                  # entry point: state machine (menu → game → maps → editor)
physics/                 # the math. Pure simulators, no pygame.
    common.py            #   MotionResult / System dataclasses
    apparatus.py         #   interpreters: raw sim output → named nodes + connectors
    <system>.py          #   one Lagrangian/Newton system each (10 systems)
game/                    # the game layer (uses pygame)
    config.py            #   constants, palette, sizing
    motion.py            #   MotionResult → world‑frame Motion (scale/position)
    bodies.py            #   Ball, Launcher, Hoop
    render.py            #   faithful apparatus + hoop + background drawing
    engine.py            #   Game: per‑run state, scoring, input, update/draw
    ui.py                #   HUD + button/panel helpers
    menu.py              #   Menu, MapGallery, MapEditor, About
    maps.py              #   MapLevel + run save/load + validation
basketball_sprites/      # original hoop sprite + art helpers (kept)
assets/                  # images (hoop, ball, backgrounds)
maps/                    # your saved runs (JSON)
legacy/                  # the original 1,372‑line main.py, kept for reference
docs/                    # physics derivations, architecture, map format
```

See [`docs/architecture.md`](docs/architecture.md) for how a frame flows from the
ODEs to pixels, and how to add a new system.
