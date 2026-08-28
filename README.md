# LeHoop and Ball Game

A 2‑D basketball game powered by real Lagrangian / Newtonian mechanics. The
**ball hangs on a pendulum** you time; the **hoop rides a real physics system**
(pendulum, double pendulum, spring‑mass, cart, …) that swings it around as a
**moving target**. Release the ball at the right instant and its arc meets the
rim where it will be.

```
   pendulum launcher (left)                moving goal (right)
   ┌──╥──┐  ball released off the bob       ╔══╗  hoop rides the
   │   │  ──────── arc ────────────────▶    ╚══╝  driven node of a
   ╰───┴╯                                            physics system
```

Built from the **Physics Hackathon 2026** prototype. The math from the original
was kept (Lagrangian / Newton equations of motion, integrated with `solve_ivp`);
the game layer was rebuilt around it so that **every** system is represented
faithfully, the UI is a real application (menu, map gallery, editor, HUD), and
the original's signature elements — the *LeHoop* title, the LeBron backdrop, the
**green "fn"** score flash and the **moonshot** miss flash — are all restored.

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
| `SPACE` | release the ball off the pendulum at that instant (it departs with the pendulum's velocity) |
| `R` | re‑attach the ball (reset this shot, keep score/level) |
| `P` / `ESC` | pause / resume |
| `M` | back to the main menu |

### Scoring
Pass the ball through the rim. A score (green flash) advances the **run** to its
next level — the next physics system, with a larger motion. A miss (ball leaves
the field) triggers the moonshot flash and the ball re‑attaches.

---

## The ten moving goals

Each level uses a different system to drive the hoop. All are solved from their
actual equations of motion with `scipy.integrate.solve_ivp` (not hand‑tweened).
See [`docs/physics.md`](docs/physics.md) for the derivation of each.

| # | System (drives the hoop) | Module |
|---|--------------------------|--------|
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

The hoop's motion amplitude grows with level so later goals are harder to catch.

---

## Custom maps

Build your own runs from the **Map Gallery** (or by hand — each run is a small
JSON file in `maps/`). In the editor you pick the system, then **drag** the
`LAUNCHER` (pendulum pivot) and `HOOP` (hoop base center) markers to place them,
and set amplitude, gravity and ball size with the sliders. `Test` plays the level;
`Save` writes it to `maps/`.

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
    bodies.py            #   Pendulum (launcher), Ball, Hoop, HoopRig (hoop on the system)
    render.py            #   apparatus + hoop + pendulum + background drawing
    engine.py            #   Game: per‑run state, scoring, flashes, input, update/draw
    ui.py                #   HUD + button/panel helpers
    menu.py              #   Menu (LeHoop), MapGallery, MapEditor, About
    maps.py              #   MapLevel + run save/load + validation
basketball_sprites/      # original hoop sprite + art helpers (kept)
assets/                  # images (hoop, ball, bliss, lebron, green_fn, moonshot)
maps/                    # your saved runs (JSON)
legacy/                  # the original 1,372‑line main.py, kept for reference
docs/                    # physics derivations, architecture, map format
```

See [`docs/architecture.md`](docs/architecture.md) for how a frame flows from the
ODEs to pixels, and how to add a new system.
