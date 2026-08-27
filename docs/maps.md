# Custom maps

A **run** is an ordered list of **levels**. Playing a run means advancing through
its levels in sequence — exactly the built‑in ladder, but authored by you. Runs
live in `maps/` as `<name>.json`. A file may hold a single level *or* a run
(`{"levels": [...]}`); both are accepted.

## Build one in the editor

`Map Gallery → + New map`. In the editor:

- **System** — pick the launcher (the 10 systems).
- **Drag the `LAUNCHER` and `HOOP` markers** in the right panel to place them
  (positions are stored as screen fractions `0..1`).
- **Sliders** — `Amplitude` (0.2–2.2 m, how hard the launcher throws),
  `Gravity` (1–20 m/s²), `Ball size` (0.05–0.30 m).
- **Test** — play the level right away. **Save** — write it to `maps/`. **Back** —
  return to the gallery.

A red `! <problem>` line under the panel lists any validation failure.

## JSON schema

A single level:

```json
{
  "name": "my_level",
  "system": "double_pendulum",
  "launcher": [0.22, 0.46],
  "hoop":     [0.68, 0.44],
  "amp_m": 0.8,
  "ball_radius_m": 0.12,
  "gravity": 9.81
}
```

A run (multiple levels, played in order):

```json
{
  "name": "my_run",
  "levels": [
    { "name": "warmup", "system": "simple_pendulum", "amp_m": 0.6 },
    { "name": "chaos",  "system": "double_pendulum", "amp_m": 1.0 }
  ]
}
```

### Fields

| Field | Type | Meaning | Range |
|-------|------|---------|-------|
| `name` | str | display name | any |
| `system` | str | launcher system id | see table below |
| `launcher` | [x, y] | launcher screen fraction | each `0..1` |
| `hoop` | [x, y] | hoop screen fraction | each `0..1` |
| `amp_m` | float | launcher peak displacement (m) | `0.05..3.0` |
| `ball_radius_m` | float | ball radius (m) | `0.03..0.4` |
| `gravity` | float | gravity (m/s²) | `1..25` |

Missing fields fall back to defaults (a bare `{"system": "double_pendulum"}` is
valid). `launcher`/`hoop` are **fractions of the window**, so a map works at any
window size.

### Valid `system` values

`simple_pendulum`, `double_pendulum`, `spring_pendulum`, `pendulum_cart`,
`horizontal_spring`, `horizontal_three_pend`, `damped_spring`, `springs_2d`,
`verticle_double_spring`, `stationiary`.

## Tips for interesting levels

- **Higher `amp_m`** → the launcher throws harder; pair with a farther/higher hoop.
- **Low `gravity`** (e.g. 3) → floaty, long arcs; **high** (e.g. 18) → steep drops.
- Place the **hoop low and far right** with a high‑amplitude chaotic system for a
  hard mode; place it **near the launcher** for an easy warm‑up.
- A run of mixed systems is a mini‑campaign; `stationiary` is a fixed‑hoop
  baseline (the ball starts at rest, so you rely on the release velocity).

## Programmatic API

```python
from game.maps import MapLevel, save_run, load_run
from game.config import MAP_DIR
import pathlib

lv = MapLevel(name="hard", system="double_pendulum",
              launcher=(0.20, 0.50), hoop=(0.80, 0.35),
              amp_m=1.1, ball_radius_m=0.10, gravity=9.81)
save_run("my_run", [lv], MAP_DIR)          # writes maps/my_run.json
levels = load_run("my_run", MAP_DIR)       # -> [MapLevel]
```
