# Custom maps

A **run** is an ordered list of **levels**. Playing a run means advancing through
its levels in sequence — exactly the built‑in ladder, but authored by you. Runs
live in `maps/` as `<name>.json`. A file may hold a single level *or* a run
(`{"levels": [...]}`); both are accepted.

The **ball launcher** (the simple pendulum) is the same on every level — you only
place it once. Each level then chooses the **physics system that drives the
hoop** and where that hoop's motion is centered.

## Build one in the editor

`Map Gallery → + New map`. In the editor:

- **System** — pick the hoop's driving system (the 10 systems).
- **Drag the `LAUNCHER` marker** (the pendulum's pivot, upper‑left) and the
  `HOOP` marker (the hoop's base center, right) to place them. Positions are
  stored as screen fractions `0..1`.
- **Sliders** — `Amplitude` (0.2–2.2 m, how far the hoop swings), `Gravity`
  (1–20 m/s², affects the ball's arc), `Ball size` (0.05–0.30 m).
- **Test** — play the level right away. **Save** — write it to `maps/`. **Back** —
  return to the gallery.

A red `! <problem>` line under the panel lists any validation failure.

> The `HOOP` marker sets the **base center** of the hoop's motion; the game then
> clamps the *whole* apparatus (ceiling/floor/anchors included) so it stays on
> screen around that center.

## JSON schema

A single level:

```json
{
  "name": "my_level",
  "system": "double_pendulum",
  "launcher": [0.20, 0.16],
  "hoop":     [0.60, 0.66],
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
| `system` | str | hoop‑driving system id | see table below |
| `launcher` | [x, y] | pendulum‑pivot screen fraction | each `0..1` |
| `hoop` | [x, y] | hoop base‑center screen fraction | each `0..1` |
| `amp_m` | float | hoop peak displacement (m) | `0.05..3.0` |
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

- **Higher `amp_m`** → the hoop swings harder (a faster, wider‑moving target).
  Pair with a farther hoop for a hard challenge.
- **Low `gravity`** (e.g. 3) → floaty, long ball arcs; **high** (e.g. 18) → steep
  drops. Gravity only affects the ball's flight, not the hoop.
- Place the **hoop low and far right** with a high‑amplitude chaotic system (e.g.
  the double pendulum) for a hard mode; place it **near the launcher** for an
  easy warm‑up.
- A run of mixed systems is a mini‑campaign; `stationiary` is a fixed‑hoop
  baseline (the hoop doesn't move, so you rely purely on the release).

## Programmatic API

```python
from game.maps import MapLevel, save_run, load_run
from game.config import MAP_DIR

lv = MapLevel(name="hard", system="double_pendulum",
              launcher=(0.20, 0.16), hoop=(0.60, 0.66),
              amp_m=1.1, ball_radius_m=0.10, gravity=9.81)
save_run("my_run", [lv], MAP_DIR)          # writes maps/my_run.json
levels = load_run("my_run", MAP_DIR)       # -> [MapLevel]
```
