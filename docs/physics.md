# The physics

Every **goal** is a genuine mechanical system integrated from its **equations of
motion** with `scipy.integrate.solve_ivp` (`rtol=1e-8, atol=1e-10`). The solvers
return plain numpy arrays in a **y‑UP** frame with the system's natural origin at
`(0, 0)`; the game layer (see `architecture.md`) re‑centers, re‑scales, and
y‑flips them into the on‑screen world frame. No system is hand‑animated.

In the game the **ball** hangs on a separate simple pendulum (the launcher), and
the **hoop rides the driven (free) node** of the system described here. So each
system below is the *moving target* the ball must be timed against.

> Convention: `theta` is measured from the **downward vertical**, so `y = -L
> cos(theta) < 0` while a pendulum hangs down. `ω = theta'`.

Each module exposes `simulate(**params, t_max, fps)` and returns
`(t, *coordinate_arrays)`. The default parameters shown below are the ones the
game actually uses (from `physics/apparatus.py`).

---

## 1 · Horizontal spring–mass — `horizontal_spring.py`

A mass on a frictionless horizontal line, spring to a fixed wall (at `x = 1`).
Simple harmonic motion:

```
x'' = -(k/m) x
```

Default `x0=0.5, m=2, k=6`. Returns `(t, x_mass, x_wall)` (`x_wall` constant).
Rendered as a wall + zig‑zag spring + mass; the **hoop rides the mass**.

## 2 · Spring pendulum (elastic pendulum) — `spring_pendulum.py`

A mass on a **stretchable** rod that also swings (the classic coupled
radial/pendular system). Polar coordinates `r` (spring length), `θ` (swing):

```
r''   = r θ'² - (k/m)(r - L0) + g cos θ
θ''   = -(2 r' θ')/r - (g/r) sin θ
```

Default `m=1, k=20, L0=1, r0=1.1, θ0=1, r'=1, θ'=2`. Returns `(t, x, y)`.
The free end (mass) carries the **hoop**; the spring visibly stretches *and*
swings.

## 3 · Pendulum cart — `pendulum_cart.py`

A pendulum of length `r` mounted on a **free** cart of mass `m1` on a horizontal
rail; the bob has mass `m2`. No external force — momentum is exchanged between
cart and bob. State `θ, ω, x, v`:

```
θ'' = [ -m2 ω² sinθ cosθ - (g/r)(m1+m2) sinθ ] / [ m1 + m2 - m2 cos²θ ]
x'' = -(m2 r/(m1+m2)) [ θ'' cosθ - ω² sinθ ]
```

The game sets the initial cart velocity `v` so total horizontal momentum ≈ 0,
which keeps the free cart bounded (otherwise it drifts off forever). Returns
`(t, x_pend, y_pend, x_cart, y_cart)`. Rendered as a rail + wheeled cart + rod +
bob (the bob, not the cart, is the driven node the **hoop** rides).

## 4 · 3‑mass horizontal spring chain — `horizontal_three_pend.py`

`wall – s1 – m1 – s2 – m2 – s3 – m3`, a 3‑DOF coupled oscillator (undamped normal
modes). With spring extensions `e_i`:

```
F1 = -k1 e1 + k2 e2
F2 = -k2 e2 + k3 e3
F3 = -k3 e3
```

Default `m=2, k=15, L=0.10`, `x10=-0.05`. Returns `(t, x1, x2, x3, x_wall)`.
The **last** mass `m3` is the driven node and carries the **hoop**. Rendered as a
wall + three springs + three masses.

## 5 · Driven, damped spring — `damped_spring.py`

A wall oscillates sinusoidally (the driver) and drives a mass through a spring,
with linear damping. The mass obeys a **forced, damped** oscillator about the
moving wall `x_eq(t) = x0 + d sin(ω t)`:

```
x'' = -(b/m) x' - (k/m)(x - x_eq(t))
```

Default `b=0.1, ω=3, d=0.3, m=2, k=8`. Returns `(t, x_mass, x_wall)` with
`x_wall = x_eq` (so the wall is *drawn moving*). This is the one system whose
support is explicitly time‑driven — resonance/forced response is the whole idea.
The driven mass carries the **hoop**.

## 6 · Double pendulum (chaotic) — `double_pendulum.py`

The two‑link, non‑integrable, chaotic system. State `θ1, ω1, θ2, ω2` (angles from
downward vertical, `δ = θ2 - θ1`):

```
θ1'' = [ m2 L1 ω1² sinδ cosδ + m2 g sinθ2 cosδ + m2 L2 ω2² sinδ - (m1+m2) g sinθ1 ]
       / [ (m1+m2) L1 - m2 L1 cos²δ ]

θ2'' = [ -m2 L2 ω2² sinδ cosδ + (m1+m2)( g sinθ1 cosδ - L1 ω1² sinδ - g sinθ2 ) ]
       / [ (L2/L1) · ((m1+m2) L1 - m2 L1 cos²δ ) ]
```

Positions `x1 = L1 sinθ1, y1 = -L1 cosθ1; x2 = x1 + L2 sinθ2, y2 = y1 - L2 cosθ2`.
Default `m1=2, m2=1, L1=1.2, L2=1, θ1=1, ω1=-2, θ2=-2, ω2=1`. Returns
`(t, x1, y1, x2, y2)`. The second bob (free end) carries the **hoop**; both bobs
and the hatched ceiling pivot are drawn. Sensitive initial conditions ⇒ the goal
trajectory is genuinely chaotic (the hardest level to time).

## 7 · 2‑D springs — `springs_2d.py`

A mass held by an independent **horizontal** and **vertical** spring (two
decoupled 1D oscillators) so it moves in a 2‑D plane:

```
x'' = -(k_x/m) x
y'' = -(k_y/m) y
```

Default `x0=0.3, y0=0.6, m=2, k_x=4, k_y=6`. Returns `(t, x, y)`. Rendered with a
horizontal spring from a wall and a vertical spring from a ceiling meeting at the
mass — the **hoop** rides the mass (the only system where the goal moves in both
axes).

## 8 · Simple pendulum — `simple_pendulum.py`

Rigid rod, point mass, full (non‑small‑angle) dynamics:

```
θ'' = -(g/L) sin θ
```

Default `L=3, θ0=1, ω0=2`. Returns `(t, x, y)` with `x = L sinθ, y = -L cosθ`.
Drawn as a hatched ceiling pivot + rod + bob (the bob carries the **hoop**).

## 9 · Vertical 2‑mass spring stack — `vertical_double_spring.py`

`floor – s1 – m1 – s2 – m2 – s3 – ceiling`, a vertical 2‑DOF spring chain.
Positions `x1, x2` measured upward from the floor (`total_length = L1+L2+L3`):

```
e1 = x1 - L1
e2 = (x2 - x1) - L2
e3 = (total_length - x2) - L3
F1 = -k1 e1 + k2 e2 - m1 g
F2 = -k2 e2 + k3 e3 - m2 g
```

From these extensions, `x1` is the **lower** mass (spring 1 runs floor→m1) and
`x2` the **upper** mass (spring 3 runs ceiling→m2), so the physical stack is
`floor → m1 → m2 → ceiling`. Default `m1=1, m2=1.3, k1=15, k2=20, k3=20,
L1=0.7, L2=0.6, L3=0.4`. Returns `(t, x1, x2, total_length)`. The **lower** mass
`m1` is the driven node and carries the **hoop**. Rendered as
ceiling → spring → mass → spring → mass → floor, all on screen.

## 10 · Stationary — `stationary.py`

No drive: `(t, x, y)` both constant. The hoop is a plain fixed goal and the ball
starts at rest — a "just aim with your release velocity" baseline (useful for
maps). *(Filename keeps the original typo.)*

---

## How a raw sim becomes a moving goal

`physics/apparatus.py` holds one **builder** per system. Each:

1. reads the raw coordinate arrays from `simulate`,
2. re‑centers them to their mean,
3. names the nodes (`pivot`, `bob`, `cart`, `mass1..3`, `wall`, `ceiling`, …),
4. declares the **fixed** nodes (anchors), the **driven** node (the free end the
   **hoop** rides), and the **connectors** (`rod` / `spring` / `rail`),
5. returns a `MotionResult`.

`Motion` (`game/motion.py`) then re‑scales the whole apparatus so it fits on
screen, y‑flips to the world frame, and anchors it at the hoop's base center.
`HoopRig` (`game/bodies.py`) exposes the driven node's world position each frame
as the hoop's rim; `choose_hoop_base` places the whole envelope on screen. The
ball, meanwhile, is independent — it hangs on the simple pendulum launcher and is
released by the player.
