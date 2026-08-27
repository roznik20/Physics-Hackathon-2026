"""Custom maps: data model + save/load.

A *level* is a complete description of one playable round: which physics system
drives the launcher, where the launcher and hoop sit (as screen fractions), the
launcher amplitude, ball radius, and gravity. A *run* is an ordered list of
levels — playing a run means advancing through its levels in sequence, exactly
like the built-in level ladder but authored by the player.

Maps live in ``maps/`` as ``<name>.json``. A file may hold a single level or a
run ({"levels": [...]}). See docs/maps.md for the full schema.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

from physics.apparatus import SYSTEMS, system_by_id


@dataclass
class MapLevel:
    name: str = "level"
    system: str = "simple_pendulum"
    launcher: Tuple[float, float] = (0.22, 0.46)   # screen fraction (x, y)
    hoop: Tuple[float, float] = (0.68, 0.44)        # screen fraction (x, y)
    amp_m: float = 0.6                              # launcher peak displacement (m)
    ball_radius_m: float = 0.12
    gravity: float = 9.81

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "MapLevel":
        def _frac(v, default):
            try:
                return (float(v[0]), float(v[1]))
            except Exception:
                return default
        return MapLevel(
            name=str(d.get("name", "level")),
            system=str(d.get("system", "simple_pendulum")),
            launcher=_frac(d.get("launcher"), (0.22, 0.46)),
            hoop=_frac(d.get("hoop"), (0.68, 0.44)),
            amp_m=float(d.get("amp_m", 0.6)),
            ball_radius_m=float(d.get("ball_radius_m", 0.12)),
            gravity=float(d.get("gravity", 9.81)),
        )


def validate(cfg: MapLevel) -> List[str]:
    """Return a list of problems (empty if the level is valid)."""
    problems = []
    if cfg.system not in {s.id for s in SYSTEMS}:
        problems.append(f"unknown system {cfg.system!r}")
    for label, (fx, fy) in (("launcher", cfg.launcher), ("hoop", cfg.hoop)):
        if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
            problems.append(f"{label} fraction out of [0,1]: {cfg.launcher}")
    if not (0.05 <= cfg.amp_m <= 3.0):
        problems.append(f"amp_m out of range: {cfg.amp_m}")
    if not (0.03 <= cfg.ball_radius_m <= 0.4):
        problems.append(f"ball_radius_m out of range: {cfg.ball_radius_m}")
    if not (1.0 <= cfg.gravity <= 25.0):
        problems.append(f"gravity out of range: {cfg.gravity}")
    return problems


# ---------------------------------------------------------------------------
# Run (list of levels) save / load
# ---------------------------------------------------------------------------

def list_maps(map_dir: Path) -> List[str]:
    if not map_dir.exists():
        return []
    return sorted(p.stem for p in map_dir.glob("*.json"))


def save_run(name: str, levels: List[MapLevel], map_dir: Path) -> Path:
    map_dir.mkdir(parents=True, exist_ok=True)
    path = map_dir / f"{name}.json"
    path.write_text(json.dumps({"name": name, "levels": [lv.to_dict() for lv in levels]},
                               indent=2), encoding="utf-8")
    return path


def load_run(name: str, map_dir: Path) -> List[MapLevel]:
    path = map_dir / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "levels" in data:
        raw = data["levels"]
    else:  # a single level
        raw = [data]
    return [MapLevel.from_dict(d) for d in raw]


def load_file(path: Path) -> List[MapLevel]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data["levels"] if (isinstance(data, dict) and "levels" in data) else [data]
    return [MapLevel.from_dict(d) for d in raw]
