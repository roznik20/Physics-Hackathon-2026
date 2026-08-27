"""The game package: physics-driven launcher + fixed hoop + engine + rendering."""
from .config import C, PX_PER_M, FPS, G  # noqa: F401
from .bodies import Ball, Hoop, Launcher  # noqa: F401
from .motion import Motion, launcher_amp_for_level  # noqa: F401
from .engine import Game  # noqa: F401
