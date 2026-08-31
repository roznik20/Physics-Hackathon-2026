"""Tiny sound layer: three short cues synthesized with numpy (no asset files).

  score — a bright ascending two-note "swish/chime" (made it!)
  rim   — a short metallic "clank" (banked off the rim)
  miss  — a low "thud" (moonshot)

Synthesized once at first use and cached. Silently does nothing if there is no
audio device (headless), so it is safe in tests. Call :func:`play` from the
engine hooks.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pygame

_SAMPLE_RATE = 22050
_cache: Dict[str, Optional[pygame.mixer.Sound]] = {}
_muted = False


def _envelope(n: int, attack: float = 0.01, release: float = 0.06) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    env = np.ones(n)
    a = max(1, int(attack * n))
    r = max(1, int(release * n))
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a)
    if r > 0:
        env[-r:] = np.linspace(1.0, 0.0, r)
    return env


def _to_sound(wave: np.ndarray) -> Optional[pygame.mixer.Sound]:
    # 16-bit signed mono
    arr = (np.clip(wave, -1.0, 1.0) * 32767).astype(np.int16)
    # pygame.mixer.init must have been called (mono, 16-bit) for sndarray
    return pygame.sndarray.make_sound(np.ascontiguousarray(arr))


def _score() -> np.ndarray:
    n = int(0.28 * _SAMPLE_RATE)
    t = np.linspace(0, 0.28, n)
    # two ascending notes (C5 -> G5) with a quick decay
    f1, f2 = 523.25, 783.99
    wave = 0.5 * np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
    wave *= np.exp(-t * 6.0)
    return wave * _envelope(n)


def _rim() -> np.ndarray:
    n = int(0.10 * _SAMPLE_RATE)
    t = np.linspace(0, 0.10, n)
    # a metallic clank: a high partial with fast decay
    wave = (np.sin(2 * np.pi * 2200 * t) + 0.4 * np.sin(2 * np.pi * 3400 * t))
    wave *= np.exp(-t * 40.0)
    return wave * 0.8 * _envelope(n, release=0.04)


def _miss() -> np.ndarray:
    n = int(0.16 * _SAMPLE_RATE)
    t = np.linspace(0, 0.16, n)
    # low thud
    wave = np.sin(2 * np.pi * 140 * t)
    wave *= np.exp(-t * 18.0)
    return wave * 0.9 * _envelope(n, release=0.08)


def _make(name: str) -> Optional[pygame.mixer.Sound]:
    if not pygame.mixer.get_init():
        return None
    wave = {"score": _score, "rim": _rim, "miss": _miss}[name]()
    return _to_sound(wave)


def play(name: str) -> None:
    """Play a cue by name (``score`` / ``rim`` / ``miss``). Safe if no audio."""
    global _muted
    if _muted or name not in _cache:
        if name not in _cache:
            try:
                _cache[name] = _make(name)
            except Exception:
                _cache[name] = None
    snd = _cache.get(name)
    if snd is not None:
        try:
            snd.play()
        except Exception:
            pass


def toggle_mute() -> bool:
    global _muted
    _muted = not _muted
    return _muted
