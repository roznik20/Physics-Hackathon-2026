"""Headless test of the physics layer: run every system, sanity-check the MotionResult."""
import os, traceback
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
from physics.apparatus import SYSTEMS, run

ok = True
for s in SYSTEMS:
    try:
        mr = run(s, t_max=30.0, fps=60)
        # checks
        n = len(mr.t)
        assert n > 10, f"too few frames {n}"
        # every node finite
        for name, arr in mr.points.items():
            assert np.all(np.isfinite(arr)), f"{s.id} node {name} has non-finite"
            assert arr.shape[1] == 2, f"{s.id} node {name} not Nx2: {arr.shape}"
        # connectors reference valid nodes
        for a, b, kind in mr.connectors:
            assert a in mr.points and b in mr.points, f"{s.id} bad connector {a}->{b}"
        assert mr.driven in mr.points, f"{s.id} driven {mr.driven} missing"
        for name in mr.fixed:
            assert name in mr.points, f"{s.id} fixed {name} missing"
        for name, kind in mr.anchors:
            if name in ("rail",):  # rail anchor is conceptual, endpoints are nodes
                continue
            assert name in mr.points, f"{s.id} anchor {name} missing"
        env = mr.envelope()
        # driven node should have real motion (not all zeros) unless stationary
        drv = mr.points[mr.driven]
        c = drv - np.mean(drv, axis=0)
        amp = float(np.max(np.hypot(c[:, 0], c[:, 1])))
        status = "OK" if amp > 1e-4 or s.id == "stationary" else "FLAT"
        if status != "OK":
            ok = False
        print(f"[{status}] {s.id:24} n={n:5d} nodes={list(mr.points)} "
              f"env=({env[0]:+.2f},{env[1]:+.2f})-({env[2]:+.2f},{env[3]:+.2f}) "
              f"amp={amp:+.3f} conns={len(mr.connectors)}")
    except Exception:
        ok = False
        print(f"[ERR] {s.id}")
        traceback.print_exc()

print("\nALL OK" if ok else "\nPROBLEMS FOUND")
