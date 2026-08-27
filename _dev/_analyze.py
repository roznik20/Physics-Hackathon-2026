"""Grounded pixel analysis of baseline screenshots (no vision model)."""
from PIL import Image
import os, numpy as np
d = "_baseline"
for f in sorted(os.listdir(d)):
    if not f.endswith(".png"):
        continue
    im = np.asarray(Image.open(os.path.join(d, f)).convert("RGB")).astype(float)
    h, w, _ = im.shape
    mean = im.mean()
    # sample regions
    def reg(name, x0, y0, x1, y1):
        a = im[int(y0*h):int(y1*h), int(x0*w):int(x1*w)]
        return round(float(a.mean()), 1)
    print(f"{f:14} {w}x{h} mean={mean:5.1f} "
          f"topleft={reg('tl',0,0,.15,.15)} center={reg('c',.45,.45,.55,.55)} "
          f"hoopzone={reg('hz',.55,.15,.85,.45)} ballzone={reg('bz',.05,.40,.25,.65)}")
