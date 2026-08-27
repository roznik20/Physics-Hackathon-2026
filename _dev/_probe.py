"""Temporary environment probe. Prints versions, asset info, and hoop-anchor sanity."""
import sys, os
print("python", sys.version.split()[0])
for mod, ver in [("pygame", "__version__"), ("numpy", "__version__"), ("scipy", "__version__"), ("PIL", "__version__")]:
    try:
        m = __import__(mod)
        print(mod, getattr(m, ver, "?"))
    except Exception as e:
        print(mod, "MISSING:", e)

try:
    from PIL import Image
    for f in sorted(os.listdir("assets")):
        p = os.path.join("assets", f)
        if os.path.isfile(p):
            try:
                im = Image.open(p)
                print(f"asset {f:24} {im.size} {im.mode} {os.path.getsize(p)//1024}KB")
            except Exception as e:
                print(f"asset {f}: not image ({e})")
    # hoop anchor sanity: is (94,183) inside the cropped right-half surface?
    im = Image.open("assets/hoopnobgd.png")
    w, h = im.size
    print(f"hoopnobgd full {im.size}; right-half width={w//2}, anchor (94,183) inside right half: {0<=94<w//2 and 0<=183<h}")
except Exception as e:
    print("PIL inspect failed:", e)
