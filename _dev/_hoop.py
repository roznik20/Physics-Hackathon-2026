import numpy as np
from PIL import Image
im = Image.open("_hoop_surf.png")
arr = np.array(im)
print("shape", arr.shape)
if arr.ndim == 3:
    a = arr[:, :, 3] if arr.shape[2] == 4 else np.full(arr.shape[:2], 255)
    ys, xs = np.where(a > 8)
    print("content bbox x", xs.min(), xs.max(), "y", ys.min(), ys.mean().__round__() if False else round(float(ys.mean()),1), "ymax", ys.max())
    # horizontal profile: where is content densest per row?
    col = a.max(axis=0)  # not needed
    # find rim: the anchor is (94,183). Show a downsampled ASCII of alpha
    small = np.array(Image.open("_hoop_surf.png").convert("L").resize((66, 48)))
    for r in small[::3]:
        print("".join("#" if v > 20 else ("." if v > 8 else " ") for v in r))
    print("anchor (94,183) in 66x48 grid ~", (94/198*66).__round__(), (183/360*48).__round__())
