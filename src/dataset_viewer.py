import cv2
import numpy as np
from pathlib import Path

RAW_DIR = Path("rawImages")
OUT_DIR = Path("dataset_previews")
OUT_DIR.mkdir(exist_ok=True)

def normalize(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    return (img * 255).astype(np.uint8)

for p in sorted(RAW_DIR.glob("*.npy")):

    arr = np.load(p)
    img = normalize(arr)

    out = OUT_DIR / f"{p.stem}.jpg"

    cv2.imwrite(str(out), img)

    print("Saved:", out)