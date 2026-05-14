import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("output_images", exist_ok=True)

with h5py.File("nyu_depth_v2_labeled.mat", "r") as f:
    images = f["images"]

    for i in range(10):
        img = images[i]
        img = np.transpose(img, (2, 1, 0))

        plt.imsave(f"output_images/img_{i}.png", img.astype(np.uint8))