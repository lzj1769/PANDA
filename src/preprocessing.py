import numpy as np
import pandas as pd
import skimage.io
import cv2
import os
from PIL import Image

import configure


def get_patches(image_id, patch_size=256, num_patches=32):
    """
    Description
    __________
    Tilizer module made by @iafoss that can be found in the notebook:
    https://www.kaggle.com/iafoss/panda-concat-tile-pooling-starter-inference
    Takes a base image and returns the N tiles with the largest differnce
    from a white backgound each with a given square size of input-sz.
    """
    assert patch_size is not None, "patch size cannot be none"
    if os.path.exists(f"{configure.PATCH_PATH}/{image_id}.npy"):
        return 0

    # Get the shape of the input image
    file_path = f'{configure.TRAIN_IMAGE_PATH}/{image_id}.tiff'
    wsi = skimage.io.MultiImage(file_path)[1]
    shape = wsi.shape

    # Find the padding such that the image divides evenly by the desired size
    pad0, pad1 = (patch_size - shape[0] % patch_size) % patch_size, (patch_size - shape[1] % patch_size) % patch_size

    # Pad the image with blank space to reach the above found targets
    wsi = np.pad(wsi,
                 [[pad0 // 2, pad0 - pad0 // 2],
                  [pad1 // 2, pad1 - pad1 // 2], [0, 0]],
                 constant_values=255, mode="constant")

    # Reshape and Transpose to get the images into tiles
    patches = wsi.reshape(wsi.shape[0] // patch_size, patch_size, wsi.shape[1] // patch_size, patch_size, 3)
    patches = patches.transpose(0, 2, 1, 3, 4).reshape(-1, patch_size, patch_size, 3)

    # If there are not enough tiles to meet desired N pad again
    if len(patches) < num_patches:
        patches = np.pad(patches,
                         [[0, num_patches - len(patches)],
                          [0, 0], [0, 0], [0, 0]],
                         constant_values=255, mode="constant")

    idxs = np.argsort(patches.reshape(patches.shape[0], -1).sum(-1))[:num_patches]
    patches = patches[idxs]
    np.save(f"{configure.PATCH_PATH}/{image_id}.npy", patches)


if __name__ == "__main__":
    from multiprocessing import Pool

    df = pd.read_csv(configure.TRAIN_DF)
    with Pool(48) as p:
        p.map(get_patches, df['image_id'].values.tolist())
