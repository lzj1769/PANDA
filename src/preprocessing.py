import numpy as np
import pandas as pd
import skimage.io
import cv2
import os
from PIL import Image

import configure


def color_cut(img, color=[255, 255, 255]):
    """
    Description
    ----------
    Take a input image and remove all rows or columns that
    are only made of the input color [R,G,B]. The default color
    to cut from image is white.

    Parameters
    ----------
    input_slide: numpy array
        Slide to cut white cols/rows
    color: list
        List of [R,G,B] pixels to cut from the input slide

    Returns (1)
    -------
    - Numpy array of input_slide with white removed
    """
    # Remove by row
    row_not_blank = [row.all() for row in ~np.all(img == color, axis=1)]
    img = img[row_not_blank, :]

    # Remove by col
    col_not_blank = [col.all() for col in ~np.all(img == color, axis=0)]
    img = img[:, col_not_blank]
    return img


def tile(wsi, num_tiles=12, tile_size=256):
    """
    Description
    __________
    Tilizer module made by @iafoss that can be found in the notebook:
    https://www.kaggle.com/iafoss/panda-concat-tile-pooling-starter-inference
    Takes a base image and returns the N tiles with the largest differnce
    from a white backgound each with a given square size of input-sz.

    Parameters
    __________
    base_image: numpy array
        Image array to split into tiles and plot
    N: int
        This is the number of tiles to split the image into
    sz: int
        This is the size for each side of the square tiles

    Returns
    __________
    - List of size N with each item being a numpy array tile.
    """

    # Get the shape of the input image
    shape = wsi.shape

    # Find the padding such that the image divides evenly by the desired size
    pad0, pad1 = (tile_size - shape[0] % tile_size) % tile_size, (tile_size - shape[1] % tile_size) % tile_size

    # Pad the image with blank space to reach the above found targets
    wsi = np.pad(wsi,
                 [[pad0 // 2, pad0 - pad0 // 2],
                  [pad1 // 2, pad1 - pad1 // 2], [0, 0]],
                 constant_values=255, mode="constant")

    # Reshape and Transpose to get the images into tiles
    patches = wsi.reshape(wsi.shape[0] // tile_size, tile_size, wsi.shape[1] // tile_size, tile_size, 3)
    patches = patches.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, 3)

    idxs = []
    for idx, patch in enumerate(patches):
        summed_matrix = np.sum(patch, axis=-1)
        num_white_pixels = np.count_nonzero(summed_matrix > 620)
        ratio_tissue_pixels = 1 - num_white_pixels / (patch.shape[0] * patch.shape[1])

        print(ratio_tissue_pixels)
        # tissue_proportion = np.sum(~np.all(patch == [255, 255, 255], axis=-1)) / (patch.shape[0] * patch.shape[1])
        if ratio_tissue_pixels > 0.5:
            idxs.append(idx)

    # Sort the images by those with the lowest sum (i.e the least white)
    # idxs = np.argsort(patches.reshape(patches.shape[0], -1).sum(-1))[:num_tiles]

    # Select by index those returned from the above function
    patches = patches[idxs]

    return patches


if __name__ == "__main__":
    df = pd.read_csv(configure.TRAIN_DF)

    images = dict()
    mean, std = [], []
    for image_id in df['image_id'].values.tolist()[:2]:
        file_path = f'{configure.TRAIN_IMAGE_PATH}/{image_id}.tiff'
        image = skimage.io.MultiImage(file_path)[0]
        image = color_cut(image)
        images[image_id] = tile(image, tile_size=256)

        for i in range(images[image_id].shape[0]):
            img = Image.fromarray(images[image_id][i])
            img.save(f'{image_id}_{i}.png')

        exit(0)

        mean.append((images[image_id] / 255.0).reshape(-1, 3).mean(0))
        std.append(((images[image_id] / 255.0) ** 2).reshape(-1, 3).mean(0))

    # image stats
    img_avr = np.array(mean).mean(0)
    img_std = np.sqrt(np.array(std).mean(0) - img_avr ** 2)
    print('mean:', img_avr, ', std:', np.sqrt(img_std))

    images['mean'] = img_avr
    images['std'] = img_std

    np.save(os.path.join(configure.DATA_PATH, "train_images_level_0_256_16"),
            images)
