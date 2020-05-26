import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy
from functools import partial
import random
from sklearn.metrics import cohen_kappa_score
import torch
import matplotlib.pyplot as plt
import io
from itertools import product
import tensorflow as tf
from numba import jit


@jit
def fast_qwk(a1, a2, max_rat=5):
    assert (len(a1) == len(a2))
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1,))
    hist2 = np.zeros((max_rat + 1,))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o += (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def pred_to_isup(pred, threshold=None):
    if threshold is None:
        threshold = [0.5, 1.5, 2.5, 3.5, 4.5]

    pred[pred < threshold[0]] = 0
    pred[(pred >= threshold[0]) & (pred < threshold[1])] = 1
    pred[(pred >= threshold[1]) & (pred < threshold[2])] = 2
    pred[(pred >= threshold[2]) & (pred < threshold[3])] = 3
    pred[(pred >= threshold[3]) & (pred < threshold[4])] = 4
    pred[pred >= threshold[4]] = 5

    return pred


def kappa_loss(coef, X, y):
    preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf],
                   labels=[0, 1, 2, 3, 4, 5])

    return -fast_qwk(y, preds)


def find_threshold(y_true, y_pred):
    loss_partial = partial(kappa_loss, X=y_pred, y=y_true)
    initial_threshold = [0.5, 1.5, 2.5, 3.5, 4.5]
    threshold = minimize(loss_partial,
                         initial_threshold,
                         method='nelder-mead')['x']

    return threshold


def plot_confusion_matrix(cm, class_names, score):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion matrix: {score}")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # tf.Tensor to torch.tensor
    image = torch.from_numpy(image.numpy()).permute(2, 0, 1)

    return image


def get_patches(wsi, patch_size, num_patches=None, tissue_threshold=None):
    """
    Description
    __________
    Tilizer module made by @iafoss that can be found in the notebook:
    https://www.kaggle.com/iafoss/panda-concat-tile-pooling-starter-inference
    Takes a base image and returns the N tiles with the largest differnce
    from a white backgound each with a given square size of input-sz.

    Returns
    __________
    - List of size N with each item being a numpy array tile.
    """

    assert patch_size is not None, "patch size cannot be none"

    # Get the shape of the input image
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

    if tissue_threshold:
        idxs = []
        for idx, patch in enumerate(patches):
            summed_matrix = np.sum(patch, axis=-1)
            num_white_pixels = np.count_nonzero(summed_matrix > 620)
            ratio_tissue_pixels = 1 - num_white_pixels / (patch.shape[0] * patch.shape[1])

            if ratio_tissue_pixels > tissue_threshold:
                idxs.append(idx)

        patches = patches[idxs]

    if num_patches:
        # If there are not enough tiles to meet desired N pad again
        if len(patches) < num_patches:
            patches = np.pad(patches,
                             [[0, num_patches - len(patches)],
                              [0, 0], [0, 0], [0, 0]],
                             constant_values=255, mode="constant")

        idxs = np.argsort(patches.reshape(patches.shape[0], -1).sum(-1))[:num_patches]
        patches = patches[idxs]

    return patches


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


if __name__ == "__main__":
    import configure
    import skimage.io
    import cv2
    from PIL import Image

    df = pd.read_csv(configure.TRAIN_DF)
    print(len(df))
    image_id = df['image_id'].values.tolist()[999]
    file_path = f'{configure.TRAIN_IMAGE_PATH}/{image_id}.tiff'
    image = skimage.io.MultiImage(file_path)[0]
    print(image.shape)
    patches = get_patches(image, patch_size=512, num_patches=128)

    for i in range(patches.shape[0]):
        img = Image.fromarray(cv2.resize(patches[i], (128, 128)))
        img.save(f'{image_id}_{i}.png')

    exit(0)
