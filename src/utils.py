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


def get_patches(wsi, patch_size=None, num_patches=None):
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

    # If there are not enough tiles to meet desired N pad again
    if len(patches) < num_patches:
        patches = np.pad(patches,
                         [[0, num_patches - len(patches)],
                          [0, 0], [0, 0], [0, 0]],
                         constant_values=255, mode="constant")

    idxs = np.argsort(patches.reshape(patches.shape[0], -1).sum(-1))[:num_patches]

    return patches[idxs]


def crop_white(image):
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    ys, = (image.min((1, 2)) != 255).nonzero()
    xs, = (image.min(0).min(1) != 255).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return image
    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


if __name__ == "__main__":
    import configure
    import skimage.io
    import cv2
    from PIL import Image

    df = pd.read_csv(configure.TRAIN_DF)
    image_id = df['image_id'].values.tolist()[121]
    file_path = f'{configure.TRAIN_IMAGE_PATH}/{image_id}.tiff'
    image = skimage.io.MultiImage(file_path)[0]
    image = crop_white(image)
    patches = get_patches(image, patch_size=512, num_patches=128)

    for i in range(patches.shape[0]):
        # img = Image.fromarray(cv2.resize(patches[i], (128, 128)))
        img = Image.fromarray(patches[i])
        img.save(f'{image_id}_{i}.png')

    exit(0)
