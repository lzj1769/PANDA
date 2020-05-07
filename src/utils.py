import os
import numpy as np
import cv2
import random
from sklearn.metrics import cohen_kappa_score
import torch
import matplotlib.pyplot as plt
import io
from itertools import product
import tensorflow as tf


def quadratic_weighted_kappa(y_hat, y):
    return cohen_kappa_score(y_hat, y, weights='quadratic')


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def crop_white(image: np.ndarray) -> np.ndarray:
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    ys, = (image.min((1, 2)) != 255).nonzero()
    xs, = (image.min(0).min(1) != 255).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return image

    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def pred_to_isup(pred):
    threshold = [0.5, 1.5, 2.5, 3.5, 4.5]
    pred[pred < threshold[0]] = 0
    pred[(pred >= threshold[0]) & (pred < threshold[1])] = 1
    pred[(pred >= threshold[1]) & (pred < threshold[2])] = 2
    pred[(pred >= threshold[2]) & (pred < threshold[3])] = 3
    pred[(pred >= threshold[3]) & (pred < threshold[4])] = 4
    pred[pred >= threshold[4]] = 5

    return pred


def tile(img, tile_size=128, num_tiles=12):
    shape = img.shape
    # image = Image.fromarray(img)
    # image.save("/home/rs619065/raw.png")

    pad0, pad1 = (tile_size - shape[0] % tile_size) % tile_size, \
                 (tile_size - shape[1] % tile_size) % tile_size

    img = np.pad(img, [[pad0 // 2, pad0 - pad0 // 2],
                       [pad1 // 2, pad1 - pad1 // 2],
                       [0, 0]], constant_values=255, mode='constant')
    # image = Image.fromarray(img)
    # image.save("/home/rs619065/raw_pad.png")
    img = img.reshape(img.shape[0] // tile_size, tile_size,
                      img.shape[1] // tile_size, tile_size, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, 3)
    if len(img) < num_tiles:
        img = np.pad(img, [[0, num_tiles - len(img)],
                           [0, 0], [0, 0], [0, 0]],
                     constant_values=255, mode='constant')

    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:num_tiles]
    img = img[idxs]

    # for i in range(img.shape[0]):
    #    image = Image.fromarray(img[i])
    #    image.save(f"/home/rs619065/raw_{i}.png")

    return img


def enhance_image(image, contrast=1, brightness=15):
    """
    Enhance constrast and brightness of images
    """
    img_enhanced = cv2.addWeighted(image, contrast, image, 0, brightness)
    return img_enhanced


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
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

