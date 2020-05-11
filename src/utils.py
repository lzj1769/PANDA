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


if __name__ == "__main__":
    y_true = [1, 2, 3, 3, 4, 5, 0, 1, 2, 3, 4, 5]
    y_pred = np.array([-10, 2.3, 4.3, 4.5, 2.4, 4.4, 5.5, 2.7, 0.1, 0.3, 0.4, 3.7])

    score, threshold = find_threshold(y_true=y_true, y_pred=y_pred)

    print(score)
    print(threshold)
