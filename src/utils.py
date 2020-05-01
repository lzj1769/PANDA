import os
import numpy as np
import random
from sklearn.metrics import cohen_kappa_score
import torch
from PIL import Image
from albumentations import (
    OneOf, IAAAdditiveGaussianNoise, GaussNoise,
    Compose, Normalize, HorizontalFlip,
    VerticalFlip, ShiftScaleRotate, RandomBrightnessContrast,
    RandomRotate90)
from albumentations.pytorch import ToTensorV2


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


def get_transforms(*, data):
    assert data in ('train', 'valid')

    if data == 'train':
        return Compose([
            # HorizontalFlip(p=0.5),
            # VerticalFlip(p=0.5),
            # RandomRotate90(p=0.5),
            # OneOf([
            #     IAAAdditiveGaussianNoise(),
            #     GaussNoise(),
            # ], p=0.2),
            # RandomBrightnessContrast(p=0.5),
            # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2,
            #                  rotate_limit=45, p=0.2),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            ToTensorV2(),
        ])


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


def tile(img, tile_size=256, num_tiles=12):
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

    #for i in range(img.shape[0]):
    #    image = Image.fromarray(img[i])
    #    image.save(f"/home/rs619065/raw_{i}.png")

    return img
