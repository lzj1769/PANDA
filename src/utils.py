import os
import numpy as np
import random
from sklearn.metrics import cohen_kappa_score
import torch
from albumentations import (
    OneOf, IAAAdditiveGaussianNoise, GaussNoise,
    ElasticTransform, Blur, GridDistortion,
    Compose, Normalize, HorizontalFlip, HueSaturationValue,
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
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomRotate90(p=0.5),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.2),
            RandomBrightnessContrast(p=0.5),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2,
                             rotate_limit=180, p=0.5),
            # Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225],
            # ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            # Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225],
            # ),
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
