import os
import numpy as np
import pandas as pd
import random
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
import torch
from albumentations import (
    OneOf, IAAAdditiveGaussianNoise, GaussNoise,
    Compose, Normalize, HorizontalFlip, HueSaturationValue,
    VerticalFlip, ShiftScaleRotate, RandomBrightnessContrast,
    RandomRotate90)
from albumentations.pytorch import ToTensorV2

import configure
import datasets

Gleason_ISUP = {'0+0': 0,
                '3+3': 1,
                '3+4': 2,
                '4+3': 3,
                '4+4': 4,
                '3+5': 4,
                '5+3': 4,
                '4+5': 5,
                '5+4': 5,
                '5+5': 5}


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
            HueSaturationValue(p=0.5),
            RandomBrightnessContrast(p=0.5),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


def get_dataloader(data, fold, batch_size, num_workers, image_width, image_height):
    assert data in ('train', 'valid')

    dataloader = ""
    if data == "train":
        df_train_path = os.path.join(configure.SPLIT_FOLDER, "fold_{}_train.csv".format(fold))
        df_train = pd.read_csv(df_train_path)

        train_dataset = datasets.PandaDataset(
            df=df_train,
            data_dir=configure.TRAIN_IMAGE_PATH,
            image_width=image_width,
            image_height=image_height,
            transform=get_transforms(data="train"))

        train_sampler = RandomSampler(train_dataset)
        dataloader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=True,
                                sampler=train_sampler)
    elif data == "valid":
        df_valid_path = os.path.join(configure.SPLIT_FOLDER, "fold_{}_valid.csv".format(fold))
        df_valid = pd.read_csv(df_valid_path)

        valid_dataset = datasets.PandaDataset(
            df=df_valid,
            data_dir=configure.TRAIN_IMAGE_PATH,
            image_width=image_width,
            image_height=image_height,
            transform=get_transforms(data="valid"))

        valid_sampler = SequentialSampler(valid_dataset)
        dataloader = DataLoader(dataset=valid_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=True,
                                sampler=valid_sampler)

    return dataloader


def crop_white(image: np.ndarray) -> np.ndarray:
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    ys, = (image.min((1, 2)) != 255).nonzero()
    xs, = (image.min(0).min(1) != 255).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return image
    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def pred_to_gleason(pred):
    thres = [2.5, 3.5, 4.5]  # 0, 3, 4, 5
    pred[pred < thres[0]] = 0
    pred[(pred >= thres[0]) & (pred < thres[1])] = 3
    pred[(pred >= thres[1]) & (pred < thres[2])] = 4
    pred[pred >= thres[2]] = 5

    return map(int, pred)


def gleason_to_isup(gleason_score_1, gleason_score_2):
    gleason_score = list()
    for (item1, item2) in zip(gleason_score_1, gleason_score_2):
        gleason_score.append(str(item1) + "+" + str(item2))

    isup_grade = []
    for gleason in gleason_score:
        if gleason not in Gleason_ISUP:
            isup_grade.append(0)
        else:
            isup_grade.append(Gleason_ISUP[gleason])

    return isup_grade



