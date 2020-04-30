import os
import numpy as np
import pandas as pd
import random
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
import torch
from albumentations import (
    OneOf, IAAAdditiveGaussianNoise, GaussNoise,
    ElasticTransform, Blur, GridDistortion,
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
            Blur(p=0.5),
            GridDistortion(p=0.5),
            ElasticTransform(p=0.5),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=180, p=0.5),
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


def pred_to_isup(pred):
    threshold = [0.5, 1.5, 2.5, 3.5, 4.5]
    pred[pred < threshold[0]] = 0
    pred[(pred >= threshold[0]) & (pred < threshold[1])] = 1
    pred[(pred >= threshold[1]) & (pred < threshold[2])] = 2
    pred[(pred >= threshold[2]) & (pred < threshold[3])] = 3
    pred[(pred >= threshold[3]) & (pred < threshold[4])] = 4
    pred[pred >= threshold[4]] = 5

    return pred
