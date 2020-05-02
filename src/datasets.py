import os
import cv2
import pandas as pd
import numpy as np
import skimage.io
import openslide
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations import (
    OneOf, IAAAdditiveGaussianNoise, GaussNoise,
    Compose, HorizontalFlip,
    VerticalFlip, ShiftScaleRotate, RandomBrightnessContrast,
    RandomRotate90)

import configure
import utils

mean = torch.tensor([1.0 - 0.90949707, 1.0 - 0.8188697, 1.0 - 0.87795304])
std = torch.tensor([0.36357649, 0.49984502, 0.40477625])


class PandaDataset(Dataset):
    def __init__(self, df, data_dir, data=None, transform=None,
                 image_width=256, image_height=256):
        self.df = df
        self.data_dir = data_dir
        self.data = data
        self.image_width = image_width
        self.image_height = image_height
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['image_id'].values[idx]
        file_path = f'{self.data_dir}/{file_name}.tiff'
        image = skimage.io.MultiImage(file_path)[-1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.df['isup_grade'].values[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        # split image
        image = utils.tile(image, tile_size=128, num_tiles=12)
        image = torch.from_numpy(1.0 - image / 255.0).float()
        image = (image - mean) / std
        image = image.permute(0, 3, 1, 2)

        return image, label


def get_transforms():
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        # OneOf([
        #     IAAAdditiveGaussianNoise(),
        #     GaussNoise(),
        # ], p=0.2),
        # RandomBrightnessContrast(p=0.5),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2,
                         rotate_limit=45, p=0.2),
    ])


def get_dataloader(data, data_dir, fold, batch_size,
                   num_workers, image_width, image_height):
    assert data in ('train', 'valid')

    dataloader = ""
    if data == "train":
        df_train_path = os.path.join(configure.SPLIT_FOLDER, "fold_{}_train.csv".format(fold))
        df_train = pd.read_csv(df_train_path)

        train_dataset = PandaDataset(
            df=df_train,
            data_dir=data_dir,
            data="train",
            transform=get_transforms(),
            image_width=image_width,
            image_height=image_height)

        dataloader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=True,
                                shuffle=True)
    elif data == "valid":
        df_valid_path = os.path.join(configure.SPLIT_FOLDER, "fold_{}_valid.csv".format(fold))
        df_valid = pd.read_csv(df_valid_path)

        valid_dataset = PandaDataset(
            df=df_valid,
            data_dir=data_dir,
            data="valid",
            transform=None,
            image_width=image_width,
            image_height=image_height)

        dataloader = DataLoader(dataset=valid_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=False,
                                shuffle=False)

    return dataloader
