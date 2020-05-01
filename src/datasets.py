import os
import cv2
import pandas as pd
import numpy as np
import skimage.io
import openslide
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations.augmentations.functional as F
import PIL

import configure
import utils

PIL.Image.MAX_IMAGE_PIXELS = 933120000


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

        image = utils.tile(image, tile_size=128, num_tiles=12)
        label = self.df['isup_grade'].values[idx]

        aug_image = np.empty(shape=(12, 3, 128, 128), dtype=np.float32)
        for i in range(image.shape[0]):
            augmented = self.transform(image=image[i])
            aug_image[i] = augmented['image'] / 255.0

        return aug_image, label


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
            transform=utils.get_transforms(data="train"),
            image_width=image_width,
            image_height=image_height)

        dataloader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=True,
                                shuffle=False)
    elif data == "valid":
        df_valid_path = os.path.join(configure.SPLIT_FOLDER, "fold_{}_valid.csv".format(fold))
        df_valid = pd.read_csv(df_valid_path)

        valid_dataset = PandaDataset(
            df=df_valid,
            data_dir=data_dir,
            data="valid",
            transform=utils.get_transforms(data="valid"),
            image_width=image_width,
            image_height=image_height)

        dataloader = DataLoader(dataset=valid_dataset,
                                batch_size=batch_size // 3,
                                num_workers=num_workers,
                                pin_memory=False,
                                shuffle=False)

    return dataloader
