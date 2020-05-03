import os
import cv2
import pandas as pd
import skimage.io
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations import (
    CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose
)

import configure
import utils


class PandaDataset(Dataset):
    def __init__(self, df, data_dir, data,
                 tile_size, num_tiles):
        self.df = df
        self.data_dir = data_dir
        self.data = data
        self.tile_size = tile_size
        self.num_tiles = num_tiles
        self.transform = get_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['image_id'].values[idx]
        file_path = f'{self.data_dir}/{file_name}.tiff'
        image = skimage.io.MultiImage(file_path)[-1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.df['isup_grade'].values[idx]

        if self.data == "train":
            image = self.transform(image=image)['image']

        # split image
        image = utils.tile(image, tile_size=self.tile_size, num_tiles=self.num_tiles)
        image = torch.from_numpy(image / 255.0).float()
        image = image.permute(0, 3, 1, 2)

        return image, label


def get_transforms():
    return Compose([
        RandomRotate90(p=0.5),
        Flip(p=0.5),
        Transpose(p=0.5),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ])


def get_dataloader(data, data_dir, fold, batch_size,
                   num_workers, tile_size, num_tiles):
    assert data in ('train', 'valid')

    dataloader = ""
    if data == "train":
        df_train_path = os.path.join(configure.SPLIT_FOLDER, "fold_{}_train.csv".format(fold))
        df_train = pd.read_csv(df_train_path)

        train_dataset = PandaDataset(
            df=df_train,
            data_dir=data_dir,
            data="train",
            tile_size=tile_size,
            num_tiles=num_tiles)

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
            tile_size=tile_size,
            num_tiles=num_tiles)

        dataloader = DataLoader(dataset=valid_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=False,
                                shuffle=False)

    return dataloader
