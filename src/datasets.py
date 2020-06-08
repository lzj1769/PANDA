import os
import pandas as pd
import torch
import skimage.io
from torch.utils.data import Dataset, DataLoader
from albumentations import (
    CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose
)

import configure
import utils


class PandaDataset(Dataset):
    def __init__(self, df, level, patch_size, num_patches, transform):
        self.df = df
        self.level = level
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df['image_id'].values[idx]
        file_path = f'{configure.TRAIN_IMAGE_PATH}/{image_id}.tiff'
        wsi = skimage.io.MultiImage(file_path)[self.level]
        wsi = utils.crop_white(wsi)
        images = utils.get_patches(wsi,
                                   patch_size=self.patch_size,
                                   num_patches=self.num_patches)

        if self.transform:
            for i in range(images.shape[0]):
                images[i] = self.transform(image=images[i])['image']

        images = torch.from_numpy(1 - images / 255.0).float()
        images = images.permute(0, 3, 1, 2)

        label = self.df['isup_grade'].values[idx]

        return images, label


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


def get_dataloader(fold, batch_size, level, patch_size, num_patches, num_workers):
    df_train = pd.read_csv(os.path.join(configure.SPLIT_FOLDER,
                                        f"fold_{fold}_train.csv"))

    df_valid = pd.read_csv(os.path.join(configure.SPLIT_FOLDER,
                                        f"fold_{fold}_valid.csv"))

    train_dataset = PandaDataset(df=df_train,
                                 level=level,
                                 patch_size=patch_size,
                                 num_patches=num_patches,
                                 transform=get_transforms())

    valid_dataset = PandaDataset(df=df_valid,
                                 level=level,
                                 patch_size=patch_size,
                                 num_patches=num_patches,
                                 transform=None)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  shuffle=True,
                                  drop_last=True)

    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  shuffle=False)

    return train_dataloader, valid_dataloader
