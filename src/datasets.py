import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations import (
    CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose
)

import configure

MEAN = torch.tensor([0.64455969, 0.47587813, 0.72864011])
STD = torch.tensor([0.39921443, 0.46409423, 0.4326094])


class PandaDataset(Dataset):
    def __init__(self, df, data, transform):
        self.df = df
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df['image_id'].values[idx]
        image = self.data.item().get(image_id)

        print(type(image))
        exit(0)

        label = self.df['isup_grade'].values[idx]
        # smooth_label = np.random.normal(loc=label, scale=0.2)

        if self.transform:
            for i in range(image.shape[0]):
                image[i] = self.transform(image=image[i])['image']

        # split image
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


def get_dataloader(data, fold, batch_size, num_workers):
    df_train = pd.read_csv(os.path.join(configure.SPLIT_FOLDER, f"fold_{fold}_train.csv"))
    train_dataset = PandaDataset(df=df_train,
                                 data=data,
                                 transform=get_transforms())
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  shuffle=True)

    df_valid = pd.read_csv(os.path.join(configure.SPLIT_FOLDER, f"fold_{fold}_valid.csv"))
    valid_dataset = PandaDataset(df=df_valid,
                                 data=data,
                                 transform=None)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  pin_memory=False,
                                  shuffle=False)

    return train_dataloader, valid_dataloader
