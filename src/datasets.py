import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations import (
    CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose
)

import configure


class PandaDataset(Dataset):
    def __init__(self, df, data, transform):
        self.df = df
        self.data = data
        self.transform = transform
        self.mean = torch.from_numpy(data.item().get('mean')).float()
        self.std = torch.from_numpy(data.item().get('mean')).float()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df['image_id'].values[idx]
        image = self.data.item().get(image_id)

        label = self.df['isup_grade'].values[idx]
        # smooth_label = np.random.normal(loc=label, scale=0.2)

        if self.transform:
            for i in range(image.shape[0]):
                image[i] = self.transform(image=image[i])['image']

        # split image
        image = torch.from_numpy(image / 255.0).float()
        image = (image - self.mean) / self.std
        image = image.permute(0, 3, 1, 2)

        return image, label


def get_transforms():
    return Compose([
        RandomRotate90(p=0.5),
        Flip(p=0.5),
        Transpose(p=0.5),
    ])


def get_dataloader(data, fold, batch_size, num_workers):
    df_train = pd.read_csv(os.path.join(configure.SPLIT_FOLDER, f"fold_{fold}_train.csv"))
    train_dataset = PandaDataset(df=df_train,
                                 data=data,
                                 transform=get_transforms())
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  pin_memory=False,
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
