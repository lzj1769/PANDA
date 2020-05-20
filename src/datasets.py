import os
import pandas as pd
import numpy as np
import torch
import cv2
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

        # resize
        image_resize = np.empty(shape=(image.shape[0], 192, 192, 3), dtype=np.uint8)
        for i in range(image.shape[0]):
            image_resize[i] = cv2.resize(image[i], (192, 192))

        image = torch.from_numpy(image_resize / 255.0).float()
        # image = torch.from_numpy(image / 255.0).float()
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
    df_train = pd.read_csv(os.path.join(configure.SPLIT_FOLDER,
                                        f"fold_{fold}_train.csv"))
    train_dataset = PandaDataset(df=df_train,
                                 data=data,
                                 transform=get_transforms())

    # weights = make_weights_for_balanced_classes(df_train, num_classes=6)
    # weights = torch.DoubleTensor(weights)
    # train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  shuffle=True)

    df_valid = pd.read_csv(os.path.join(configure.SPLIT_FOLDER,
                                        f"fold_{fold}_valid.csv"))
    valid_dataset = PandaDataset(df=df_valid,
                                 data=data,
                                 transform=None)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  shuffle=False)

    return train_dataloader, valid_dataloader


def make_weights_for_balanced_classes(df, num_classes):
    count = [0] * num_classes

    for label in df['isup_grade'].values.tolist():
        count[label] += 1

    weight_per_class = [0.] * num_classes
    N = float(sum(count))

    for i in range(num_classes):
        weight_per_class[i] = N / float(count[i])

    print(f"weight for each class: {weight_per_class}")

    weight = [0.] * len(df)

    for idx, label in enumerate(df['isup_grade'].values.tolist()):
        weight[idx] = weight_per_class[label]

    return weight
