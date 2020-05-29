import os
import pandas as pd
import numpy as np
import torch
import skimage.io
import cv2
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
    def __init__(self, df, data_dir, level, patch_size, num_patches, transform):
        self.df = df
        self.data_dir = data_dir
        self.level = level
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df['image_id'].values[idx]
        file_path = f'{self.data_dir}/{image_id}.tiff'
        image = skimage.io.MultiImage(file_path)[self.level]
        image = utils.get_patches(image,
                                  patch_size=self.patch_size,
                                  num_patches=self.num_patches)

        # if self.transform:
        #     for i in range(image.shape[0]):
        #         image[i] = self.transform(image=image[i])['image']

        image = torch.from_numpy(1 - image / 255.0).float()
        image = image.permute(0, 3, 1, 2)

        # image augment
        # t = np.random.choice(8, 1)
        # if t[0] == 0:
        #     image = image
        # elif t[0] == 1:
        #     image = image.flip(-1)
        # elif t[0] == 2:
        #     image = image.flip(-2)
        # elif t[0] == 3:
        #     image = image.flip(-1, -2)
        # elif t[0] == 4:
        #     image = image.transpose(-1, -2)
        # elif t[0] == 5:
        #     image = image.transpose(-1, -2).flip(-1)
        # elif t[0] == 6:
        #     image = image.transpose(-1, -2).flip(-2)
        # elif t[0] == 7:
        #     image = image.transpose(-1, -2).flip(-1, -2)

        label = self.df['isup_grade'].values[idx]

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


def get_dataloader(fold, batch_size, level, patch_size, num_patches, num_workers):
    df_train = pd.read_csv(os.path.join(configure.SPLIT_FOLDER,
                                        f"fold_{fold}_train.csv"))

    df_valid = pd.read_csv(os.path.join(configure.SPLIT_FOLDER,
                                        f"fold_{fold}_valid.csv"))

    train_dataset = PandaDataset(df=df_train,
                                 data_dir=configure.TRAIN_IMAGE_PATH,
                                 level=level,
                                 patch_size=patch_size,
                                 num_patches=num_patches,
                                 transform=get_transforms())

    valid_dataset = PandaDataset(df=df_valid,
                                 data_dir=configure.TRAIN_IMAGE_PATH,
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
