import os
import cv2
import pandas as pd
import skimage.io
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations.augmentations.functional as F

import configure
import utils


class PandaDataset(Dataset):
    def __init__(self, df, data_dir, data=None,
                 transform=None, image_width=256, image_height=256):
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
        file_path = f'{self.data_dir}/{file_name}.png'
        image = skimage.io.imread(file_path)

        # remove white background
        # image = utils.crop_white(image[-1])
        image = cv2.resize(image, (self.image_width, self.image_height))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.df['isup_grade'].values[idx]

        if self.data == "train":
            augmented = self.transform(image=image)
            image = augmented['image'] / 255.0

        # do tta
        elif self.data == "valid":
            augmented1 = self.transform(image=image)
            augmented2 = self.transform(image=F.rot90(image, factor=1))
            augmented3 = self.transform(image=F.rot90(image, factor=2))
            augmented4 = self.transform(image=F.rot90(image, factor=3))

            augmented5 = self.transform(image=F.hflip(image))
            augmented6 = self.transform(image=F.rot90(F.hflip(image), factor=1))
            augmented7 = self.transform(image=F.rot90(F.hflip(image), factor=2))
            augmented8 = self.transform(image=F.rot90(F.hflip(image), factor=3))

            image = torch.stack([augmented1['image'],
                                 augmented2['image'],
                                 augmented3['image'],
                                 augmented4['image'],
                                 augmented5['image'],
                                 augmented6['image'],
                                 augmented7['image'],
                                 augmented8['image']])

        return image, label


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
                                batch_size=2,
                                num_workers=num_workers,
                                pin_memory=False,
                                shuffle=False)

    return dataloader
