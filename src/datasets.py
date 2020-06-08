import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from albumentations import VerticalFlip, HorizontalFlip, Transpose, Compose

import configure


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
        images = np.load(f"{configure.PATCH_PATH}/{image_id}.npy")[:self.num_patches]

        images = torch.from_numpy(1 - images / 255.0).float()
        images = images.permute(0, 3, 1, 2)

        if self.transform:
            choice = np.random.choice(8, 1)[0]
            if choice == 0:
                images = images.flip(-1)
            elif choice == 1:
                images = images.flip(-2)
            elif choice == 2:
                images = images.flip(-1, -2)
            elif choice == 3:
                images = images.transpose(-1, -2)
            elif choice == 4:
                images = images.transpose(-1, -2).flip(-1)
            elif choice == 5:
                images = images.transpose(-1, -2).flip(-2)
            elif choice == 6:
                images = images.transpose(-1, -2).flip(-1, -2)
            elif choice == 7:
                images = images

        label = self.df['isup_grade'].values[idx]

        return images, label


def get_transforms():
    return Compose([
        Transpose(p=0.5),
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5)
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
                                 transform=True)

    valid_dataset = PandaDataset(df=df_valid,
                                 level=level,
                                 patch_size=patch_size,
                                 num_patches=num_patches,
                                 transform=False)

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
