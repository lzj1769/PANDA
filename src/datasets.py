import os
import pandas as pd
import torch
import skimage.io
from torch.utils.data import Dataset, DataLoader
from albumentations import VerticalFlip, HorizontalFlip, Transpose, ShiftScaleRotate, Compose

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
        # patches = np.load(f"{configure.PATCH_PATH}/{image_id}.npy")

        file_path = f'{configure.TRAIN_IMAGE_PATH}/{image_id}.tiff'
        wsi = skimage.io.MultiImage(file_path)[self.level]
        patches = utils.get_patches(wsi,
                                    patch_size=self.patch_size,
                                    num_patches=self.num_patches)

        if self.transform:
            for i in range(patches.shape[0]):
                patches[i] = self.transform(image=patches[i])['image']

        patches = torch.from_numpy(1 - patches / 255.0).float()
        patches = patches.permute(0, 3, 1, 2)

        label = self.df['isup_grade'].values[idx]

        return patches, label


def get_transforms():
    return Compose([
        Transpose(p=0.5),
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=90, p=0.2)
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
