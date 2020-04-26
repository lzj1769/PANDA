import os
import numpy as np
import pandas as pd
import random
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
import torch
from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip
from albumentations.pytorch import ToTensorV2

import configure
import datasets


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_transforms(*, data):
    assert data in ('train', 'valid')

    if data == 'train':
        return Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


def get_dataloader(data, fold, batch_size, num_workers):
    assert data in ('train', 'valid')

    dataloader = ""
    if data == "train":
        df_train_path = os.path.join(configure.SPLIT_FOLDER, "fold_{}_train.csv".format(fold))
        df_train = pd.read_csv(df_train_path)

        train_dataset = datasets.TrainDataset(
            df=df_train,
            data_dir=configure.TRAIN_IMAGE_PATH,
            transform=get_transforms(data="train"))

        train_sampler = RandomSampler(train_dataset)
        dataloader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=True,
                                sampler=train_sampler)
    elif data == "valid":
        df_valid_path = os.path.join(configure.SPLIT_FOLDER, "fold_{}_valid.csv".format(fold))
        df_valid = pd.read_csv(df_valid_path)

        valid_dataset = datasets.TrainDataset(
            df=df_valid,
            data_dir=configure.TRAIN_IMAGE_PATH,
            transform=get_transforms(data="valid"))

        valid_sampler = SequentialSampler(valid_dataset)
        dataloader = DataLoader(dataset=valid_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=True,
                                sampler=valid_sampler)

    return dataloader
