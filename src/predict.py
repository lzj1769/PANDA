import argparse
import os
import sys
import cv2
import skimage.io
import pandas as pd
import numpy as np
import warnings

import torch
from torch.utils.data import Dataset, DataLoader

from model import PandaNet
import configure
import utils

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--arch', metavar='ARCH', default='efficientnet-b0',
                        help='model architecture (default: efficientnet-b0)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_workers", default=24, type=int,
                        help="How many sub-processes to use for data.")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")

    return parser.parse_args()


mean = torch.tensor([1.0 - 0.90949707, 1.0 - 0.8188697, 1.0 - 0.87795304])
std = torch.tensor([0.36357649, 0.49984502, 0.40477625])


class PandaDataset(Dataset):
    def __init__(self, df, data_dir):
        self.df = df
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['image_id'].values[idx]
        file_path = f'{self.data_dir}/{file_name}.tiff'
        image = skimage.io.MultiImage(file_path)[-1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # split image
        image = utils.tile(image, tile_size=128, num_tiles=12)
        image = torch.from_numpy(1.0 - image / 255.0).float()
        image = (image - mean) / std
        image = image.permute(0, 3, 1, 2)

        return image


def predict(dataloader, model):
    model.eval()
    with torch.no_grad():
        preds = []
        for image in dataloader:
            image = image.to("cuda")
            output = model(image)
            pred_isup = output.view(output.size(0), -1).argmax(-1).cpu().numpy()
            preds.append(pred_isup)

        preds = np.concatenate(preds)

        return preds


def main():
    args = parse_args()

    # set random seed
    utils.seed_torch(42)

    model = PandaNet(arch=args.arch, pretrained=False)
    model_path = os.path.join(configure.MODEL_PATH,
                              f'{args.arch}_fold_{args.fold}_128_12.pth')

    model.load_state_dict(torch.load(model_path))
    model.cuda()

    df = pd.read_csv(configure.TRAIN_DF)

    dataset = PandaDataset(df=df,
                           data_dir=configure.TRAIN_IMAGE_PATH)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            shuffle=False)

    preds = predict(dataloader, model)
    score = utils.quadratic_weighted_kappa(preds, df['isup_grade'])
    print(score)


if __name__ == "__main__":
    main()
