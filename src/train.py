import argparse
import os
import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from model import PandaNet
import datasets
import configure
import utils

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--arch', metavar='ARCH', default='se_resnext50_32x4d',
                        help='model architecture (default: se_resnext50_32x4d)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument("--level", default=1, type=int,
                        help="which level to use, can only be 0, 1, 2")
    parser.add_argument("--tile_size", default=256, type=int,
                        help="size of tile, available are 128 and 256")
    parser.add_argument("--num_tiles", default=12, type=int,
                        help="how many tiles for each image. Default: 12")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_workers", default=4, type=int,
                        help="How many sub-processes to use for data.")
    parser.add_argument("--per_gpu_batch_size", default=6, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=3e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=1e-04, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--loss", default='mse', type=str,
                        help="Which loss function to use."
                             "Available: l1, l2, smooth_l1. Default: mse")
    parser.add_argument("--log",
                        action="store_true",
                        help='write training history')
    parser.add_argument("--epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    return parser.parse_args()


def train(dataloader, model, criterion, optimizer, args):
    model.train()

    train_loss = 0.0
    preds, train_labels = [], []
    for i, (images, target) in enumerate(dataloader):
        images = images.to(args.device)
        target = target.to(args.device)

        output = model(images)

        loss = criterion(output.view(-1), target.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds.append(output.view(-1).detach().cpu().numpy())
        train_labels.append(target.detach().cpu().numpy())
        train_loss += loss.item() / len(dataloader)

    preds = np.concatenate(preds)
    train_labels = np.concatenate(train_labels)

    # threshold = utils.find_threshold(y_true=train_labels,
    #                                 y_pred=preds)

    threshold = [0.5, 1.5, 2.5, 3.5, 4.5]

    isup_preds = pd.cut(preds, [-np.inf] + list(np.sort(threshold)) + [np.inf],
                        labels=[0, 1, 2, 3, 4, 5])
    score = utils.fast_qwk(isup_preds, train_labels)
    cm = confusion_matrix(train_labels, isup_preds)
    del preds

    return train_loss, score, cm, threshold


def valid(dataloader, model, criterion, threshold, args):
    model.eval()

    with torch.no_grad():
        valid_loss = 0.0
        preds, valid_labels = [], []
        for i, (images, target) in enumerate(dataloader):
            bs = images.size(0)

            images = images.to(args.device)
            target = target.to(args.device)

            # dihedral TTA
            images = torch.stack([images, images.flip(-1),
                                  images.flip(-2), images.flip(-1, -2),
                                  images.transpose(-1, -2), images.transpose(-1, -2).flip(-1),
                                  images.transpose(-1, -2).flip(-2), images.transpose(-1, -2).flip(-1, -2)], 1)
            images = images.view(-1, args.num_tiles, 3, args.tile_size, args.tile_size)

            output = model(images).view(bs, 8, -1).mean(1).view(-1)
            loss = criterion(output, target.float())

            preds.append(output.detach().cpu().numpy())
            valid_labels.append(target.detach().cpu().numpy())
            valid_loss += loss.item() / len(dataloader)

        preds = np.concatenate(preds)
        valid_labels = np.concatenate(valid_labels)

        isup_preds = pd.cut(preds, [-np.inf] + list(np.sort(threshold)) + [np.inf],
                            labels=[0, 1, 2, 3, 4, 5])
        score = utils.fast_qwk(isup_preds, valid_labels)
        cm = confusion_matrix(valid_labels, isup_preds)
        del preds

        return valid_loss, score, cm


def main():
    args = parse_args()

    # set random seed
    utils.seed_torch(args.seed)

    # Setup CUDA, GPU
    if not torch.cuda.is_available():
        print("cuda is not available")
        exit(0)
    else:
        args.device = torch.device("cuda")
        print(f"available cuda: {args.device}")

    # Setup model
    model = PandaNet(arch=args.arch, num_classes=1)
    model.to(args.device)

    # Setup data
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    print(f"loading data: {current_time}")
    filename = f"train_images_level_{args.level}_{args.tile_size}_{args.num_tiles}.npy"
    data = np.load(os.path.join(configure.DATA_PATH, filename),
                   allow_pickle=True)

    train_loader, valid_loader = datasets.get_dataloader(
        data=data,
        fold=args.fold,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers)

    # define loss function (criterion) and optimizer
    if args.loss == "l1":
        criterion = torch.nn.L1Loss()
    elif args.loss == "mse":
        criterion = torch.nn.MSELoss()
    elif args.loss == "smooth_l1":
        criterion = torch.nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 60, 90], gamma=0.5)

    """ Train the model """
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_prefix = f'{current_time}_{args.arch}_fold_{args.fold}_{args.tile_size}_{args.num_tiles}'
    log_dir = os.path.join(configure.TRAINING_LOG_PATH,
                           log_prefix)

    tb_writer = None
    if args.log:
        tb_writer = SummaryWriter(log_dir=log_dir)

    best_score = 0.0
    model_path = os.path.join(configure.MODEL_PATH,
                              f'{args.arch}_fold_{args.fold}_{args.tile_size}_{args.num_tiles}.pth')

    print(f'training started: {current_time}')
    for epoch in range(args.epochs):
        train_loss, train_score, train_cm, threshold = train(
            dataloader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            args=args)

        valid_loss, valid_score, valid_cm = valid(
            dataloader=valid_loader,
            model=model,
            criterion=criterion,
            threshold=threshold,
            args=args)

        learning_rate = scheduler.get_lr()[0]
        if args.log:
            tb_writer.add_scalar("learning_rate", learning_rate, epoch)
            tb_writer.add_scalar("Loss/train", train_loss, epoch)
            tb_writer.add_scalar("Loss/valid", valid_loss, epoch)
            tb_writer.add_scalar("Score/train", train_score, epoch)
            tb_writer.add_scalar("Score/valid", valid_score, epoch)

            # Log the confusion matrix as an image summary.
            figure = utils.plot_confusion_matrix(train_cm, class_names=[0, 1, 2, 3, 4, 5], score=train_score)
            cm_image = utils.plot_to_image(figure)
            tb_writer.add_image("Confusion Matrix train", cm_image, epoch)

            figure = utils.plot_confusion_matrix(valid_cm, class_names=[0, 1, 2, 3, 4, 5], score=valid_score)
            cm_image = utils.plot_to_image(figure)
            tb_writer.add_image("Confusion Matrix valid", cm_image, epoch)

        if valid_score > best_score:
            best_score = valid_score
            state = {'state_dict': model.state_dict(),
                     'threshold': threshold,
                     'train_score': train_score,
                     'valid_score': valid_score}
            torch.save(state, model_path)

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        print(f"epoch:{epoch:02d}, "
              f"train:{train_score:0.3f}, valid:{valid_score:0.3f}, "
              f"best:{best_score:0.3f}, date:{current_time}")

        scheduler.step()

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    print(f'training finished: {current_time}')

    if args.log:
        tb_writer.close()


if __name__ == "__main__":
    main()
