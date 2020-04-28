import argparse
import os
import sys
import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from model import EfficientNet
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
    parser.add_argument("--num_workers", default=4, type=int,
                        help="How many sub-processes to use for data.")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--image_width", default=384, type=int,
                        help="Image width.")
    parser.add_argument("--image_height", default=384, type=int,
                        help="Image height.")
    parser.add_argument("--weight_decay", default=1e-04, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--epochs", default=200, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    return parser.parse_args()


def train(train_loader, model, criterion, optimizer, args):
    model.train()

    train_loss = 0.0
    preds, train_labels = [], []
    for i, (images, target) in enumerate(train_loader):
        images = images.to(args.device)
        target = target.to(args.device)

        # compute output
        output = model(images)

        # loss = criterion(output, target)
        loss = criterion(output.view(-1), target.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        thrs = [0.5, 1.5, 2.5, 3.5, 4.5]
        output[output < thrs[0]] = 0
        output[(output >= thrs[0]) & (output < thrs[1])] = 1
        output[(output >= thrs[1]) & (output < thrs[2])] = 2
        output[(output >= thrs[2]) & (output < thrs[3])] = 3
        output[(output >= thrs[3]) & (output < thrs[4])] = 4
        output[output >= thrs[4]] = 5

        preds.append(output.detach().cpu().numpy())
        train_labels.append(target.detach().cpu().numpy())

        train_loss += loss.item() / len(train_loader)

    preds = np.concatenate(preds)
    train_labels = np.concatenate(train_labels)
    score = utils.quadratic_weighted_kappa(train_labels, preds)

    return train_loss, score


def valid(valid_loader, model, criterion, args):
    model.eval()

    with torch.no_grad():
        valid_loss = 0.0
        preds, valid_labels = [], []
        for i, (images, target) in enumerate(valid_loader):
            images = images.to(args.device)
            target = target.to(args.device)

            # compute output
            output = model(images)
            loss = criterion(output.view(-1), target.float())

            thrs = [0.5, 1.5, 2.5, 3.5, 4.5]
            output[output < thrs[0]] = 0
            output[(output >= thrs[0]) & (output < thrs[1])] = 1
            output[(output >= thrs[1]) & (output < thrs[2])] = 2
            output[(output >= thrs[2]) & (output < thrs[3])] = 3
            output[(output >= thrs[3]) & (output < thrs[4])] = 4
            output[output >= thrs[4]] = 5

            preds.append(output.detach().cpu().numpy())
            valid_labels.append(target.detach().cpu().numpy())

            valid_loss += loss.item() / len(valid_loader)

        preds = np.concatenate(preds)
        valid_labels = np.concatenate(valid_labels)
        score = utils.quadratic_weighted_kappa(valid_labels, preds)

        return valid_loss, score


def main():
    args = parse_args()

    # set random seed
    utils.seed_torch(args.seed)

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    if args.pretrained:
        model = EfficientNet.from_pretrained(model_name=args.arch, num_classes=1)
    else:
        model = EfficientNet.from_name(model_name=args.arch)

    model.to(args.device)

    train_loader = utils.get_dataloader(data="train",
                                        fold=args.fold,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        image_width=args.image_width,
                                        image_height=args.image_height)

    valid_loader = utils.get_dataloader(data="valid",
                                        fold=args.fold,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        image_width=args.image_width,
                                        image_height=args.image_height)

    # define loss function (criterion) and optimizer
    criterion = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 60, 90, 120], gamma=0.5)

    """ Train the model """
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_prefix = f'{current_time}_{args.arch}_fold_{args.fold}_image_{args.image_width}_{args.image_height}'
    log_dir = os.path.join(configure.TRAINING_LOG_PATH,
                           log_prefix)

    tb_writer = SummaryWriter(log_dir=log_dir)
    best_score = 0.0
    model_path = os.path.join(configure.MODEL_PATH,
                              f'{args.arch}_fold_{args.fold}_image_{args.image_width}_{args.image_height}.pth')

    for epoch in range(args.epochs):
        train_loss, train_score = train(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            args=args
        )

        valid_loss, valid_score = valid(
            valid_loader=valid_loader,
            model=model,
            criterion=criterion,
            args=args
        )

        scheduler.step()

        learning_rate = scheduler.get_lr()[0]

        tb_writer.add_scalar("learning_rate", learning_rate, epoch)
        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("train_qwk", train_score, epoch)
        tb_writer.add_scalar("valid_loss", valid_loss, epoch)
        tb_writer.add_scalar("valid_score", valid_score, epoch)

        if valid_score > best_score:
            best_score = valid_score
            torch.save(model.state_dict(), model_path)
            print(f"epoch: {epoch}, best score: {valid_score}")


if __name__ == "__main__":
    main()
