import argparse
import os
import sys
import numpy as np
import warnings

import torch
from torch.utils.tensorboard import SummaryWriter

from model import PandaEfficientNet
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
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--image_width", default=384, type=int,
                        help="Image width.")
    parser.add_argument("--image_height", default=384, type=int,
                        help="Image height.")
    parser.add_argument("--weight_decay", default=1e-04, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--log",
                        action="store_true",
                        help='write training history')
    parser.add_argument("--epochs", default=200, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    return parser.parse_args()


def train(train_loader, model, criterion, optimizer, args):
    model.train()

    train_loss = 0.0
    preds, train_labels = [], []
    for i, (images, target1, target2) in enumerate(train_loader):
        # get ISUP grade
        target_isup = utils.gleason_to_isup(target1.tolist(), target2.tolist())

        images = images.to(args.device)
        target1 = target1.to(args.device)
        target2 = target2.to(args.device)

        # compute output
        output1, output2 = model(images)
        loss1 = criterion(output1.view(-1), target1.float())
        loss2 = criterion(output2.view(-1), target2.float())
        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output1 = utils.pred_to_gleason(output1.detach().cpu().numpy())
        output2 = utils.pred_to_gleason(output2.detach().cpu().numpy())
        pred_isup = utils.gleason_to_isup(output1, output2)

        preds.append(pred_isup)
        train_labels.append(target_isup)

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
        for i, (images, target1, target2) in enumerate(valid_loader):
            # get ISUP grade
            target_isup = utils.gleason_to_isup(target1.tolist(), target2.tolist())

            images = images.to(args.device)
            target1 = target1.to(args.device)
            target2 = target2.to(args.device)

            # compute output
            output1, output2 = model(images)
            loss = criterion(output1.view(-1), target1.float()) + criterion(output2.view(-1), target2.float())

            output1 = utils.pred_to_gleason(output1.detach().cpu().numpy())
            output2 = utils.pred_to_gleason(output2.detach().cpu().numpy())
            pred_isup = utils.gleason_to_isup(output1, output2)

            preds.append(pred_isup)
            valid_labels.append(target_isup)

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

    model = PandaEfficientNet(arch=args.arch)
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
    criterion = torch.nn.SmoothL1Loss()
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

    if args.log:
        tb_writer = SummaryWriter(log_dir=log_dir)

    best_score = 0.0
    model_path = os.path.join(configure.MODEL_PATH,
                              f'{args.arch}_fold_{args.fold}_image_{args.image_width}_{args.image_height}.pth')

    print(f'training started: {current_time}')
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
        if args.log:
            tb_writer.add_scalar("learning_rate", learning_rate, epoch)
            tb_writer.add_scalar("train_loss", train_loss, epoch)
            tb_writer.add_scalar("train_qwk", train_score, epoch)
            tb_writer.add_scalar("valid_loss", valid_loss, epoch)
            tb_writer.add_scalar("valid_score", valid_score, epoch)

        if valid_score > best_score:
            best_score = valid_score
            torch.save(model.state_dict(), model_path)
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            print(f"epoch: {epoch}, best score: {valid_score}, date: {current_time}")

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    print(f'training finished: {current_time}')


if __name__ == "__main__":
    main()
