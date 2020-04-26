import argparse
import os
import sys
import pandas as pd
import time
import warnings
import PIL

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
    parser.add_argument('-a', '--arch', metavar='ARCH', default='efficientnet-b0',
                        help='model architecture (default: efficientnet-b0)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_workers", default=4, type=int,
                        help="How many sub-processes to use for data.")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=500,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    return parser.parse_args()


def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()

    tr_loss = 0.0
    for i, (images, target) in enumerate(train_loader):
        images = images.to(args.device)
        target = target.to(args.device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tr_loss += loss.item() / len(train_loader)

    return tr_loss


def main():
    args = parse_args()

    # set random seed
    utils.seed_torch(args.seed)

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    if args.pretrained:
        model = EfficientNet.from_pretrained(model_name=args.arch, num_classes=6)
    else:
        model = EfficientNet.from_name(model_name=args.arch)

    model.to(args.device)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    train_loader = utils.get_dataloader(data="train",
                                        fold=args.fold,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers)

    valid_loader = utils.get_dataloader(data="valid",
                                        fold=args.fold,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate)

    """ Train the model """
    import socket
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(configure.TRAINING_LOG_PATH,
                           current_time + '_' + socket.gethostname())

    tb_writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(args.num_train_epochs):
        loss = train(train_loader=train_loader,
                     model=model,
                     criterion=criterion,
                     optimizer=optimizer,
                     epoch=epoch,
                     args=args)

        tb_writer.add_scalar("train_loss", loss, epoch)


if __name__ == "__main__":
    main()
