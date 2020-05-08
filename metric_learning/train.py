import argparse
import os
import sys
import numpy as np
import warnings
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.tensorboard import SummaryWriter

from model import PandaNet, ArcMarginProduct
import configure
import datasets
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
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_workers", default=24, type=int,
                        help="How many sub-processes to use for data.")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--tile_size", default=128, type=int)
    parser.add_argument("--num_tiles", default=12, type=int)
    parser.add_argument("--learning_rate", default=3e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=1e-04, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--log",
                        action="store_true",
                        help='write training history')
    parser.add_argument("--epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    return parser.parse_args()


def train(dataloader, model, metric, criterion, optimizer):
    model.train()

    train_loss = 0.0
    preds, train_labels = [], []
    for i, (images, target) in enumerate(dataloader):
        images = images.to("cuda")
        target = target.to("cuda")

        embeddings = model(images)
        thetas = metric(embeddings, target)

        loss = criterion(thetas, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(dataloader)

    return train_loss


def valid(train_dataloader, valid_dataloader, model):
    model.eval()

    with torch.no_grad():
        # get embedding for training data
        for images, labels in train_dataloader:


def main():
    args = parse_args()

    # set random seed
    utils.seed_torch(args.seed)

    # Setup CUDA, GPU
    if not torch.cuda.is_available():
        print("cuda is not available")
        exit(0)

    train_loader, valid_loader = datasets.get_dataloader(
        fold=args.fold,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    model = PandaNet(arch=args.arch)
    model.to("cuda")

    metric = ArcMarginProduct(in_features=512, out_features=6).to("cuda")

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 60, 90], gamma=0.5)

    """ Train the model """
    from datetime import datetime
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
        train_loss = train(
            dataloader=train_loader,
            model=model,
            criterion=criterion,
            metric=metric,
            optimizer=optimizer)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    print(f'training finished: {current_time}')


if __name__ == "__main__":
    main()
