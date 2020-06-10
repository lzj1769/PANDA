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
    parser.add_argument("--patch_size", default=128, type=int,
                        help="size of patch, available are 128 and 256")
    parser.add_argument("--num_patches", default=64, type=int,
                        help="how many tiles for each image. Default: 12")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_workers", default=36, type=int,
                        help="How many sub-processes to use for data.")
    parser.add_argument("--per_gpu_batch_size", default=6, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--accumulation_steps", default=2, type=int,
                        help="accumulate the gradients. Default: 2")
    parser.add_argument("--learning_rate", default=3e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=1e-4, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--loss", default='smooth_l1', type=str,
                        help="Which loss function to use."
                             "Available: l1, l2, smooth_l1. Default: smooth_l1")
    parser.add_argument("--log",
                        action="store_true",
                        help='write training history')
    parser.add_argument("--resume",
                        action="store_true",
                        help='training model from check point')
    parser.add_argument("--epochs", default=50, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    return parser.parse_args()


def train(dataloader, model, criterion, optimizer, args):
    model.train()

    train_loss = 0.0
    for i, (images, target) in enumerate(dataloader):
        images = images.to(args.device)
        target = target.to(args.device)

        output = model(images)

        loss = criterion(output.view(-1), target.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() / len(dataloader)

    return train_loss


def valid(dataloader, model, criterion, args):
    model.eval()

    with torch.no_grad():
        valid_loss = 0.0
        preds, valid_labels = [], []
        for i, (images, target) in enumerate(dataloader):
            bs, num_patches, c, h, w = images.size()

            images = images.to(args.device)
            target = target.to(args.device)

            # dihedral TTA
            images = torch.stack([images, images.flip(-1),
                                  images.flip(-2), images.flip(-1, -2),
                                  images.transpose(-1, -2), images.transpose(-1, -2).flip(-1),
                                  images.transpose(-1, -2).flip(-2), images.transpose(-1, -2).flip(-1, -2)], 1)
            images = images.view(-1, num_patches, c, h, w)

            output = model(images).view(bs, 8, -1).mean(1).view(-1)
            loss = criterion(output, target.float())

            preds.append(output.detach().cpu().numpy())
            valid_labels.append(target.detach().cpu().numpy())
            valid_loss += loss.item() / len(dataloader)

        preds = np.concatenate(preds)
        valid_labels = np.concatenate(valid_labels)

        threshold = utils.find_threshold(y_true=valid_labels, y_pred=preds)
        # threshold = [0.5, 1.5, 2.5, 3.5, 4.5]

        isup_preds = pd.cut(preds, [-np.inf] + list(np.sort(threshold)) + [np.inf], labels=[0, 1, 2, 3, 4, 5])
        score = utils.fast_qwk(isup_preds, valid_labels)
        cm = confusion_matrix(valid_labels, isup_preds)

        return valid_loss, score, cm, threshold


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
        args.n_gpus = torch.cuda.device_count()
        print(f"available cuda: {args.n_gpus}")

    # Setup model
    model = PandaNet(arch=args.arch)
    model_path = os.path.join(configure.MODEL_PATH,
                              f'{args.arch}_fold_{args.fold}_{args.patch_size}_{args.num_patches}.pth')
    if args.resume:
        assert os.path.exists(model_path), "checkpoint does not exist"
        state_dict = torch.load(model_path)
        valid_score = state_dict['valid_score']
        threshold = state_dict['threshold']
        print(f"load model from checkpoint, threshold: {threshold}, valid score: {state_dict['valid_score']:0.3f}")
        model.load_state_dict(state_dict['state_dict'])
        best_score = valid_score
        args.learning_rate = 3e-05
    else:
        best_score = 0.0

    if args.n_gpus > 1:
        model = torch.nn.DataParallel(module=model)
    model.to(args.device)

    # Setup data
    total_batch_size = args.per_gpu_batch_size * args.n_gpus
    train_loader, valid_loader = datasets.get_dataloader(
        fold=args.fold,
        batch_size=total_batch_size,
        num_workers=args.num_workers,
        level=args.level,
        patch_size=args.patch_size,
        num_patches=args.num_patches
    )

    # define loss function (criterion) and optimizer
    if args.loss == "l1":
        criterion = torch.nn.L1Loss()
    elif args.loss == "mse":
        criterion = torch.nn.MSELoss()
    elif args.loss == "smooth_l1":
        criterion = torch.nn.SmoothL1Loss()

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    """ Train the model """
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_prefix = f'{current_time}_{args.arch}_fold_{args.fold}_{args.patch_size}_{args.num_patches}'
    log_dir = os.path.join(configure.TRAINING_LOG_PATH,
                           log_prefix)

    tb_writer = None
    if args.log:
        tb_writer = SummaryWriter(log_dir=log_dir)

    print(f'training started: {current_time}')
    for epoch in range(args.epochs):
        train_loss = train(
            dataloader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            args=args)

        valid_loss, valid_score, valid_cm, threshold = valid(
            dataloader=valid_loader,
            model=model,
            criterion=criterion,
            args=args)

        learning_rate = scheduler.get_lr()[0]
        if args.log:
            tb_writer.add_scalar("learning_rate", learning_rate, epoch)
            tb_writer.add_scalar("Loss/train", train_loss, epoch)
            tb_writer.add_scalar("Loss/valid", valid_loss, epoch)
            tb_writer.add_scalar("Score/valid", valid_score, epoch)

            # Log the confusion matrix as an image summary.
            figure = utils.plot_confusion_matrix(valid_cm, class_names=[0, 1, 2, 3, 4, 5], score=valid_score)
            cm_image = utils.plot_to_image(figure)
            tb_writer.add_image("Confusion Matrix valid", cm_image, epoch)

        if valid_score > best_score:
            best_score = valid_score
            state = {'state_dict': model.module.state_dict(),
                     'train_loss': train_loss,
                     'valid_loss': valid_loss,
                     'valid_score': valid_score,
                     'threshold': np.sort(threshold)}
            torch.save(state, model_path)

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        print(f"epoch:{epoch:02d}, "
              f"train:{train_loss:0.3f}, valid:{valid_loss:0.3f}, "
              f"threshold: {np.sort(threshold)}, "
              f"score:{valid_score:0.3f}, best:{best_score:0.3f}, date:{current_time}")

        scheduler.step()

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    print(f'training finished: {current_time}')

    if args.log:
        tb_writer.close()


if __name__ == "__main__":
    main()
