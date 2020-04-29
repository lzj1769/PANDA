#!/usr/local_rwth/bin/zsh

module load cuda
source ~/.zshrc
source ~/miniconda3/bin/activate kaggle

python train.py \
--arch $1 \
--fold $2 \
--image_width 512 \
--image_height 512 \
--pretrained \
--epochs 200 \
--batch_size 16 \
--num_workers 24 \
--log