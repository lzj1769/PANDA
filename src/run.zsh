#!/usr/local_rwth/bin/zsh

module load cuda
source ~/.zshrc
source ~/miniconda3/bin/activate kaggle

python train.py \
--arch $1 \
--fold $2 \
--batch_size $3 \
--pretrained \
--log