#!/usr/local_rwth/bin/zsh

source ~/.zshrc
source ~/miniconda3/bin/activate kaggle

python preprocessing.py --level $1 --tile_size $2