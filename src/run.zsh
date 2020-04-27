#!/usr/local_rwth/bin/zsh

#SBATCH -J run
#SBATCH -o run.txt

#SBATCH -t 120:00:00 --mem=60G
#SBATCH --gres=gpu:1 -A rwth0455 -c 24

module load cuda
source ~/.zshrc
source ~/miniconda3/bin/activate kaggle

python train.py \
--pretrained \
--epochs 200 \
--batch_size 48 \
--num_workers 24