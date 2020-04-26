#!/usr/local_rwth/bin/zsh

#SBATCH -J run
#SBATCH -o run.txt

#SBATCH -t 120:00:00 --mem=60G
#SBATCH --gres=gpu:2 -A rwth0455 -c 16

#module load cuda
#source ~/.zshrc
#source ~/miniconda3/bin/activate kaggle

python train.py \
--pretrained \
--num_train_epochs 3 \
--batch_size 2 \
--learning_rate 3e-05 \
--weight_decay 0.0001 \
--num_workers 16