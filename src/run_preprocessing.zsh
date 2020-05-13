#!/usr/local_rwth/bin/zsh

#SBATCH -J preprocessing
#SBATCH -o ./cluster_out/preprocessing.txt
#SBATCH -e ./cluster_err/preprocessing.txt

#SBATCH -t 30:00:00 --mem=900G
#SBATCH -A rwth0429 -c 16 --partition=c16s

source ~/.zshrc
source ~/miniconda3/bin/activate kaggle

time python preprocessing.py