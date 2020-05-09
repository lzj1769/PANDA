#!/usr/local_rwth/bin/zsh

#SBATCH -J preprocessing
#SBATCH -o ./cluster_out/preprocessing.txt
#SBATCH -e ./cluster_err/preprocessing.txt

#SBATCH -t 30:00:00 --mem=180G
#SBATCH -A rwth0455 -c 16

source ~/.zshrc
source ~/miniconda3/bin/activate kaggle

time python preprocessing.py