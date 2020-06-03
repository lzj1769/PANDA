#!/usr/local_rwth/bin/zsh

#SBATCH -J preprocessing
#SBATCH -o ./cluster_out/preprocessing.txt
#SBATCH -e ./cluster_err/preprocessing.txt

#SBATCH -t 5:00:00 --mem=180G
#SBATCH -A rwth0429 -c 36

source ~/.zshrc
source ~/miniconda3/bin/activate kaggle

python preprocessing.py
