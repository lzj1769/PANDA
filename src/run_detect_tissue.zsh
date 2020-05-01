#!/usr/local_rwth/bin/zsh

#SBATCH -J detect_tissue
#SBATCH -o ./cluster_out/detect_tissue.txt
#SBATCH -e ./cluster_err/detect_tissue.txt

#SBATCH -t 30:00:00 --mem=180G
#SBATCH -A rwth0455 -c 16

source ~/.zshrc
source ~/miniconda3/bin/activate kaggle

python detect_tissue.py