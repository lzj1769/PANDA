import os

model_list = ['efficientnet-b3', 'efficientnet-b4',
              'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']
model_list = ['efficientnet-b5']
fold_list = [0, 1, 2, 3, 4]

batch_size = {'efficientnet-b2': 24,
              'efficientnet-b3': 24,
              'efficientnet-b5': 8,
              'se_resnext50_32x4d': 24}

for model in model_list:
    for fold in fold_list:
        job_name = "{}_fold_{}".format(model, fold)
        command = f"sbatch -J {job_name} -o ./cluster_out/{job_name}.txt -e ./cluster_err/{job_name}.txt \
        -t 20:00:00 --mem 90G -A rwth0455 --partition=c18g -c 24 --gres=gpu:1 run.zsh"
        os.system(command + " " + model + " " + str(fold) + " " + str(batch_size[model]))
