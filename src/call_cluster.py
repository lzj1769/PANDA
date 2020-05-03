import os
import subprocess

model_list = ['efficientnet-b3', 'efficientnet-b4',
              'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']
model_list = ['se_resnext50_32x4d']
fold_list = [0, 1, 2, 3, 4]

batch_size = {'se_resnext50_32x4d': 24}

for model in model_list:
    for fold in fold_list:
        job_name = "{}_fold_{}".format(model, fold)
        subprocess.run(["sbatch", "-J", job_name,
                        "-o", f"./cluster_out/{job_name}.txt",
                        "-e", f"./cluster_err/{job_name}.txt",
                        "--time", "20:00:00",
                        "--mem", "90G",
#                        "--account", "rwth0455",
                        "-c", "24",
                        "--gres", "gpu:1",
                        "run.zsh", model, str(fold), str(batch_size[model])])
