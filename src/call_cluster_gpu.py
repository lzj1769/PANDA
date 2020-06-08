import subprocess

model_list = ['se_resnext50_32x4d']
fold_list = [0, 1, 2, 3, 4]

batch_size = {'se_resnext50_32x4d': 4,
              'se_resnext101_32x4d': 3,
              'inceptionv4': 6,
              'inceptionresnetv2': 4}

level = 1
patch_size = 128
num_patches = 80

for model in model_list:
    for fold in fold_list:
        job_name = f"{model}_{level}_{patch_size}_{num_patches}_fold_{fold}"
        subprocess.run(["sbatch", "-J", job_name,
                        "-o", f"./cluster_out/{job_name}.txt",
                        "-e", f"./cluster_err/{job_name}.txt",
                        "--time", "12:00:00",
                        "--mem", "180G",
                        "-c", "48",
                        "-A", "rwth0455",
                        "--gres", "gpu:2",
                        "run.zsh", model,
                        str(level), str(patch_size), str(num_patches),
                        str(fold), str(batch_size[model])])
