import subprocess

model_list = ['se_resnext50_32x4d']
fold_list = [0]

batch_size = {'se_resnext50_32x4d': 6,
              'se_resnext101_32x4d': 4}

level = 1
tile_size = 256
num_tiles = 12

for model in model_list:
    for fold in fold_list:
        job_name = f"{model}_{level}_{tile_size}_{num_tiles}_fold_{fold}"
        subprocess.run(["sbatch", "-J", job_name,
                        "-o", f"./cluster_out/{job_name}.txt",
                        "-e", f"./cluster_err/{job_name}.txt",
                        "--time", "48:00:00",
                        "--mem", "180G",
                        "-c", "8",
                        "-A", "rwth0455",
                        "--gres", "gpu:1",
                        "run.zsh", model,
                        str(level), str(tile_size), str(num_tiles),
                        str(fold), str(batch_size[model])])
