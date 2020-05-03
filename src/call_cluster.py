import subprocess

model_list = ['efficientnet-b3', 'efficientnet-b4',
              'efficientnet-b5']
fold_list = [0]

batch_size = {'se_resnext50_32x4d': 24,
              'se_resnext101_32x4d': 16,
              'efficientnet-b3': 8,
              'efficientnet-b4': 8,
              'efficientnet-b5': 8}

tile_size = 128
num_tiles = 12
task = 'regression'

for model in model_list:
    for fold in fold_list:
        job_name = f"{model}_{task}_fold_{fold}_{tile_size}_{num_tiles}"
        subprocess.run(["sbatch", "-J", job_name,
                        "-o", f"./cluster_out/{job_name}.txt",
                        "-e", f"./cluster_err/{job_name}.txt",
                        "--time", "20:00:00",
                        "--mem", "90G",
                        "-c", "24",
                        "--gres", "gpu:1",
                        "run.zsh", model, str(fold), str(batch_size[model]),
                        str(tile_size), str(num_tiles), task])
