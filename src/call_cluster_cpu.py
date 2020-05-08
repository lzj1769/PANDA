import subprocess

for tile_size in [128, 256]:
    for level in [0, 1]:
        job_name = f"preprocessing_{level}_{tile_size}"
        subprocess.run(["sbatch", "-J", job_name,
                        "-o", f"./cluster_out/{job_name}.txt",
                        "-e", f"./cluster_err/{job_name}.txt",
                        "--time", "120:00:00",
                        "--mem", "60G",
                        "-A", "rwth0455",
                        "run_preprocessing.zsh", str(level), str(tile_size)])
