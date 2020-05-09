import subprocess

tile_size = 128
level = 1

job_name = f"preprocessing_{level}_{tile_size}"
subprocess.run(["sbatch", "-J", job_name,
                "-o", f"./cluster_out/{job_name}.txt",
                "-e", f"./cluster_err/{job_name}.txt",
                "--time", "120:00:00",
                "--mem", "180G",
                "-A", "rwth0455",
                "run_preprocessing.zsh", str(level), str(tile_size)])
