import os

model_list = ['efficientnet-b3', 'efficientnet-b4',
              'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']
model_list = ['efficientnet-b3']
fold_list = [0]

for model in model_list:
    for fold in fold_list:
        job_name = "{}_fold_{}".format(model, fold)
        command = "sbatch -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
                  "./cluster_err/" + job_name + "_err.txt -t 100:00:00 --mem 90G -A rwth0455 "
        command += "--partition=c18g -c 24 --gres=gpu:1 run.zsh"
        os.system(command + " " + model + " " + str(fold))
