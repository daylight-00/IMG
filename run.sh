#!/bin/bash
#SBATCH -J IMG_training         # job name
#SBATCH -o %A.log               # log file name
#SBATCH -c 8                    # num of cpus
#SBATCH --gpus-per-node=1       # num of gpus
#SBATCH -p g4090_short                                  # 1. Change to your partition

source ~/.bashrc

CONFIG_FILES=(
    "config.py"                                         # 2. Change to your config file path
    )

for CONFIG in "${CONFIG_FILES[@]}"; do
    echo "Running with config: $CONFIG"
    srun python ~/project/IMG/code/train.py $CONFIG     # 3. Change to your train.py path
done