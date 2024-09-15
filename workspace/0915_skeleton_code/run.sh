#!/bin/bash
#SBATCH -J IMG_training # job name
#SBATCH -o %A_cross_70.log #output name
#SBATCH -N 1
##SBATCH -n 1
#SBATCH -c 16  # num of cpus
#SBATCH -p g4090_short
#SBATCH --gpus-per-node=1
##SBATCH -w gpu9
source ~/.slurmrc
slurm_start
#####################
source /home/hwjang/miniconda3/bin/activate venv

CONFIG_FILES=(
    "config.py"
    )

for CONFIG in "${CONFIG_FILES[@]}"; do
    echo "Running with config: $CONFIG"
    srun python train.py $CONFIG
done

#####################
slurm_end