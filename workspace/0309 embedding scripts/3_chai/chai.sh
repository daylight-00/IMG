#!/bin/bash
#SBATCH --job-name=plm
#SBATCH --partition=gpu-farm
#SBATCH --output=%A.out
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --qos=low_gpu_users
source /home/hwjang/miniforge3/bin/activate chai
source ~/.slurmrc
slurm_start
###############################################
python run_chai.py
###############################################
slurm_end