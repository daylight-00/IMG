#!/bin/bash
#SBATCH --job-name=plm
#SBATCH --partition=gpu-farm
#SBATCH --output=%A.out
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --qos=low_gpu_users
source /home/hwjang/miniforge3/bin/activate boltz
source ~/.slurmrc
slurm_start
###############################################
python /home/hwjang/project/boltz/src/boltz/main_test_click.py predict \
    /home/hwjang/project/boltz/src/boltz/test.fasta \
    --out_dir output \
    --cache /home/hwjang/project/boltz/cache \
    --use_msa_server
###############################################
slurm_end