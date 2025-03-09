#!/bin/bash
#SBATCH --job-name=plm
#SBATCH --partition=gpu-farm
#SBATCH --output=%A.out
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --qos=low_gpu_users
source /home/hwjang/miniforge3/bin/activate af3
source ~/.slurmrc
slurm_start
###############################################
python /home/hwjang/project/alphafold3/run_alphafold_custom.py \
    --run_inference=true \
    --run_data_pipeline=false \
    --jackhmmer_n_cpu 8 \
    --input_dir "json" \
    --output_dir "out_pip" \
    --save_embeddings=true \
###############################################
slurm_end