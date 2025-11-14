#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=10
#SBATCH --error=/oscar/home/hnam16/hazel/code/dit_linprobe/logs_slurm/%j_0_log.err
#SBATCH --gpus-per-node=1
#SBATCH --job-name=submitit
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/oscar/home/hnam16/hazel/code/dit_linprobe/logs_slurm/%j_0_log.out
#SBATCH --partition=gpu
#SBATCH --signal=USR2@120
#SBATCH --time=540
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /oscar/home/hnam16/hazel/code/dit_linprobe/logs_slurm/%j_%t_log.out --error /oscar/home/hnam16/hazel/code/dit_linprobe/logs_slurm/%j_%t_log.err /users/hnam16/miniconda3/envs/py312/bin/python -u -m submitit.core._submit /oscar/home/hnam16/hazel/code/dit_linprobe/logs_slurm
