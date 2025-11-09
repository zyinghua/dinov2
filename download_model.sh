#!/bin/bash
#SBATCH -J download_model       # Job name
#SBATCH -p gpu                  # GPU partition
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH -N 1                    # Number of nodes
#SBATCH -n 1                    # Number of tasks (one per GPU)
#SBATCH --cpus-per-task=4       # Number of CPU cores per task
#SBATCH --mem=16G               # Memory per node
#SBATCH -t 1:00:00              # Time limit (1 hour)
#SBATCH -o download_model.out   # Standard output file
#SBATCH -e download_model.err   # Standard error file
#SBATCH --mail-type=END,FAIL    # Email notifications
#SBATCH --mail-user=manav_chakravarthy@brown.edu

# Load modules
module load cuda cudnn

# Activate virtual environment
conda activate $DINO_ENV

# run the download_model.py script
python download_model.py

echo "Model download completed!"
