#!/bin/bash
#SBATCH -J linear_probe_noise_sweep
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH -t 20:00:00
#SBATCH -o linear_probe_noise_sweep.out
#SBATCH -e linear_probe_noise_sweep.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
trap 'echo "ERROR at line $LINENO"; exit 1' ERR

# Files
REPO_ROOT="$HOME/CS2952X/dinov2" # Change this variable based on location of repo
ROOT="$HOME/scratch/dinov2_data/imagenet100/mini-imagenet"
PRETRAINED="$HOME/scratch/dinov2_data/pretrained/teacher_checkpoint.pth"
CONFIG="$REPO_ROOT/dinov2/configs/eval/vits16_pretrain.yaml"
BASE_OUTPUT="$ROOT/linear16_50ep_128bs_noise_sweep"

# Environment setup 
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Load right version of cuda
module unload cuda        
module load cuda/11.8.0-lpttyok

# Check gpu
echo "CUDA setup check:"
nvidia-smi
python -c "import torch; print('Torch CUDA available:', torch.cuda.is_available()); print('Torch CUDA version:', torch.version.cuda)"

# Create base output dir
mkdir -p "$BASE_OUTPUT"

echo "Running Linear Probe Noise Sweep (DINOv2 Evaluation)"
echo "Repository Root: $REPO_ROOT"
echo "Config: $CONFIG"
echo "Pretrained Weights: $PRETRAINED"
echo "Dataset Root: $ROOT"
echo "Base Output Directory: $BASE_OUTPUT"
echo ""

# Loop over noise values from 0.1 to 0.9 in 0.1 increments
for NOISE_STD in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    OUTPUT="$BASE_OUTPUT/noise_${NOISE_STD}"
    mkdir -p "$OUTPUT"
    
    echo "=========================================="
    echo "Running with Noise Std: $NOISE_STD"
    echo "Output: $OUTPUT"
    echo "=========================================="
    
    python -m dinov2.eval.linear \
      --config-file "$CONFIG" \
      --pretrained-weights "$PRETRAINED" \
      --output-dir "$OUTPUT" \
      --train-dataset "ImageNet:root=$ROOT:split=TRAIN" \
      --val-dataset   "ImageNet:root=$ROOT:split=VAL" \
      --epochs 10 \
      --epoch-length 1250 \
      --batch-size 128 \
      --num-workers 4 \
      --train-noise-std "$NOISE_STD"
    
    echo "Completed noise level: $NOISE_STD"
    echo ""
done

echo "=========================================="
echo "All noise levels completed!"
echo "Results saved to: $BASE_OUTPUT"
echo "=========================================="
