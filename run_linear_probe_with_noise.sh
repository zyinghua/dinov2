#!/bin/bash
#SBATCH -J linear_probe_noise
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o linear_probe_noise.out
#SBATCH -e linear_probe_noise.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
trap 'echo "ERROR at line $LINENO"; exit 1' ERR

# Files
REPO_ROOT="/users/mchakra3/data/mchakra3/dinov2"
ROOT="/users/mchakra3/scratch/dinov2_data/imagenet100/mini-imagenet"
PRETRAINED="/users/mchakra3/scratch/dinov2_data/pretrained/teacher_checkpoint.pth"
CONFIG="$REPO_ROOT/dinov2/configs/eval/vits16_pretrain.yaml"
NOISE_STD=${1:-0.2}  # Default to 0.1, override with argument
OUTPUT="$ROOT/linear16_50ep_128bs_noise_${NOISE_STD}"

# Enrviornment setup 
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Load right version of cuda
module unload cuda        
module load cuda/11.8.0-lpttyok

# Check gpu
echo "CUDA setup check:"
nvidia-smi
python -c "import torch; print('Torch CUDA available:', torch.cuda.is_available()); print('Torch CUDA version:', torch.version.cuda)"

# Create outpit dir
mkdir -p "$OUTPUT"

echo "Running Linear Probe with Noise (DINOv2 Evaluation)"
echo "Repository Root: $REPO_ROOT"
echo "Config: $CONFIG"
echo "Pretrained Weights: $PRETRAINED"
echo "Dataset Root: $ROOT"
echo "Noise Std Dev: $NOISE_STD"
echo "Output Directory: $OUTPUT"

# Run dion with noise
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

echo "Done! Results saved to: $OUTPUT"
