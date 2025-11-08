#!/bin/bash
#SBATCH -J linear_probe
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o linear_probe.out
#SBATCH -e linear_probe.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=manav_chakravarthy@brown.edu

set -euo pipefail
trap 'echo "ERROR at line $LINENO"; exit 1' ERR

# Files
REPO_ROOT="/users/mchakra3/data/mchakra3/dinov2"
ROOT="/users/mchakra3/scratch/dinov2_data/imagenet100/mini-imagenet"
PRETRAINED="/users/mchakra3/scratch/dinov2_data/pretrained/base_dinov2_vits_in100_200ep_extracted.pth"
CONFIG="$REPO_ROOT/dinov2/configs/eval/vits16_pretrain.yaml"
OUTPUT="$ROOT/linear16_50ep_128bs"

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

echo "Running Linear Probe (DINOv2 Evaluation)"
echo "Repository Root: $REPO_ROOT"
echo "Config: $CONFIG"
echo "Pretrained Weights: $PRETRAINED"
echo "Dataset Root: $ROOT"
echo "Output Directory: $OUTPUT"

# Rune dion
python -m dinov2.eval.linear \
  --config-file "$CONFIG" \
  --pretrained-weights "$PRETRAINED" \
  --output-dir "$OUTPUT" \
  --train-dataset "ImageNet:root=$ROOT:split=TRAIN" \
  --val-dataset   "ImageNet:root=$ROOT:split=VAL" \
  --epochs 10 \
  --epoch-length 1250 \
  --batch-size 128 \
  --num-workers 4

echo "Done! Results saved to: $OUTPUT"
