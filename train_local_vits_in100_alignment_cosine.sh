#!/bin/bash
#SBATCH --job-name=dino-poc
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=100G
#SBATCH --output=dino-cka-%j.out
#SBATCH --error=dion-cka-%j.err


# Training ViT-S on ImageNet-100 with WandB logging
export PYTHONPATH=/root/dinov2:$PYTHONPATH

# Set your WandB entity (optional)
# export WANDB_ENTITY="your-entity-name"

python dinov2/run/train/local_train.py \
    --config-file dinov2/configs/train/vits16_alignment_cosine.yaml \
    --output-dir /cs/data/people/hnam16/base_dinov2_in100_alignment_output_cosine \
    --save_frequency 10 \
    --max_to_keep 3 \
    --use-wandb \
    --wandb-project dinov2-alignment \
    train.dataset_path=ImageNet:split=TRAIN:root=/cs/data/people/hnam16/dinov2_data/imagenet100/mini-imagenet:extra=/cs/data/people/hnam16/dinov2_data/imagenet100/mini-imagenet