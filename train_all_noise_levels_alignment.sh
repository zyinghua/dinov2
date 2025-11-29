#!/bin/bash

# Meta script to train and evaluate DINOv2 alignment models on noisy ImageNet-100 datasets
# Trains on noise levels: 25, 50, 75, 100 + some extra trials
# Each run includes: pretraining + linear probing

export PYTHONPATH=/root/dinov2:$PYTHONPATH

# Noise levels to process
NOISE_LEVELS=(25 50 75 100)

for noise_std in "${NOISE_LEVELS[@]}"; do
    echo "=========================================="
    echo "Processing noise_std=$noise_std"
    echo "=========================================="
    
    DATASET_ROOT="/root/autodl-tmp/noisy_mini-imagenet-gauss${noise_std}"
    OUTPUT_DIR="/root/autodl-tmp/exp-out/base_dinov2_in100_alignment_4gpus_gauss${noise_std}-output"
    
    # Step 1: Pretraining
    echo "Starting pretraining for noise_std=$noise_std..."
    torchrun --nproc_per_node=4 \
        dinov2/run/train/local_train.py \
        --config-file dinov2/configs/train/vits16_alignment_4gpus.yaml \
        --output-dir "$OUTPUT_DIR" \
        --save_frequency 10 \
        --max_to_keep 100 \
        train.dataset_path=ImageNet:split=TRAIN:root=$DATASET_ROOT:extra=$DATASET_ROOT
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Pretraining failed for noise_std=$noise_std"
        continue
    fi
    
    echo "Pretraining completed for noise_std=$noise_std"
    
    # Step 2: Clean up checkpoint files (keep only model_0149999.pth)
    echo "Cleaning up checkpoint files for noise_std=$noise_std..."
    
    # Delete all model pth files except model_0149999.pth in the main output directory
    if [ -d "$OUTPUT_DIR" ]; then
        find "$OUTPUT_DIR" -maxdepth 1 -name "model_*.pth" ! -name "model_0149999.pth" -type f -delete
        echo "Cleaned up model checkpoints in $OUTPUT_DIR"
    fi
    
    # Delete all pth files in eval/ directories except in training_149999
    if [ -d "$OUTPUT_DIR/eval" ]; then
        find "$OUTPUT_DIR/eval" -mindepth 1 -maxdepth 1 -type d ! -name "training_149999" -exec rm -rf {} \;
        echo "Cleaned up eval subdirectories (kept training_149999)"
    fi
    
    echo "Checkpoint cleanup completed for noise_std=$noise_std"
    
    # Step 3: Linear probing
    echo "Starting linear probing for noise_std=$noise_std..."
    
    # Use checkpoint and config from this noise level's own pretraining
    NOISE_CHECKPOINT="${OUTPUT_DIR}/eval/training_149999/teacher_checkpoint.pth"
    NOISE_CONFIG="${OUTPUT_DIR}/config.yaml"
    
    # Check if the checkpoint and config exist
    if [ ! -f "$NOISE_CHECKPOINT" ] || [ ! -f "$NOISE_CONFIG" ]; then
        echo "WARNING: Checkpoint or config not found for noise_std=$noise_std"
        echo "  Checkpoint: $NOISE_CHECKPOINT"
        echo "  Config: $NOISE_CONFIG"
        echo "Skipping linear probing for noise_std=$noise_std"
    else
    
        output_dir="${OUTPUT_DIR}/eval/training_149999/linear_noise_${noise_std}"
        
        torchrun --nproc_per_node=1 \
            dinov2/eval/linear.py \
            --config-file "$NOISE_CONFIG" \
            --pretrained-weights "$NOISE_CHECKPOINT" \
            --output-dir "$output_dir" \
            --train-dataset ImageNet:split=TRAIN:root=$DATASET_ROOT:extra=$DATASET_ROOT \
            --val-dataset ImageNet:split=VAL:root=$DATASET_ROOT:extra=$DATASET_ROOT
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Linear probing failed for noise_std=$noise_std"
        else
            echo "Linear probing completed for noise_std=$noise_std"
        fi
    fi
    
    echo "Completed processing noise_std=$noise_std"
    echo "=========================================="
    echo ""
done

echo "All noise levels processed!"

echo ""
echo "=========================================="
echo "Extra Trial 1: DINOv2 layer 10, DiT layer 14, fixed timestep"
echo "Noise level: 50/255"
echo "=========================================="

NOISE_STD=50
DATASET_ROOT="/root/autodl-tmp/noisy_mini-imagenet-gauss${NOISE_STD}"
OUTPUT_DIR="/root/autodl-tmp/exp-out/base_dinov2_in100_alignment_layer10_dit14_4gpus_gauss${NOISE_STD}-output"
CONFIG_FILE="dinov2/configs/train/vits16_alignment_layer10_dit14_4gpus.yaml"

# Step 1: Pretraining
echo "Starting pretraining for Trial 1 (layer 10, DiT 14)..."
torchrun --nproc_per_node=4 \
    dinov2/run/train/local_train.py \
    --config-file "$CONFIG_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --save_frequency 10 \
    --max_to_keep 100 \
    train.dataset_path=ImageNet:split=TRAIN:root=$DATASET_ROOT:extra=$DATASET_ROOT

if [ $? -ne 0 ]; then
    echo "ERROR: Pretraining failed for Trial 1"
else
    echo "Pretraining completed for Trial 1"
    
    # Step 2: Clean up checkpoint files
    echo "Cleaning up checkpoint files for Trial 1..."
    if [ -d "$OUTPUT_DIR" ]; then
        find "$OUTPUT_DIR" -maxdepth 1 -name "model_*.pth" ! -name "model_0149999.pth" -type f -delete
        echo "Cleaned up model checkpoints in $OUTPUT_DIR"
    fi
    
    if [ -d "$OUTPUT_DIR/eval" ]; then
        find "$OUTPUT_DIR/eval" -mindepth 1 -maxdepth 1 -type d ! -name "training_149999" -exec rm -rf {} \;
        echo "Cleaned up eval subdirectories (kept training_149999)"
    fi
    echo "Checkpoint cleanup completed for Trial 1"
    
    # Step 3: Linear probing
    echo "Starting linear probing for Trial 1..."
    TRIAL1_CHECKPOINT="${OUTPUT_DIR}/eval/training_149999/teacher_checkpoint.pth"
    TRIAL1_CONFIG="${OUTPUT_DIR}/config.yaml"
    
    if [ -f "$TRIAL1_CHECKPOINT" ] && [ -f "$TRIAL1_CONFIG" ]; then
        output_dir="${OUTPUT_DIR}/eval/training_149999/linear_noise_${NOISE_STD}"
        torchrun --nproc_per_node=1 \
            dinov2/eval/linear.py \
            --config-file "$TRIAL1_CONFIG" \
            --pretrained-weights "$TRIAL1_CHECKPOINT" \
            --output-dir "$output_dir" \
            --train-dataset ImageNet:split=TRAIN:root=$DATASET_ROOT:extra=$DATASET_ROOT \
            --val-dataset ImageNet:split=VAL:root=$DATASET_ROOT:extra=$DATASET_ROOT
        
        if [ $? -eq 0 ]; then
            echo "Linear probing completed for Trial 1"
        else
            echo "ERROR: Linear probing failed for Trial 1"
        fi
    else
        echo "WARNING: Skipping linear probing for Trial 1 (checkpoint or config not found)"
        echo "  Expected checkpoint: $TRIAL1_CHECKPOINT"
        echo "  Expected config: $TRIAL1_CONFIG"
    fi
fi

echo "Completed Trial 1"
echo "=========================================="
echo ""


echo "=========================================="
echo "Extra Trial 2: DINOv2 layer 4, DiT last layer, fixed timestep"
echo "Noise level: 50/255"
echo "=========================================="

NOISE_STD=50
DATASET_ROOT="/root/autodl-tmp/noisy_mini-imagenet-gauss${NOISE_STD}"
OUTPUT_DIR="/root/autodl-tmp/exp-out/base_dinov2_in100_alignment_layer4_ditlast_4gpus_gauss${NOISE_STD}-output"
CONFIG_FILE="dinov2/configs/train/vits16_alignment_layer4_ditlast_4gpus.yaml"

# Step 1: Pretraining
echo "Starting pretraining for Trial 2 (layer 4, DiT last)..."
torchrun --nproc_per_node=4 \
    dinov2/run/train/local_train.py \
    --config-file "$CONFIG_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --save_frequency 10 \
    --max_to_keep 100 \
    train.dataset_path=ImageNet:split=TRAIN:root=$DATASET_ROOT:extra=$DATASET_ROOT

if [ $? -ne 0 ]; then
    echo "ERROR: Pretraining failed for Trial 2"
else
    echo "Pretraining completed for Trial 2"
    
    # Step 2: Clean up checkpoint files
    echo "Cleaning up checkpoint files for Trial 2..."
    if [ -d "$OUTPUT_DIR" ]; then
        find "$OUTPUT_DIR" -maxdepth 1 -name "model_*.pth" ! -name "model_0149999.pth" -type f -delete
        echo "Cleaned up model checkpoints in $OUTPUT_DIR"
    fi
    
    if [ -d "$OUTPUT_DIR/eval" ]; then
        find "$OUTPUT_DIR/eval" -mindepth 1 -maxdepth 1 -type d ! -name "training_149999" -exec rm -rf {} \;
        echo "Cleaned up eval subdirectories (kept training_149999)"
    fi
    echo "Checkpoint cleanup completed for Trial 2"
    
    # Step 3: Linear probing
    echo "Starting linear probing for Trial 2..."
    TRIAL2_CHECKPOINT="${OUTPUT_DIR}/eval/training_149999/teacher_checkpoint.pth"
    TRIAL2_CONFIG="${OUTPUT_DIR}/config.yaml"
    
    if [ -f "$TRIAL2_CHECKPOINT" ] && [ -f "$TRIAL2_CONFIG" ]; then
        output_dir="${OUTPUT_DIR}/eval/training_149999/linear_noise_${NOISE_STD}"
        torchrun --nproc_per_node=1 \
            dinov2/eval/linear.py \
            --config-file "$TRIAL2_CONFIG" \
            --pretrained-weights "$TRIAL2_CHECKPOINT" \
            --output-dir "$output_dir" \
            --train-dataset ImageNet:split=TRAIN:root=$DATASET_ROOT:extra=$DATASET_ROOT \
            --val-dataset ImageNet:split=VAL:root=$DATASET_ROOT:extra=$DATASET_ROOT
        
        if [ $? -eq 0 ]; then
            echo "Linear probing completed for Trial 2"
        else
            echo "ERROR: Linear probing failed for Trial 2"
        fi
    else
        echo "WARNING: Skipping linear probing for Trial 2 (checkpoint or config not found)"
        echo "  Expected checkpoint: $TRIAL2_CHECKPOINT"
        echo "  Expected config: $TRIAL2_CONFIG"
    fi
fi

echo "Completed Trial 2"
echo "=========================================="
echo ""
echo "All trials completed!"

