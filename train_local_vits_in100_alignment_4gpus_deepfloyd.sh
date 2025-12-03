#!/bin/bash

# Sweep DINOv2 alignment experiments with DeepFloyd IF targets (4-GPU variant)
# Mirrors train_local_vits_in100_alignment_4gpus.sh but sweeps IF stages/blocks/layers.

export PYTHONPATH=/root/dinov2:$PYTHONPATH

CONFIG_FILE="dinov2/configs/train/vits16_alignment_4gpus.yaml"
OUTPUT_ROOT="/root/autodl-tmp/exp-out/base_dinov2_in100_alignment_4gpus_deepfloyd_output"
DATASET_ROOT="/root/autodl-tmp/mini-imagenet"
NUM_LAYERS=12

DEEPFLOYD_STAGES=("I" "II" "III")
DEEPFLOYD_BLOCKS=("first" "mid" "last")
ALIGNMENT_LAYERS=($(seq 0 $((NUM_LAYERS - 1))))

declare -A DEEPFLOYD_MODEL_PATHS=(
    ["I"]="DeepFloyd/IF-I-XL-v1.0"
    ["II"]="DeepFloyd/IF-II-L-v1.0"
    ["III"]="DeepFloyd/IF-III-L-v1.0"
)

declare -A DEEPFLOYD_IMAGE_SIZES=(
    ["I"]=64
    ["II"]=256
    ["III"]=1024
)

declare -A DEEPFLOYD_TARGET_HIDDEN_DIMS=(
    ["I"]=2560
    ["II"]=1536
    ["III"]=1024
)

run_experiment() {
    local stage="$1"
    local block="$2"
    local layer="$3"

    local model_path="${DEEPFLOYD_MODEL_PATHS[$stage]}"
    local image_size="${DEEPFLOYD_IMAGE_SIZES[$stage]}"
    local target_dim="${DEEPFLOYD_TARGET_HIDDEN_DIMS[$stage]}"

    if [[ -z "$model_path" || -z "$image_size" || -z "$target_dim" ]]; then
        echo "[WARN] Missing DeepFloyd metadata for stage $stage. Skipping."
        return
    fi

    local run_name="stage${stage}_block${block}_layer${layer}"
    local output_dir="${OUTPUT_ROOT}/${run_name}"

    echo "=========================================="
    echo "Stage: $stage | Block: $block | Layer: $layer"
    echo "Output: $output_dir"
    echo "=========================================="

    torchrun --nproc_per_node=4 \
        dinov2/run/train/local_train.py \
        --config-file "$CONFIG_FILE" \
        --output-dir "$output_dir" \
        --save_frequency 10 \
        --max_to_keep 100 \
        train.dataset_path=ImageNet:split=TRAIN:root=$DATASET_ROOT:extra=$DATASET_ROOT \
        alignment.target_model=deepfloyd \
        alignment.deepfloyd_stage=$stage \
        alignment.deepfloyd_extraction_block=$block \
        alignment.deepfloyd_model_path="$model_path" \
        alignment.deepfloyd_image_size=$image_size \
        alignment.alignment_depth=$layer \
        alignment.target_hidden_dim=$target_dim
}

for stage in "${DEEPFLOYD_STAGES[@]}"; do
    for block in "${DEEPFLOYD_BLOCKS[@]}"; do
        for layer in "${ALIGNMENT_LAYERS[@]}"; do
            run_experiment "$stage" "$block" "$layer"
        done
    done
done
