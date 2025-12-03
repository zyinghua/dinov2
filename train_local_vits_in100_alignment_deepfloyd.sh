#!/bin/bash

# Sweep DINOv2 alignment experiments with DeepFloyd IF targets (single GPU variant)
# Loops over DeepFloyd stages, extraction blocks, and DINOv2 alignment layers.

export PYTHONPATH=/root/dinov2:$PYTHONPATH

CONFIG_FILE="dinov2/configs/train/vits16_alignment.yaml"
OUTPUT_ROOT="/cs/data/people/hnam16/base_dinov2_in100_alignment_deepfloyd_output"
DATASET_ROOT="/cs/data/people/hnam16/dinov2_data/imagenet100/mini-imagenet"
NUM_LAYERS=12  # ViT-S has 12 transformer blocks

DEEPFLOYD_STAGES=("I" "II" "III")
DEEPFLOYD_BLOCKS=("first" "mid" "last")
ALIGNMENT_LAYERS=($(seq 0 $((NUM_LAYERS - 1))))

# Map each stage to its default HuggingFace repo (update to local paths if desired)
declare -A DEEPFLOYD_MODEL_PATHS=(
    ["I"]="DeepFloyd/IF-I-XL-v1.0"
    ["II"]="DeepFloyd/IF-II-L-v1.0"
    ["III"]="DeepFloyd/IF-III-L-v1.0"
)

# Stage-specific resize resolutions
declare -A DEEPFLOYD_IMAGE_SIZES=(
    ["I"]=64
    ["II"]=256
    ["III"]=1024
)

# Estimated hidden dims for each stage/block combination.
# Adjust to match the actual checkpoint architecture if needed.
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

    HF_HOME=/cs/data/people/hnam16/hf_home
    TRANSFORMERS_CACHE=/cs/data/people/hnam16/hf_home

    echo $HF_HOME 

    torchrun --nproc_per_node=1 \
        dinov2/run/train/local_train.py \
        --config-file "$CONFIG_FILE" \
        --output-dir "$output_dir" \
        --save_frequency 10 \
        --max_to_keep 3 \
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
