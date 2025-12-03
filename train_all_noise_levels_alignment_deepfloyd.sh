#!/bin/bash

# Meta script to sweep noisy ImageNet-100 alignment runs using DeepFloyd IF targets.
# Sweeps Gaussian noise levels, DeepFloyd stages, extraction blocks, and DINO layers.

export PYTHONPATH=/root/dinov2:$PYTHONPATH

CONFIG_FILE="dinov2/configs/train/vits16_alignment_4gpus.yaml"
NOISE_LEVELS=(100 75 50 25)
DEEPFLOYD_STAGES=("I" "II" "III")
DEEPFLOYD_BLOCKS=("first" "mid" "last")
NUM_LAYERS=12
ALIGNMENT_LAYERS=($(seq 0 $((NUM_LAYERS - 1))))

OUTPUT_ROOT="/root/autodl-tmp/exp-out/base_dinov2_in100_alignment_deepfloyd_sweep"

# Stage-specific metadata (update to local weights/dims as needed)
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

run_stage_layer_noise() {
    local stage="$1"
    local block="$2"
    local layer="$3"
    local noise_std="$4"

    local model_path="${DEEPFLOYD_MODEL_PATHS[$stage]}"
    local image_size="${DEEPFLOYD_IMAGE_SIZES[$stage]}"
    local target_dim="${DEEPFLOYD_TARGET_HIDDEN_DIMS[$stage]}"

    if [[ -z "$model_path" || -z "$image_size" || -z "$target_dim" ]]; then
        echo "[WARN] Missing DeepFloyd metadata for stage $stage. Skipping."
        return
    fi

    local dataset_root="/root/autodl-tmp/noisy_mini-imagenet-gauss${noise_std}"
    local run_name="noise${noise_std}_stage${stage}_block${block}_layer${layer}"
    local output_dir="${OUTPUT_ROOT}/${run_name}"

    echo "=========================================="
    echo "Noise: $noise_std | Stage: $stage | Block: $block | Layer: $layer"
    echo "Dataset: $dataset_root"
    echo "Output: $output_dir"
    echo "=========================================="

    torchrun --nproc_per_node=4 \
        dinov2/run/train/local_train.py \
        --config-file "$CONFIG_FILE" \
        --output-dir "$output_dir" \
        --save_frequency 10 \
        --max_to_keep 100 \
        train.dataset_path=ImageNet:split=TRAIN:root=$dataset_root:extra=$dataset_root \
        alignment.target_model=deepfloyd \
        alignment.deepfloyd_stage=$stage \
        alignment.deepfloyd_extraction_block=$block \
        alignment.deepfloyd_model_path="$model_path" \
        alignment.deepfloyd_image_size=$image_size \
        alignment.alignment_depth=$layer \
        alignment.target_hidden_dim=$target_dim

    if [ $? -ne 0 ]; then
        echo "[ERROR] Pretraining failed for $run_name"
        return
    fi

    # Clean up checkpoints except the final one (if present)
    if [ -d "$output_dir" ]; then
        find "$output_dir" -maxdepth 1 -name "model_*.pth" ! -name "model_0149999.pth" -type f -delete
    fi

    local checkpoint_path="${output_dir}/eval/training_149999/teacher_checkpoint.pth"
    local config_path="${output_dir}/config.yaml"

    if [ -f "$checkpoint_path" ] && [ -f "$config_path" ]; then
        local linear_output="${output_dir}/eval/training_149999/linear_noise_${noise_std}"
        torchrun --nproc_per_node=1 \
            dinov2/eval/linear.py \
            --config-file "$config_path" \
            --pretrained-weights "$checkpoint_path" \
            --output-dir "$linear_output" \
            --train-dataset ImageNet:split=TRAIN:root=$dataset_root:extra=$dataset_root \
            --val-dataset ImageNet:split=VAL:root=$dataset_root:extra=$dataset_root
        if [ $? -ne 0 ]; then
            echo "[WARN] Linear probing failed for $run_name"
        fi
    else
        echo "[WARN] Skipping linear probing for $run_name (missing checkpoint/config)"
    fi
}

for stage in "${DEEPFLOYD_STAGES[@]}"; do
    for block in "${DEEPFLOYD_BLOCKS[@]}"; do
        for layer in "${ALIGNMENT_LAYERS[@]}"; do
            for noise_std in "${NOISE_LEVELS[@]}"; do
                run_stage_layer_noise "$stage" "$block" "$layer" "$noise_std"
            done
        done
    done
done

echo "DeepFloyd alignment sweep complete!"
