#!/bin/bash

# Usage: ./linear_probing_in100_noise.sh 249999 /root/autodl-tmp/exp-out/base_dinov2_in100_output

STEP=$1
OUTPUT=$2

# Feel free to comment this out
export PYTHONPATH=/root/dinov2:$PYTHONPATH

for noise_std in 25 50 75; do
    echo "Running linear probing with noise_std=$noise_std"
    
    output_dir="${OUTPUT}/eval/training_${STEP}/linear_noise_${noise_std}"
    
    torchrun --nproc_per_node=1 \
        dinov2/eval/linear.py \
        --config-file $OUTPUT/config.yaml \
        --pretrained-weights $OUTPUT/eval/training_${STEP}/teacher_checkpoint.pth \
        --output-dir $output_dir \
        --train-dataset ImageNet:split=TRAIN:root=/root/autodl-tmp/noisy_mini-imagenet-gauss${noise_std}:extra=/root/autodl-tmp/noisy_mini-imagenet-gauss${noise_std} \
        --val-dataset ImageNet:split=VAL:root=/root/autodl-tmp/noisy_mini-imagenet-gauss${noise_std}:extra=/root/autodl-tmp/noisy_mini-imagenet-gauss${noise_std} \
    
    echo "Completed linear probing with noise_std=$noise_std"
    echo "----------------------------------------"
done
