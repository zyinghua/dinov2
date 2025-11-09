#!/bin/bash

# Usage: ./linear_probing_in100.sh 249999 /root/autodl-tmp/exp-out/base_dinov2_in100_output

STEP=$1
OUTPUT=$2

# Feel free to comment this out
export PYTHONPATH=/root/dinov2:$PYTHONPATH

torchrun --nproc_per_node=1 \
    dinov2/eval/linear.py \
	--config-file $OUTPUT/config.yaml \
    --pretrained-weights $OUTPUT/eval/training_$STEP/teacher_checkpoint.pth \
    --output-dir $OUTPUT/eval/training_$STEP/linear \
	--train-dataset ImageNet:split=TRAIN:root=/root/autodl-tmp/mini-imagenet:extra=/root/autodl-tmp/mini-imagenet \
    --val-dataset ImageNet:split=VAL:root=/root/autodl-tmp/mini-imagenet:extra=/root/autodl-tmp/mini-imagenet
