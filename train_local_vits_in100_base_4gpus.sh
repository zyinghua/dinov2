# Training ViT-S on ImageNet-100 (4 GPUs)
export PYTHONPATH=/root/dinov2:$PYTHONPATH
torchrun --nproc_per_node=4 \
    dinov2/run/train/local_train.py \
    --config-file dinov2/configs/train/vits16_4gpus.yaml \
    --output-dir /root/autodl-tmp/exp-out/base_dinov2_in100_output \
    --save_frequency 10 \
    --max_to_keep 10 \
    train.dataset_path=ImageNet:split=TRAIN:root=/root/autodl-tmp/mini-imagenet:extra=/root/autodl-tmp/mini-imagenet