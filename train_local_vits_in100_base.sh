# Using torch.distributed.launch (deprecated, but still works)
# python -m torch.distributed.launch --nproc_per_node=4 \
#     dinov2/run/train/local_train.py \
#     --config-file dinov2/configs/train/vits14.yaml \
#     --output-dir ./output \
#     train.dataset_path=ImageNet:split=TRAIN:root=/path/to/imagenet

# Using torchrun (recommended) - Training ViT-S on ImageNet-100
export PYTHONPATH=/root/dinov2:$PYTHONPATH
torchrun --nproc_per_node=1 \
    dinov2/run/train/local_train.py \
    --config-file dinov2/configs/train/vits16.yaml \
    --output-dir /root/autodl-tmp/exp-out/base_dinov2_in100_output \
    --save_frequency 10 \
    --max_to_keep 10 \
    train.dataset_path=ImageNet:split=TRAIN:root=/root/autodl-tmp/mini-imagenet:extra=/root/autodl-tmp/mini-imagenet