# Training ViT-S on ImageNet-100
export PYTHONPATH=/root/dinov2:$PYTHONPATH
torchrun --nproc_per_node=1 \
    dinov2/run/train/local_train.py \
    --config-file dinov2/configs/train/vits16_alignment.yaml \
    --output-dir /cs/data/people/hnam16/base_dinov2_in100_alignment_output \
    --save_frequency 10 \
    --max_to_keep 3 \
    train.dataset_path=ImageNet:split=TRAIN:root=/cs/data/people/hnam16/dinov2_data/imagenet100/mini-imagenet:extra=/cs/data/people/hnam16/dinov2_data/imagenet100/mini-imagenet