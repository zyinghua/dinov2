# DINOv2 Repository General Guide

> **Note:** This repository inherits the original DINOv2 repository. For full information about DINOv2, please visit the [original DINOv2 repository](https://github.com/facebookresearch/dinov2).

This guide explains how to set up the environment and train a DINOv2 model from scratch.

## Step 1: Environment Setup

### 1.1 Create Conda Environment

Create a new conda environment with Python 3.9:

```bash
conda create -n dinov2 python=3.9 -y
conda activate dinov2
```

or 

```bash
conda create -p /root/dinov2 python=3.9 -y
conda activate dinov2
```

**Note:** Python 3.9 is required due to compatibility with the codebase (type hints use `Optional` instead of `|` syntax).

### 1.2 Install Dependencies

Install the required packages:

```bash
cd /path/to/dinov2
pip install -r envs/requirements.txt
```

The `requirements.txt` includes:
- PyTorch (with CUDA support)
- torchvision
- xformers
- fvcore
- iopath
- omegaconf
- torchmetrics
- And other dependencies

### 1.3 Verify Installation

Check that PyTorch can see your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

## Step 2: Prepare Your Dataset

### 2.1 Dataset Structure

Your dataset should have this structure:

```
dataset_name/
├── root/              # Main dataset directory
│   ├── train/         # Training images
│   │   ├── class_id_1/
│   │   │   ├── class_id_1_0.png
│   │   │   ├── class_id_1_1.png
│   │   │   └── ...
│   │   ├── class_id_2/
│   │   └── ...
│   ├── val/           # Validation images (optional)
│   └── labels.txt     # CSV file with class_id,class_name pairs
└── extra/             # Metadata directory (will be created)
```

The `labels.txt` file should contain:
```
n01440764,tench
n01443537,goldfish
...
```

### 2.2 Generate Metadata

The dataset requires metadata files (`.npy` files) to be generated. Create a script `process_metadata.py`:

```python
from dinov2.data.datasets import ImageNet

# Replace these paths with your actual dataset paths
ROOT_PATH = "/path/to/dataset/root"
EXTRA_PATH = "/path/to/dataset/extra"

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root=ROOT_PATH, extra=EXTRA_PATH)
    dataset.dump_extra()
```

Run the metadata generation:

```bash
python process_metadata.py
```

This will create files in the `extra/` directory:
- `entries-TRAIN.npy`
- `entries-VAL.npy` (if you have validation data)
- `class-ids-TRAIN.npy`
- `class-names-TRAIN.npy`

### 2.3 Update Dataset Length (if needed - e.g., for ImageNet-100)

If your dataset has a different number of images, update the split lengths in `dinov2/data/datasets/image_net.py`:

Find the `length` property in the `_Split` class (around line 35) and update it:

```python
@property
def length(self) -> int:
    # For your custom dataset, update these numbers
    split_lengths = {
       _Split.TRAIN: 50000,  # Update with your actual training set size
       _Split.VAL: 5000,     # Update with your actual validation set size
       _Split.TEST: 10000,   # Update if you have a test set
    }
    return split_lengths[self]
```

## Step 3: Configure Training

### 3.1 Config File Structure

DINOv2 uses a hierarchical config system:
- **Base config**: `dinov2/configs/ssl_default_config.yaml` - Contains all default settings
- **Training configs**: `dinov2/configs/train/*.yaml` - Training-specific overrides (e.g., `vits16.yaml`)

The config files are merged, with training configs overriding base config values.

### 3.2 Training Config Files

Training configs are located in `dinov2/configs/train/`:

- `vits16.yaml` - ViT-S/16 configuration
- `vitl14.yaml` - ViT-L/14 configuration
- `vitg14.yaml` - ViT-G/14 configuration
- etc.

Example training config (`dinov2/configs/train/vits16.yaml`):

```yaml
train:
  batch_size_per_gpu: 10
student:
  arch: vit_small
  patch_size: 16
  drop_path_rate: 0
optim:
  epochs: 200
evaluation:
  eval_period_iterations: 6250
```

### 3.3 Base Config

The base config (`dinov2/configs/ssl_default_config.yaml`) contains:
- Model architecture details
- Training hyperparameters (learning rate, weight decay, etc.)
- Data augmentation settings (crop sizes, scales)
- Optimizer settings
- FSDP configuration

### 3.4 Override Config Values

You can override config values via command line using the `key=value` syntax:

```bash
train.batch_size_per_gpu=20 \
optim.epochs=100 \
train.OFFICIAL_EPOCH_LENGTH=1250
```

## Step 4: Create Training Script

### 4.1 Basic Training Script

Create a training script (e.g., `train_local_vits_in100_base.sh`):

```bash
#!/bin/bash

# Set PYTHONPATH to find the dinov2 module
export PYTHONPATH=/path/to/dinov2:$PYTHONPATH

# Training ViT-S on ImageNet-100
torchrun --nproc_per_node=1 \
    dinov2/run/train/local_train.py \
    --config-file dinov2/configs/train/vits16.yaml \
    --output-dir /path/to/output/directory \
    --save_frequency 10 \
    --max_to_keep 10 \
    train.dataset_path=ImageNet:split=TRAIN:root=/path/to/dataset/root:extra=/path/to/dataset/extra
```

### 4.2 Script Parameters

- `--nproc_per_node=1`: Number of GPUs per node (change to 4 for 4 GPUs)
- `--config-file`: Path to training config file
- `--output-dir`: Directory to save checkpoints and logs
- `--save_frequency`: Save checkpoint every N "official epochs" (each epoch = 1250 iterations)
- `--max_to_keep`: Maximum number of checkpoints to keep
- `train.dataset_path`: Dataset path in format `ImageNet:split=TRAIN:root=<root>:extra=<extra>`

### 4.3 Multi-GPU Training

For multi-GPU training, change `--nproc_per_node`:

```bash
torchrun --nproc_per_node=4 \
    dinov2/run/train/local_train.py \
    ...
```

### 4.4 Make Script Executable

```bash
chmod +x train_local_vits_in100_base.sh
```

## Step 5: Start Training

### 5.1 Run Training Script

```bash
./train_local_vits_in100_base.sh
```

Or run directly:

```bash
export PYTHONPATH=/path/to/dinov2:$PYTHONPATH
torchrun --nproc_per_node=1 \
    dinov2/run/train/local_train.py \
    --config-file dinov2/configs/train/vits16.yaml \
    --output-dir /path/to/output \
    --save_frequency 10 \
    --max_to_keep 10 \
    train.dataset_path=ImageNet:split=TRAIN:root=/path/to/root:extra=/path/to/extra
```

### 5.2 Training Progress

You should see output like:

```
Training  [   540/250000]  eta: 6:41:50  lr: 0.0001 (0.0000)  ...
```

- `[540/250000]`: Current iteration / Total iterations
- Total iterations = `epochs × OFFICIAL_EPOCH_LENGTH` (e.g., 200 × 1250 = 250,000)
- `OFFICIAL_EPOCH_LENGTH`: Fixed iteration count per "epoch" for scheduling (default: 1250)
- Learning rate schedule is based on iterations, not dataset size

### 5.3 Resuming Training

By default, training will automatically attempt to resume from the latest checkpoint in the output directory. The behavior depends on:

1. **If `MODEL.WEIGHTS` is specified** (non-empty path):
   - Loads from that specific checkpoint path

2. **If `MODEL.WEIGHTS` is empty** (default: `''`) and `resume=True` (default):
   - Checks for a checkpoint in the `output_dir` directory
   - If a checkpoint exists, resumes from it
   - If no checkpoint exists, starts training from scratch (no error)

To resume from a specific checkpoint, specify it in the config file:

**In `dinov2/configs/ssl_default_config.yaml` or your training config file:**

```yaml
MODEL:
  WEIGHTS: '/path/to/checkpoint.pth'  # Path to checkpoint file
```

Or override via command line:

```bash
torchrun ... MODEL.WEIGHTS=/path/to/checkpoint.pth ...
```

## Step 6: Understanding Training Configuration

### 6.1 Iteration-Based Training

DINOv2 uses **iteration-based training**, not epoch-based:
- `OFFICIAL_EPOCH_LENGTH = 1250`: Fixed iterations per "official epoch"
- Training runs for `epochs × OFFICIAL_EPOCH_LENGTH` iterations
- This is independent of dataset size or batch size
- With infinite sampler, dataset wraps around as needed

### 6.2 Batch Size and Dataset Coverage

- **Samples seen per iteration**: `batch_size_per_gpu`
- **Samples seen per "epoch"**: `1250 × batch_size_per_gpu`
- **Dataset coverage**: `(1250 × batch_size) / dataset_size`

Example:
- Dataset: 50,000 images
- Batch size: 10
- Samples per "epoch": 1250 × 10 = 12,500
- Coverage: 12,500 / 50,000 = 25% of dataset per "epoch"


## Additional Resources

- Config files: `dinov2/configs/`
- Training code: `dinov2/train/train.py`
- Model architectures: `dinov2/models/`
