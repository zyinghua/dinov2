# Download the ImageNet-100 dataset zip file from Google Drive
FILEID="1jJz_IqCFaUD9UGD5sBCgiXyywt66sBlK"

# activat enev
conda activate dinov2_py310

# download using gdown
python -m gdown "https://drive.google.com/uc?id=$FILEID" -O ./dinov2/data/imagenet100.zip

# create target dir
TARGET="/users/mchakra3/scratch/dinov2_data/imagenet100"
mkdir -p "$TARGET"

#unzip data
unzip -q ./dinov2/data/imagenet100.zip -d "$TARGET"

# tar -xzf ./dinov2/data/imagenet100.tar.gz -C "$TARGET"
ls -la "$TARGET"
