# Download the PTH files from Google Drive
FILEID="1LMltdZlQterWtrMSNxzjDOEUdHHWEXFl"

# activat enev
conda activate $DINO_ENV

mkdir -p ~/scratch/dinov2_data/pretrained/

# download using gdown
python -m gdown "https://drive.google.com/uc?id=$FILEID" -O ~/scratch/dinov2_data/pretrained/base_dinov2_vits_in100_200ep.pth
