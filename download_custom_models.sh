# Download the PTH files from Google Drive
FILEID="1LMltdZlQterWtrMSNxzjDOEUdHHWEXFl"

# activat enev
conda activate dinov2_py310

# download using gdown
python -m gdown "https://drive.google.com/uc?id=$FILEID" -O /users/mchakra3/scratch/dinov2_data/pretrained/base_dinov2_vits_in100_200ep.pth
