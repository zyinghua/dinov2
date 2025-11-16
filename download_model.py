import torch
from pathlib import Path

# create target dir for pretrained weights
MODEL_PATH = "~/scratch/dinov2_data/pretrained"

out_dir = Path(MODEL_PATH)
out_dir.mkdir(parents=True, exist_ok=True)
out_f = out_dir / "dinov2_vits14_hub.pth"

# load pretrained model
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=True)

# Save state
torch.save(model.state_dict(), out_f)
print("Saved checkpoint to", out_f)
