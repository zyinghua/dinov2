import torch

checkpoint_path = "/users/mchakra3/scratch/dinov2_data/pretrained/base_dinov2_vits_in100_200ep.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# If the checkpoint contains 'model', extract it first
if 'model' in checkpoint:
    checkpoint = checkpoint['model']


# Remove  teacher and additional prefixes defined in utils/utils.py
teacher_state_dict = {}
for key, value in checkpoint.items():
    if key.startswith('teacher.'):
        new_key = key.replace('teacher.', '').replace('module.', '').replace('backbone.', '')
        teacher_state_dict[new_key] = value
print(f"Extracted {len(teacher_state_dict)} teacher keys")


# Save the teacher-only state dict
out_path = "/users/mchakra3/scratch/dinov2_data/pretrained/base_dinov2_vits_in100_200ep_extracted.pth"
torch.save(teacher_state_dict, out_path)
print(f"Saved teacher state dict to: {out_path}")

# Show sample keys
print("\nSample keys in teacher state dict:")
for i, key in enumerate(list(teacher_state_dict.keys())[:10]):
    print(f"  {key}")