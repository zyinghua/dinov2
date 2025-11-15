# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import random


def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    # dtype = torch.half  # TODO: Remove

    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

    collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

    # Collate raw full images for alignment (if available)
    # Apply DiT-style center cropping to get 256x256 images (matching DiT training preprocessing)
    if "full_image_raw" in samples_list[0][0]:
        full_images_list = [s[0]["full_image_raw"] for s in samples_list]
        target_size = 256  # DiT standard size for alignment
        
        def center_crop_tensor(img_tensor, image_size):
            """
            DiT-style center cropping for tensors (adapted from DiT's center_crop_arr).
            Args:
                img_tensor: (C, H, W) tensor in range [0, 1] or [0, 255]
                image_size: Target size (256)
            Returns:
                Center-cropped (C, image_size, image_size) tensor
            """
            C, H, W = img_tensor.shape
            min_dim = min(H, W)
            
            # Step 1: Downscale if min dimension >= 2 * image_size
            current_img = img_tensor
            while min_dim >= 2 * image_size:
                # Resize by half
                new_h, new_w = H // 2, W // 2
                current_img = F.interpolate(
                    current_img.unsqueeze(0),
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                H, W = new_h, new_w
                min_dim = min(H, W)
            
            # Step 2: Scale so min dimension = image_size
            scale = image_size / min_dim
            new_h = round(H * scale)
            new_w = round(W * scale)
            current_img = F.interpolate(
                current_img.unsqueeze(0),
                size=(new_h, new_w),
                mode='bicubic',
                align_corners=False
            ).squeeze(0)
            
            # Step 3: Center crop to image_size x image_size
            _, H, W = current_img.shape
            crop_y = (H - image_size) // 2
            crop_x = (W - image_size) // 2
            cropped = current_img[:, crop_y:crop_y + image_size, crop_x:crop_x + image_size]
            
            return cropped
        
        # Apply center cropping to each image
        cropped_images = []
        for img in full_images_list:
            cropped_img = center_crop_tensor(img, target_size)
            cropped_images.append(cropped_img)
        
        collated_full_images_raw = torch.stack(cropped_images)
    else:
        collated_full_images_raw = None

    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    result = {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
    
    # Add raw full images for alignment if available (will be preprocessed in training)
    if collated_full_images_raw is not None:
        result["collated_full_images_raw"] = collated_full_images_raw  # Keep as float32 for preprocessing
    
    return result
