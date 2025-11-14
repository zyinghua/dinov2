"""
Feature extraction pipeline for linear probing DiT representations on ImageNet-1k.

Each image is processed twice (clean and lightly noised) following the evaluation
protocol described in the DiT paper. For a given condition, we encode the image
with the Stable Diffusion VAE, diffuse it to specified timesteps, run DiT, and
pool activations from selected transformer blocks. Features are streamed to disk
so they can be used later by a linear classifier (e.g., logistic regression).
"""
from __future__ import annotations
import PIL.Image as Image
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple
import glob
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

import sys

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from ditmodel import DiT_XL_2
import submitit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DiT features for linear probing.")
    parser.add_argument("--data-dir", type=str, default="../../../scratch/imagenet-1k/val",
                        help="Root directory that contains ImageNet-1k split folders (train/val).")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256,
                        help="Spatial size fed into the Stable Diffusion VAE and DiT.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--layer-indices", nargs="+", type=int, default=[14, 27],
                        help="DiT block indices to probe (0-indexed).")
    parser.add_argument("--time-indices", nargs="+", type=int, default=[1, 101, 201],
                        help="Diffusion timesteps (0-999) used for feature extraction.")
    parser.add_argument("--save-path", type=str, default="../../../scratch/linprobe_DiT_XL_2",
                        help="Directory where feature shards are stored.")
    parser.add_argument("--label_path", type=str, default="ILSVRC2012_validation_ground_truth.txt",
                        help="Path to ImageNet-1k label file.")
    parser.add_argument("--gaussian-noise-std", type=float, default=0.06,
                        help="Std. dev. of pixel-level Gaussian noise added to the noisy copy.")
    parser.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-ema",
                        help="Stable Diffusion VAE identifier.")
    parser.add_argument("--shard-size", type=int, default=2048,
                        help="Number of samples per saved shard file.")
    parser.add_argument("--cfg-scale", type=float, default=1.0,
                        help="Classifier-free guidance scale.")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Optional limit on images per split (useful for debugging).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_slurm", action="store_true",
                        help="If set, submits the job to SLURM via submitit.", default=False)
    return parser.parse_args()


class ImageNetDataset(Dataset):
    def __init__(self, root_dir: str, label_path: str, transform: transforms.Compose):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted(glob.glob(self.root_dir + '/*.JPEG'))
        # open label file and read labels
        self.labels = []
        with open(label_path, 'r') as f:
            for line in f:
                self.labels.append(int(line.strip()))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        image_path = self.image_paths[index]
        image_name = self.image_paths[index].split('/')[-1].split('.')[0]
        image_num = int(image_name.split('_')[-1])
        image_idx = image_num - 1  # image numbers start from 1
        label = self.labels[image_idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label, image_name
    
def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def build_dataloaders(
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    label_path: str
) -> Dict[str, DataLoader]:

    transform = build_transform(image_size)
    
    dataset = ImageNetDataset(
        root_dir=data_dir,
        label_path=label_path,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader


class FeatureShardWriter:
    """
    Streams features/labels to disk to avoid holding the full dataset in memory.
    Supports per-image flushing by forcing a flush after each add().
    """
    def __init__(self, save_dir: str, shard_size: int):
        self.save_dir = save_dir
        self.shard_size = shard_size
        os.makedirs(self.save_dir, exist_ok=True)
        self._feature_buffers: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self._label_buffers: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self._name_buffers: Dict[str, List[str]] = defaultdict(list)
        self._counts_since_flush: Dict[str, int] = defaultdict(int)
        self._shard_ids: Dict[str, int] = defaultdict(int)

    def add(
        self,
        key: str,
        features: torch.Tensor,
        labels: torch.Tensor,
        names: List[str],
        force_flush: bool = False,
    ) -> None:
        self._feature_buffers[key].append(features.detach().cpu())
        self._label_buffers[key].append(labels.detach().cpu())
        self._name_buffers[key].extend(names)
        self._counts_since_flush[key] += features.shape[0]
        if force_flush or self._counts_since_flush[key] >= self.shard_size:
            self._flush_key(key)

    def finalize(self) -> None:
        for key in list(self._feature_buffers.keys()):
            self._flush_key(key)

    def _flush_key(self, key: str) -> None:
        if not self._feature_buffers[key]:
            return
        shard_id = self._shard_ids[key]
        out_path = os.path.join(self.save_dir, f"{key}_part{shard_id:04d}.pt")
        payload = {
            "features": torch.cat(self._feature_buffers[key], dim=0),
            "labels": torch.cat(self._label_buffers[key], dim=0),
            "names": list(self._name_buffers[key]),
        }
        torch.save(payload, out_path)
        self._feature_buffers[key].clear()
        self._label_buffers[key].clear()
        self._name_buffers[key].clear()
        self._counts_since_flush[key] = 0
        self._shard_ids[key] += 1
        print(f"[INFO] Wrote {out_path}")


def register_block_hooks(model: torch.nn.Module, layer_indices: Iterable[int]):
    activations: Dict[int, torch.Tensor] = {}
    handles = []

    for idx in layer_indices:
        block = model.blocks[idx]

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                activations[layer_idx] = output
            return hook

        handles.append(block.register_forward_hook(make_hook(idx)))
    return activations, handles


def diffusion_encode(
    diffusion,
    latents: torch.Tensor,
    timestep: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    t_tensor = torch.full((latents.size(0),), timestep, device=latents.device, dtype=torch.long)
    noise = torch.randn_like(latents)
    noisy_latents = diffusion.q_sample(latents, t_tensor, noise=noise)
    return noisy_latents, t_tensor


def sanitize_key_component(value: str) -> str:
    safe = value.replace(os.sep, "_")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    sanitized = "".join(ch if ch in allowed else "_" for ch in safe)
    return sanitized


def run_dit_with_cfg(
    model: DiT_XL_2,
    latents: torch.Tensor,
    t_tensor: torch.Tensor,
    labels: torch.Tensor,
    cfg_scale: float,
) -> int:
    """
    Runs DiT either conditionally or with classifier-free guidance. Returns the
    effective batch size corresponding to the conditional samples (before CFG duplication).
    """
    batch_size = latents.size(0)

    null_label = torch.full_like(labels, model.y_embedder.num_classes)
    latents_cfg = torch.cat([latents, latents], dim=0)
    t_cfg = torch.cat([t_tensor, t_tensor], dim=0)
    labels_cfg = torch.cat([labels, null_label], dim=0)
    model.forward_with_cfg(latents_cfg, t_cfg, labels_cfg, cfg_scale)
    return batch_size


def extract_features(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    latent_size = args.image_size // 8
    model = DiT_XL_2(input_size=latent_size).to(device)
    state_dict = find_model(f"DiT-XL-2-{args.image_size}x{args.image_size}.pt")
    model.load_state_dict(state_dict)
    model.eval()
    print("DiT model loaded.")

    vae = AutoencoderKL.from_pretrained(args.vae_model).to(device)
    vae.eval()
    print("VAE model loaded.")

    diffusion = create_diffusion(timestep_respacing="")
    loader = build_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_path=args.label_path,
    )
    print("DataLoader created.")

    writer = FeatureShardWriter(args.save_path, args.shard_size)
    metadata_path = os.path.join(args.save_path, "config.json")
    with open(metadata_path, "w") as meta_file:
        json.dump(vars(args), meta_file, indent=2)
    print(f"[INFO] Saved run configuration to {metadata_path}")

    activations, hooks = register_block_hooks(model, args.layer_indices)

    clean_key = "clean"
    noisy_key = "gaussian"
    noise_std = args.gaussian_noise_std

    try:
        with torch.no_grad():
            processed = 0
            progress = tqdm(loader, leave=False)
            for images, labels, names in progress:
                if args.max_images is not None and processed >= args.max_images:
                    break
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                noisy_images = (images + noise_std * torch.randn_like(images)).clamp(-1.0, 1.0)

                clean_latents = vae.encode(images).latent_dist.sample().mul_(0.18215) # why 0.18215? Because Stable Diffusion uses this scaling factor for the latent space which is 1/5.5
                
                noisy_latents = vae.encode(noisy_images).latent_dist.sample().mul_(0.18215)

                latents_dict = {
                    clean_key: clean_latents,
                    noisy_key: noisy_latents,
                }

                for variant, latents in latents_dict.items():
                    for timestep in args.time_indices:
                        noisy_latent, t_tensor = diffusion_encode(diffusion, latents, timestep)
                        effective_batch = run_dit_with_cfg(
                            model=model,
                            latents=noisy_latent,
                            t_tensor=t_tensor,
                            labels=labels,
                            cfg_scale=args.cfg_scale,
                        )
                        for layer_idx in args.layer_indices:
                            features = activations[layer_idx][:effective_batch].mean(dim=1) # effective batch here makes sure we only take the conditional samples
                            unconditional_features = activations[layer_idx][effective_batch:].mean(dim=1) # why mean? Because we want to pool the spatial dimensions, and dim=0 is batch dimension
                            # each feature is of shape (batch_size, feature_dim)
                            for idx, name in enumerate(names):
                                sample_key = f"{sanitize_key_component(name)}_layer{layer_idx:02d}_t{timestep:04d}_{variant}"
                                writer.add(
                                    sample_key,
                                    unconditional_features[idx:idx + 1],
                                    labels[idx:idx + 1],
                                    names=[name],
                                    force_flush=True,
                                )

                processed += images.size(0)
                if args.max_images is not None:
                    progress.set_postfix({"images": f"{processed}/{args.max_images}"})
            print(f"[INFO] Finished ({processed} images).")
    finally:
        writer.finalize()
        for handle in hooks:
            handle.remove()


def main():
    args = parse_args()

    # if args.use_slurm:
    #     try:
    #         executor = submitit.AutoExecutor(folder="logs_slurm")
    #         executor.update_parameters(
    #             mem_gb=32,
    #             gpus_per_node=1,
    #             cpus_per_task=10,
    #             nodes=1,
    #             timeout_min=9 * 60,  
    #             slurm_partition="gpu",
    #             slurm_signal_delay_s=120,
    #         )
    #         job = executor.submit(extract_features, args)
    #         print(job)
    #     except Exception as e:
    #         print("Failed to submit job to slurm, running locally")
    #         print(e)
    #         extract_features(args)s
    # else:
    extract_features(args)


if __name__ == "__main__":
    main()
