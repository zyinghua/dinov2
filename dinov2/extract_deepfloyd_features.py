"""
Module to extract DeepFloyd IF intermediate features for alignment with DINOv2.
The implementation mirrors the DiT feature extractor but loads IF pipelines
directly from diffusers for flexibility across stages.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F


def _import_if_pipelines():
    try:
        from diffusers import IFPipeline, IFImg2ImgPipeline, IFSuperResolutionPipeline
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "DeepFloydFeatureExtractor requires diffusers. "
            "Install diffusers to enable DeepFloyd alignment."
        ) from exc
    return {
        "i": IFPipeline,
        "ii": IFImg2ImgPipeline,
        "iii": IFSuperResolutionPipeline,
    }


def requires_grad(module: torch.nn.Module, flag: bool = False):
    for p in module.parameters():
        p.requires_grad = flag


class DeepFloydFeatureExtractor:
    """
    Extracts intermediate features from pretrained DeepFloyd IF pipelines.
    """

    def __init__(
        self,
        model_path: str,
        stage: str = "I",
        variant: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = torch.float16,
        extraction_block: Union[int, str] = "mid",
        timestep: float = 0.0,
        unconditional_prompt: str = "",
        image_size: Optional[int] = None,
        device: str = "cuda",
    ):
        pipelines = _import_if_pipelines()
        stage_key = stage.lower()
        if stage_key not in pipelines:
            raise ValueError(
                f"Unknown DeepFloyd stage '{stage}'. Expected one of {list(pipelines.keys())}."
            )

        pipeline_kwargs: Dict[str, object] = {}
        if variant is not None:
            pipeline_kwargs["variant"] = variant
        if torch_dtype is not None:
            pipeline_kwargs["torch_dtype"] = torch_dtype

        pipeline = pipelines[stage_key].from_pretrained(model_path, **pipeline_kwargs)

        self.device = torch.device(device)
        self.unet = pipeline.unet.to(self.device)
        self.unet.eval()
        requires_grad(self.unet, False)

        self.tokenizer = getattr(pipeline, "tokenizer", None)
        self.text_encoder = getattr(pipeline, "text_encoder", None)
        if self.tokenizer is None or self.text_encoder is None:
            raise ValueError(
                "Loaded DeepFloyd pipeline does not expose tokenizer/text_encoder components."
            )

        self.unconditional_prompt = unconditional_prompt or ""
        self.timestep = float(timestep)
        self.latent_dtype = next(self.unet.parameters()).dtype

        self.stage_image_size = image_size
        if self.stage_image_size is None:
            self.stage_image_size = getattr(pipeline, "image_size", None)
        if self.stage_image_size is None:
            # Fallback to UNet sample size if pipeline does not expose an image_size attribute
            self.stage_image_size = getattr(getattr(self.unet, "config", None), "sample_size", None)
        if self.stage_image_size is None:
            raise ValueError("Unable to infer DeepFloyd target image size. Please set image_size explicitly.")

        self.blocks = self._collect_blocks()
        if not self.blocks:
            raise ValueError("Unable to collect blocks from DeepFloyd UNet.")
        self.extraction_block_idx = self._resolve_block_index(extraction_block)

        self._cached_prompt = None
        self._cached_embeddings = None

    def _collect_blocks(self) -> List[torch.nn.Module]:
        blocks: List[torch.nn.Module] = []
        for name in ("down_blocks", "mid_block", "up_blocks"):
            module = getattr(self.unet, name, None)
            if module is None:
                continue
            if isinstance(module, torch.nn.ModuleList):
                blocks.extend(module)
            else:
                blocks.append(module)
        return blocks

    def _resolve_block_index(self, block: Union[int, str]) -> int:
        if isinstance(block, str):
            block = block.lower()
            if block == "first":
                return 0
            if block == "mid":
                # mid block is after all down blocks
                down_blocks = getattr(self.unet, "down_blocks", [])
                return len(down_blocks)
            if block == "last":
                return len(self.blocks) - 1
            raise ValueError(f"Unknown extraction block keyword '{block}'.")
        idx = int(block)
        return max(0, min(idx, len(self.blocks) - 1))

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        if images.max() > 1.0:
            images = images / 255.0
        if images.min() < 0:
            images = (images + 1.0) / 2.0

        if images.shape[-1] != self.stage_image_size:
            images = F.interpolate(
                images,
                size=(self.stage_image_size, self.stage_image_size),
                mode="bicubic",
                align_corners=False,
            )

        images = images * 2.0 - 1.0  # Normalize to [-1, 1]
        return images

    def _get_unconditional_embeddings(self, batch_size: int) -> torch.Tensor:
        if self._cached_embeddings is not None and self._cached_prompt == self.unconditional_prompt:
            if self._cached_embeddings.shape[0] == batch_size:
                return self._cached_embeddings

        prompts = [self.unconditional_prompt] * batch_size
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=getattr(self.tokenizer, "model_max_length", None),
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            text_embeddings = self.text_encoder(tokens["input_ids"], attention_mask=tokens.get("attention_mask"))[0]

        self._cached_prompt = self.unconditional_prompt
        self._cached_embeddings = text_embeddings
        return text_embeddings

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        images = self._preprocess_images(images)
        batch_size = images.shape[0]

        latents = images.to(dtype=self.latent_dtype)
        timesteps = torch.full(
            (batch_size,),
            self.timestep,
            dtype=torch.float32,
            device=self.device,
        )
        text_embeddings = self._get_unconditional_embeddings(batch_size).to(dtype=self.latent_dtype)

        target_block = self.blocks[self.extraction_block_idx]
        extracted = {}

        def hook_fn(_module, _input, output):
            extracted["features"] = output[0] if isinstance(output, (tuple, list)) else output

        handle = target_block.register_forward_hook(hook_fn)
        try:
            _ = self.unet(
                sample=latents,
                timestep=timesteps,
                encoder_hidden_states=text_embeddings,
            )
        finally:
            handle.remove()

        if "features" not in extracted:
            raise RuntimeError("Failed to capture DeepFloyd features at the requested block.")

        features = extracted["features"]
        if not torch.is_tensor(features):
            raise RuntimeError("DeepFloyd block hook did not return a tensor.")

        features = features.to(dtype=torch.float32)
        batch, channels, height, width = features.shape
        features = features.reshape(batch, channels, height * width).transpose(1, 2).contiguous()
        return features


__all__ = ["DeepFloydFeatureExtractor"]
