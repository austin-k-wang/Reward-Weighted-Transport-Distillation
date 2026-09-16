"""Hugging Face CLIP prompt-image similarity reward."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .base import Metadata, as_reward_tensor, images_to_pil, iter_batch_slices, validate_reward_inputs


def _move_processor_inputs(
    inputs: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Move processor tensors while preserving integer token dtypes.

    Args:
        inputs: Mapping-like Hugging Face processor output.
        device: Target model device.
        dtype: Target dtype for floating image tensors.

    Returns:
        A plain dictionary whose tensors retain shape, with floating values cast
        to ``dtype`` and all values moved to ``device``.
    """
    return {
        name: value.to(device=device, dtype=dtype)
        if torch.is_floating_point(value)
        else value.to(device=device)
        for name, value in inputs.items()
    }


class CLIPReward:
    """Score aligned pairs using Hugging Face CLIP similarity logits."""

    def __init__(
        self,
        model_name_or_path: str | Path = "openai/clip-vit-large-patch14",
        *,
        processor_name_or_path: str | Path | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        local_files_only: bool = False,
        logit_scale_divisor: float = 100.0,
    ) -> None:
        """Load a frozen Hugging Face CLIP model and processor.

        Args:
            model_name_or_path: Hugging Face model ID or local model directory.
            processor_name_or_path: Optional distinct processor ID or local path.
            device: Inference device, defaulting to CUDA when available.
            dtype: Floating model and pixel-input dtype.
            local_files_only: Whether loading must avoid network access.
            logit_scale_divisor: Positive divisor applied to CLIP logits; ``100``
                matches the commonly used CLIP reward scale.

        Returns:
            Nothing. The evaluator owns a frozen CLIP model and processor.

        Raises:
            ValueError: If ``logit_scale_divisor`` is not positive.
        """
        if logit_scale_divisor <= 0:
            raise ValueError("logit_scale_divisor must be positive")
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype
        source = processor_name_or_path or model_name_or_path
        self.processor = CLIPProcessor.from_pretrained(
            str(source), local_files_only=local_files_only
        )
        self.model = CLIPModel.from_pretrained(
            str(model_name_or_path), local_files_only=local_files_only
        )
        self.model.eval().requires_grad_(False)
        self.model.to(device=self.device, dtype=dtype)
        self.logit_scale_divisor = float(logit_scale_divisor)

    @torch.inference_mode()
    def score(
        self,
        prompts: Sequence[str],
        images: torch.Tensor,
        *,
        batch_size: int,
        metadata: Metadata = None,
    ) -> torch.Tensor:
        """Return scaled CLIP logits for corresponding prompt-image pairs.

        Args:
            prompts: Prompt strings aligned one-to-one with images.
            images: Floating image tensor shaped ``[N,3,H,W]`` in ``[-1,1]``.
            batch_size: Maximum pairs processed by one CLIP forward pass.
            metadata: Optional aligned metadata, accepted but ignored.

        Returns:
            Detached CPU float32 scaled diagonal logits shaped ``[N]``.
        """
        count = validate_reward_inputs(prompts, images, batch_size, metadata)
        pil_images = images_to_pil(images)
        chunks: list[torch.Tensor] = []
        for indices in iter_batch_slices(count, batch_size):
            inputs = self.processor(
                text=list(prompts[indices]),
                images=pil_images[indices],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            outputs = self.model(
                **_move_processor_inputs(inputs, self.device, self.dtype)
            )
            logits = outputs.logits_per_image
            chunks.append(
                as_reward_tensor(logits.diagonal() / self.logit_scale_divisor, indices.stop - indices.start)
            )
        return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float32)
