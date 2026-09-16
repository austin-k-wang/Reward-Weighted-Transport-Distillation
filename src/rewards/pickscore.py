"""Canonical adapter around the existing :mod:`src.pickscore` implementation."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import torch

from .base import Metadata, as_reward_tensor, validate_reward_inputs


class PickScoreReward:
    """Expose the existing PickScore implementation through the reward protocol."""

    def __init__(
        self,
        model_name_or_path: str | Path = "yuvalkirstain/PickScore_v1",
        processor_name_or_path: str | Path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        local_files_only: bool = False,
    ) -> None:
        """Load PickScore while retaining the established scoring implementation.

        Args:
            model_name_or_path: Hugging Face model ID or local PickScore path.
            processor_name_or_path: Hugging Face processor ID or local path.
            device: Inference device, defaulting to CUDA when available.
            dtype: Floating model and processor-input dtype.
            local_files_only: Whether Hugging Face loading must avoid network access.

        Returns:
            Nothing. The adapter owns a frozen legacy ``src.pickscore.PickScore``.
        """
        from src.pickscore import PickScore
        from transformers import AutoModel, AutoProcessor

        resolved_device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for PickScore but is unavailable")

        scorer = PickScore.__new__(PickScore)
        scorer.device = resolved_device
        scorer.dtype = dtype
        scorer.tensor_value_range = (-1.0, 1.0)
        scorer.logger = logging.getLogger("src.pickscore")
        scorer.processor = AutoProcessor.from_pretrained(
            str(processor_name_or_path), local_files_only=local_files_only
        )
        scorer.model = AutoModel.from_pretrained(
            str(model_name_or_path), local_files_only=local_files_only
        )
        scorer.model.eval().requires_grad_(False)
        scorer.model.to(device=resolved_device, dtype=dtype)
        self.scorer = scorer

    def score(
        self,
        prompts: Sequence[str],
        images: torch.Tensor,
        *,
        batch_size: int,
        metadata: Metadata = None,
    ) -> torch.Tensor:
        """Return canonical raw PickScore logits for aligned prompt-image pairs.

        Args:
            prompts: Prompt strings aligned with the image leading dimension.
            images: Floating image tensor shaped ``[N,3,H,W]`` in ``[-1,1]``.
            batch_size: Maximum number of pairs in each PickScore forward pass.
            metadata: Optional aligned metadata, accepted but not used.

        Returns:
            Detached CPU float32 PickScore logits shaped ``[N]``.
        """
        count = validate_reward_inputs(prompts, images, batch_size, metadata)
        if count == 0:
            return torch.empty(0, dtype=torch.float32)
        values = self.scorer.score(
            prompts, images, batch_size=batch_size, show_progress=False, metadata=metadata
        )
        return as_reward_tensor(values, count)


def legacy_pickscore_class() -> type:
    """Return the existing general-purpose PickScore class without wrapping it.

    Returns:
        The ``src.pickscore.PickScore`` class, imported only when requested.
    """
    from src.pickscore import PickScore

    return PickScore
