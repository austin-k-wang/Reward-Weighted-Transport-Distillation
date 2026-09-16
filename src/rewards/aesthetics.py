"""LAION aesthetic predictor on top of Hugging Face CLIP embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .base import Metadata, as_reward_tensor, images_to_pil, iter_batch_slices, validate_reward_inputs
from .clip import _move_processor_inputs


class AestheticMLP(torch.nn.Module):
    """Standard LAION linear aesthetic predictor for normalized CLIP features."""

    def __init__(self, embedding_dim: int) -> None:
        """Build the published LAION aesthetic multilayer perceptron.

        Args:
            embedding_dim: Width ``D`` of each incoming CLIP image embedding.

        Returns:
            Nothing. The module maps tensors shaped ``[N,D]`` to ``[N,1]``.
        """
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict aesthetic ratings from normalized CLIP image embeddings.

        Args:
            embeddings: Floating normalized CLIP features shaped ``[N,D]``.

        Returns:
            Floating aesthetic predictions shaped ``[N,1]``.
        """
        return self.layers(embeddings)


class LAIONAestheticReward:
    """Predict LAION aesthetic ratings for canonical generated images."""

    def __init__(
        self,
        aesthetic_checkpoint_path: str | Path,
        *,
        clip_model_name_or_path: str | Path = "openai/clip-vit-large-patch14",
        processor_name_or_path: str | Path | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        local_files_only: bool = False,
    ) -> None:
        """Load a CLIP encoder and local LAION aesthetic MLP checkpoint.

        Args:
            aesthetic_checkpoint_path: Local predictor state-dict checkpoint.
            clip_model_name_or_path: Hugging Face CLIP ID or local directory.
            processor_name_or_path: Optional distinct processor ID or local path.
            device: Inference device, defaulting to CUDA when available.
            dtype: Floating CLIP, MLP, and pixel-input dtype.
            local_files_only: Whether Hugging Face loading must avoid network.

        Returns:
            Nothing. The evaluator owns frozen CLIP and aesthetic models.
        """
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype
        processor_source = processor_name_or_path or clip_model_name_or_path
        self.processor = CLIPProcessor.from_pretrained(
            str(processor_source), local_files_only=local_files_only
        )
        self.clip_model = CLIPModel.from_pretrained(
            str(clip_model_name_or_path), local_files_only=local_files_only
        )
        checkpoint: Any = torch.load(
            str(aesthetic_checkpoint_path), map_location="cpu", weights_only=False
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        embedding_dim = int(self.clip_model.config.projection_dim)
        if set(state_dict).issubset({"weight", "bias"}) and "weight" in state_dict:
            self.predictor = torch.nn.Linear(embedding_dim, 1)
        else:
            self.predictor = AestheticMLP(embedding_dim)
        self.predictor.load_state_dict(state_dict)
        self.clip_model.eval().requires_grad_(False)
        self.predictor.eval().requires_grad_(False)
        self.clip_model.to(device=self.device, dtype=dtype)
        self.predictor.to(device=self.device, dtype=dtype)

    @torch.inference_mode()
    def score(
        self,
        prompts: Sequence[str],
        images: torch.Tensor,
        *,
        batch_size: int,
        metadata: Metadata = None,
    ) -> torch.Tensor:
        """Return one LAION aesthetic rating per image.

        Args:
            prompts: Aligned prompts, validated but unused by this image-only reward.
            images: Floating image tensor shaped ``[N,3,H,W]`` in ``[-1,1]``.
            batch_size: Maximum images processed in one CLIP call.
            metadata: Optional aligned metadata, accepted but ignored.

        Returns:
            Detached CPU float32 aesthetic predictions shaped ``[N]``.
        """
        count = validate_reward_inputs(prompts, images, batch_size, metadata)
        pil_images = images_to_pil(images)
        chunks: list[torch.Tensor] = []
        for indices in iter_batch_slices(count, batch_size):
            inputs = self.processor(images=pil_images[indices], return_tensors="pt")
            features = self.clip_model.get_image_features(
                **_move_processor_inputs(inputs, self.device, self.dtype)
            )
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            predictions = self.predictor(F.normalize(features.float(), dim=-1).to(self.dtype))
            chunks.append(as_reward_tensor(predictions, indices.stop - indices.start))
        return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float32)
