"""Human Preference Score v2.1 evaluator with local-checkpoint support."""

from __future__ import annotations

import gzip
import importlib.util
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .base import (
    Metadata,
    as_reward_tensor,
    images_to_unit_range_differentiable,
    iter_batch_slices,
    validate_reward_inputs,
)


_OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
_OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)


def _load_hps_open_clip() -> tuple[Any, Any]:
    """Import HPS's OpenCLIP fork with a fallback tokenizer vocabulary.

    Some PyPI builds of ``hpsv2`` omit
    ``bpe_simple_vocab_16e6.txt.gz`` even though the vendored tokenizer opens
    it at import time. The separately installed ``open_clip_torch`` package
    ships the identical vocabulary, so missing-file reads are redirected to
    that copy only for the duration of the HPS import.

    Returns:
        The HPS fork's ``create_model`` and ``get_tokenizer`` callables.

    Raises:
        FileNotFoundError: If both the HPS and OpenCLIP vocabulary files are
            unavailable.
    """
    open_clip_spec = importlib.util.find_spec("open_clip")
    fallback_bpe = None
    if open_clip_spec is not None and open_clip_spec.origin is not None:
        candidate = Path(open_clip_spec.origin).parent / "bpe_simple_vocab_16e6.txt.gz"
        if candidate.is_file():
            fallback_bpe = candidate

    original_gzip_open = gzip.open

    def compatible_gzip_open(filename: Any, *args: Any, **kwargs: Any) -> Any:
        """Open a gzip file, redirecting only a missing CLIP BPE vocabulary.

        Args:
            filename: Requested filesystem path or file-like object.
            *args: Additional positional arguments forwarded to ``gzip.open``.
            **kwargs: Additional keyword arguments forwarded to ``gzip.open``.

        Returns:
            Binary or text gzip stream returned by the original implementation.
        """
        requested = Path(filename) if isinstance(filename, (str, Path)) else None
        if (
            requested is not None
            and requested.name == "bpe_simple_vocab_16e6.txt.gz"
            and not requested.is_file()
        ):
            if fallback_bpe is None:
                raise FileNotFoundError(
                    "HPSv2 is missing bpe_simple_vocab_16e6.txt.gz and no "
                    "open_clip_torch fallback was found"
                )
            filename = fallback_bpe
        return original_gzip_open(filename, *args, **kwargs)

    gzip.open = compatible_gzip_open
    try:
        from hpsv2.src.open_clip import create_model, get_tokenizer
    finally:
        gzip.open = original_gzip_open
    return create_model, get_tokenizer


def _hps_preprocess(
    images: torch.Tensor,
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """Resize, letterbox, and normalize a canonical image batch for HPS.

    Args:
        images: Float image tensor shaped ``[N,3,H,W]`` with values in ``[0,1]``.
        image_size: Square HPS canvas side length in pixels.
        mean: Three per-channel normalization means.
        std: Three positive per-channel normalization standard deviations.

    Returns:
        Normalized float tensor shaped ``[N,3,image_size,image_size]``.
    """
    height, width = images.shape[-2:]
    scale = image_size / max(height, width)
    resized_height = max(1, round(height * scale))
    resized_width = max(1, round(width * scale))
    resized = F.interpolate(
        images, size=(resized_height, resized_width), mode="bicubic", align_corners=False
    )
    pad_height = image_size - resized_height
    pad_width = image_size - resized_width
    padded = F.pad(
        resized,
        (
            pad_width // 2,
            pad_width - pad_width // 2,
            pad_height // 2,
            pad_height - pad_height // 2,
        ),
    )
    mean_tensor = padded.new_tensor(mean).view(1, 3, 1, 1)
    std_tensor = padded.new_tensor(std).view(1, 3, 1, 1)
    return padded.sub(mean_tensor).div(std_tensor)


class HPSv21Reward:
    """Evaluate prompt-image alignment with HPS v2.1."""

    def __init__(
        self,
        hps_checkpoint_path: str | Path,
        *,
        model_checkpoint_path: str | Path | None = None,
        model_name: str = "ViT-H-14",
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Load the HPS OpenCLIP backbone and v2.1 preference checkpoint.

        Args:
            hps_checkpoint_path: Local HPS v2.1 state-dict checkpoint.
            model_checkpoint_path: Optional local OpenCLIP backbone checkpoint.
            model_name: HPS OpenCLIP architecture name.
            device: Inference device, defaulting to CUDA when available.
            dtype: Floating input and model dtype.

        Returns:
            Nothing. The evaluator owns a frozen HPS model and tokenizer.
        """
        create_model, get_tokenizer = _load_hps_open_clip()

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype
        self.model = create_model(
            model_name,
            str(model_checkpoint_path) if model_checkpoint_path is not None else None,
            precision="fp32",
            device=str(self.device),
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            output_dict=True,
        )
        checkpoint: Any = torch.load(
            str(hps_checkpoint_path), map_location="cpu", weights_only=False
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval().requires_grad_(False)
        self.model.to(device=self.device, dtype=dtype)
        self.tokenizer = get_tokenizer(model_name)
        image_size = self.model.visual.image_size
        self.image_size = int(image_size[0] if isinstance(image_size, (tuple, list)) else image_size)
        self.mean = tuple(getattr(self.model.visual, "image_mean", _OPENAI_MEAN))
        self.std = tuple(getattr(self.model.visual, "image_std", _OPENAI_STD))

    def encode_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into normalized differentiable HPS features.

        Args:
            images: Float RGB images shaped ``[N,3,H,W]`` in ``[-1,1]``.

        Returns:
            Normalized float32 image features shaped ``[N,D]``. Gradients are
            retained from the features to ``images`` while HPS stays frozen.
        """

        unit_images = images_to_unit_range_differentiable(images)
        pixels = _hps_preprocess(
            unit_images,
            self.image_size,
            self.mean,
            self.std,
        ).to(device=self.device, dtype=self.dtype)
        return F.normalize(self.model.encode_image(pixels).float(), dim=-1)

    @torch.inference_mode()
    def score(
        self,
        prompts: Sequence[str],
        images: torch.Tensor,
        *,
        batch_size: int,
        metadata: Metadata = None,
    ) -> torch.Tensor:
        """Return HPS v2.1 cosine similarities for aligned pairs.

        Args:
            prompts: Prompt strings aligned one-to-one with images.
            images: Floating image tensor shaped ``[N,3,H,W]`` in ``[-1,1]``.
            batch_size: Maximum pairs in one HPS model call.
            metadata: Optional aligned metadata, accepted but ignored.

        Returns:
            Detached CPU float32 HPS similarities shaped ``[N]``.
        """
        count = validate_reward_inputs(prompts, images, batch_size, metadata)
        unit_images = images_to_unit_range_differentiable(images)
        chunks: list[torch.Tensor] = []
        for indices in iter_batch_slices(count, batch_size):
            pixels = _hps_preprocess(
                unit_images[indices], self.image_size, self.mean, self.std
            ).to(device=self.device, dtype=self.dtype)
            tokens = self.tokenizer(list(prompts[indices])).to(self.device)
            outputs = self.model(pixels, tokens)
            logits = outputs["image_features"] @ outputs["text_features"].T
            chunks.append(logits.diagonal().to(dtype=torch.float32))
        values = (
            torch.cat(chunks)
            if chunks
            else torch.empty(0, device=self.device, dtype=torch.float32)
        )
        return as_reward_tensor(values, len(prompts))


class HPSv21FeatureEncoder:
    """Expose variance-scaled HPS v2.1 embeddings as RWTD features."""

    FEATURE_SCALE = 32.0

    def __init__(self, reward: HPSv21Reward) -> None:
        """Reuse an existing frozen HPS v2.1 evaluator for image features.

        Args:
            reward: Loaded HPS v2.1 evaluator whose normalized image embedding
                will define RWTD's transport and regression space.

        Returns:
            Nothing. The adapter retains the evaluator without copying its
            model or allocating another checkpoint.
        """
        self.reward = reward

    def vector_features(
        self,
        images: torch.Tensor,
        features: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode a generated image batch as one named HPS feature block.

        Args:
            images: Float RGB images shaped ``[N,3,H,W]`` in ``[-1,1]``.
            features: Requested feature names. RWTD must request exactly
                ``("hps",)``; ``None`` selects that block by default.

        Returns:
            Mapping from ``"hps"`` to differentiable float32 HPS image
            embeddings shaped ``[N,D]``. Unit-normalized HPS embeddings are
            multiplied by 32 (the square root of their 1024 dimensions) so
            RWTD's dimension-mean squared distances are not under-scaled.

        Raises:
            ValueError: If a feature block other than exactly ``"hps"`` is
                requested.
        """
        selected = ("hps",) if features is None else tuple(features)
        if selected != ("hps",):
            raise ValueError("HPSv21FeatureEncoder supports only the 'hps' feature block")
        return {
            "hps": self.reward.encode_image_features(images) * self.FEATURE_SCALE
        }
