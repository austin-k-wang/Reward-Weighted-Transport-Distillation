"""Shared validation and image conversion utilities for reward evaluators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from PIL import Image


Metadata = Sequence[Mapping[str, object] | None] | None


def validate_reward_inputs(
    prompts: Sequence[str],
    images: torch.Tensor,
    batch_size: int,
    metadata: Metadata = None,
) -> int:
    """Validate the canonical reward inputs and return their batch cardinality.

    Args:
        prompts: Prompt strings with one element for each image.
        images: Floating tensor shaped ``[N,3,H,W]`` interpreted in SANA's
            approximate ``[-1,1]`` range. Finite decoder overshoot is accepted
            and clipped by model-specific preprocessing.
        batch_size: Maximum number of examples processed in one model call.
        metadata: Optional metadata rows aligned one-to-one with the images.

    Returns:
        The validated image count ``N``.

    Raises:
        TypeError: If images are not a floating tensor or prompts are not strings.
        ValueError: If shape, finiteness, cardinality, or batch size is invalid.
    """
    if not isinstance(images, torch.Tensor) or not torch.is_floating_point(images):
        raise TypeError("images must be a floating-point torch.Tensor")
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"images must have shape [N,3,H,W], got {tuple(images.shape)}")
    count = images.shape[0]
    if len(prompts) != count:
        raise ValueError(f"received {len(prompts)} prompts for {count} images")
    if any(not isinstance(prompt, str) for prompt in prompts):
        raise TypeError("every prompt must be a string")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if metadata is not None and len(metadata) != count:
        raise ValueError(f"received {len(metadata)} metadata rows for {count} images")
    if images.numel() and not torch.isfinite(images.detach()).all():
        raise ValueError("floating image values must be finite")
    return count


def iter_batch_slices(count: int, batch_size: int) -> list[slice]:
    """Build contiguous slices that partition a reward input batch.

    Args:
        count: Total number of prompt-image pairs, which may be zero.
        batch_size: Positive maximum number of pairs in each slice.

    Returns:
        Ordered contiguous slices covering indices ``[0, count)``.

    Raises:
        ValueError: If ``count`` is negative or ``batch_size`` is not positive.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return [slice(start, min(start + batch_size, count)) for start in range(0, count, batch_size)]


def images_to_unit_range(images: torch.Tensor) -> torch.Tensor:
    """Convert canonical images from ``[-1,1]`` to detached float32 ``[0,1]``.

    Args:
        images: Floating image tensor shaped ``[N,3,H,W]`` in ``[-1,1]``.

    Returns:
        Detached float32 tensor shaped ``[N,3,H,W]`` on the input device, with
        values clamped to ``[0,1]``.
    """
    return images.detach().to(dtype=torch.float32).add(1.0).mul(0.5).clamp_(0.0, 1.0)


def images_to_unit_range_differentiable(images: torch.Tensor) -> torch.Tensor:
    """Convert canonical images to float32 unit range without detaching.

    Args:
        images: Floating image tensor shaped ``[N,3,H,W]`` in ``[-1,1]``.

    Returns:
        Float32 tensor shaped ``[N,3,H,W]`` on the input device with values
        clamped to ``[0,1]`` and gradients preserved with respect to ``images``.
    """
    return images.to(dtype=torch.float32).add(1.0).mul(0.5).clamp(0.0, 1.0)


def images_to_pil(images: torch.Tensor) -> list[Image.Image]:
    """Convert canonical image tensors into independent RGB PIL images.

    Args:
        images: Floating image tensor shaped ``[N,3,H,W]`` in ``[-1,1]``.

    Returns:
        A list of ``N`` RGB images quantized to uint8 using round-to-nearest.
    """
    pixels = (
        images_to_unit_range(images)
        .mul(255.0)
        .round()
        .to(dtype=torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
        .cpu()
        .numpy()
    )
    return [Image.fromarray(array) for array in pixels]


def as_reward_tensor(values: object, expected_count: int) -> torch.Tensor:
    """Normalize arbitrary scalar score output to canonical reward format.

    Args:
        values: Tensor, sequence, NumPy array, or scalar-like model output.
        expected_count: Required number of scalar rewards.

    Returns:
        Detached contiguous CPU float32 tensor shaped ``[expected_count]``.

    Raises:
        ValueError: If the flattened output does not contain exactly one value
            for every prompt-image pair.
    """
    result = torch.as_tensor(values, dtype=torch.float32).detach().reshape(-1).cpu().contiguous()
    if result.numel() != expected_count:
        raise ValueError(
            f"reward model returned {result.numel()} values for {expected_count} inputs"
        )
    return result
