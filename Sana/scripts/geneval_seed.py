"""Process-count-invariant seeds for SANA GenEval image generation."""

from __future__ import annotations

import torch


SANA_SPRINT_PAPER_4_STEP_TIMESTEPS = (
    1.5682963320032104,
    1.3,
    1.1,
    0.6,
    0.0,
)
"""SANA-Sprint's optimized four-step schedule from paper Appendix F.2."""


def validate_geneval_timesteps(
    sample_steps: int,
    timesteps: list[float] | tuple[float, ...] | None,
) -> list[float] | None:
    """Validate an optional explicit GenEval inference timestep schedule.

    Args:
        sample_steps: Number of denoising transitions to execute.
        timesteps: Optional sequence of timestep boundaries. A schedule for
            ``sample_steps`` transitions must contain ``sample_steps + 1``
            values, including the final zero boundary.

    Returns:
        The validated schedule as a mutable list, or ``None`` when the
        scheduler should construct its legacy default schedule.

    Raises:
        ValueError: If the step count is not positive, the schedule has the
            wrong length, is not strictly descending, or does not end at zero.
    """
    if sample_steps <= 0:
        raise ValueError("sample_steps must be positive")
    if timesteps is None:
        return None

    values = [float(timestep) for timestep in timesteps]
    expected_length = sample_steps + 1
    if len(values) != expected_length:
        raise ValueError(
            f"{sample_steps} inference steps require {expected_length} timestep "
            f"boundaries, got {len(values)}"
        )
    if any(current <= following for current, following in zip(values, values[1:])):
        raise ValueError("Inference timesteps must be strictly descending")
    if values[-1] != 0.0:
        raise ValueError("The final inference timestep must be 0.0")
    return values


def geneval_sample_seed(
    base_seed: int,
    prompt_index: int,
    image_index: int,
) -> int:
    """Derive one deterministic seed from global evaluation indices.

    Args:
        base_seed: Non-negative seed shared by the complete evaluation run.
        prompt_index: Global zero-based index in the 553-prompt GenEval suite.
        image_index: Zero-based sample index within one prompt.

    Returns:
        Seed in ``[0, 2**63 - 1)`` that is independent of process sharding.

    Raises:
        ValueError: If any input index or seed is negative.
    """
    if base_seed < 0 or prompt_index < 0 or image_index < 0:
        raise ValueError("GenEval seed inputs must be non-negative")
    return (base_seed + prompt_index * 1_000_003 + image_index) % (2**63 - 1)


def advance_official_geneval_generator(
    generator: torch.Generator,
    *,
    prompt_count: int,
    batch_size: int,
    latent_channels: int,
    latent_size: int,
    device: torch.device | str,
) -> None:
    """Advance a generator to an official GenEval prompt-shard boundary.

    The upstream SANA script draws one latent batch for each prompt from a
    single continuous generator. This helper discards the batches belonging to
    earlier prompts so a sharded worker begins at the same RNG state as the
    corresponding prompt in an unsharded official run.

    Args:
        generator: Seeded Torch generator whose state is advanced in place.
        prompt_count: Number of complete preceding prompts to skip.
        batch_size: Images drawn together for each prompt.
        latent_channels: Channel count of each sampled latent tensor.
        latent_size: Height and width of each square latent tensor.
        device: Torch device on which random latent tensors are generated.

    Returns:
        Nothing. ``generator`` is advanced in place.

    Raises:
        ValueError: If any count or tensor dimension is negative or zero where
            a positive dimension is required.
    """
    if prompt_count < 0:
        raise ValueError("prompt_count must be non-negative")
    if batch_size <= 0 or latent_channels <= 0 or latent_size <= 0:
        raise ValueError("latent batch dimensions must be positive")
    for _ in range(prompt_count):
        torch.randn(
            batch_size,
            latent_channels,
            latent_size,
            latent_size,
            device=device,
            generator=generator,
        )
