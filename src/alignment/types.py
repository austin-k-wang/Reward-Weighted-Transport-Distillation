"""Shared tensor containers for online-alignment rollouts and objectives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class PromptBatch:
    """Text conditions consumed by one online-alignment optimizer step.

    Args:
        prompts: Ordered prompt strings, one per independent condition.
        metadata: Optional algorithm/reward metadata aligned with ``prompts``.
    """

    prompts: tuple[str, ...]
    metadata: tuple[dict[str, Any] | None, ...] = ()

    def __post_init__(self) -> None:
        """Validate prompt and metadata cardinality after construction."""
        if not self.prompts:
            raise ValueError("PromptBatch requires at least one prompt")
        if self.metadata and len(self.metadata) != len(self.prompts):
            raise ValueError("PromptBatch metadata must align with prompts")


@dataclass(frozen=True)
class RolloutRequest:
    """Describe one population required by an alignment objective.

    Args:
        name: Stable population name, such as ``current`` or ``reference``.
        samples_per_prompt: Number of independently seeded images per prompt.
        trainable: Whether the active LoRA policy and an autograd graph are
            required. False selects the frozen base policy.
        requires_rewards: Whether the population must be reward-scored.
        feature_names: DINO feature blocks required by the objective.
        shared_noise_group: Optional name causing requests to reuse initial
            noise when their shapes match.
    """

    name: str
    samples_per_prompt: int
    trainable: bool
    requires_rewards: bool = False
    feature_names: tuple[str, ...] = ()
    shared_noise_group: str | None = None

    def __post_init__(self) -> None:
        """Validate population name and sample count after construction."""
        if not self.name:
            raise ValueError("RolloutRequest name must not be empty")
        if self.samples_per_prompt < 1:
            raise ValueError("samples_per_prompt must be positive")


@dataclass
class RolloutResult:
    """Hold generated tensors and reusable conditioning for one population.

    Args:
        name: Population name copied from its request.
        prompts: Prompt repeated once per generated image.
        initial_noise: Initial SCM latent noise shaped ``[N,C,H,W]``.
        denoised_latents: One-step denoised latents shaped ``[N,C,H,W]``.
        images: Decoded images shaped ``[N,3,H_img,W_img]`` in ``[-1,1]``.
        extras: Optional model-specific tensors or diagnostics.
    """

    name: str
    prompts: tuple[str, ...]
    initial_noise: torch.Tensor
    denoised_latents: torch.Tensor
    images: torch.Tensor
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Population:
    """Combine a rollout with optional rewards and feature blocks.

    Args:
        rollout: Generated images, latents, prompts, and rollout metadata.
        rewards: Optional scalar rewards shaped ``[N]``. They are detached
            unless the request explicitly requires reward gradients.
        features: Named feature tensors whose first dimension is ``N``.
    """

    rollout: RolloutResult
    rewards: torch.Tensor | None = None
    features: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class ObjectiveResult:
    """Return a differentiable objective loss and detached diagnostics.

    Args:
        loss: Scalar tensor used for backpropagation.
        metrics: Named scalar tensors or Python numbers used for logging.
    """

    loss: torch.Tensor
    metrics: dict[str, torch.Tensor | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Require a scalar loss so trainer reduction remains unambiguous."""
        if self.loss.ndim != 0:
            raise ValueError(f"Objective loss must be scalar, got {tuple(self.loss.shape)}")


@dataclass
class TrainerState:
    """Track resumable optimizer progress.

    Args:
        global_step: Number of completed optimizer steps.
        micro_step: Number of consumed micro-batches.
    """

    global_step: int = 0
    micro_step: int = 0
