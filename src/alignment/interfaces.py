"""Protocols that isolate policies, enrichers, and online objectives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import torch

from .types import ObjectiveResult, Population, PromptBatch, RolloutRequest, RolloutResult


@runtime_checkable
class OneStepPolicy(Protocol):
    """Model adapter capable of live and frozen-reference one-step rollouts."""

    @property
    def device(self) -> torch.device:
        """Return the device used to sample one-step populations."""

    def rollout(
        self,
        prompts: Sequence[str],
        *,
        samples_per_prompt: int,
        trainable: bool,
        initial_noise: torch.Tensor | None = None,
    ) -> RolloutResult:
        """Generate a population while respecting the requested gradient mode."""

    def sample_initial_noise(
        self,
        count: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample device-resident particles from the policy noise prior.

        Args:
            count: Number ``N`` of independent particles.
            generator: Optional device-compatible random generator.

        Returns:
            Initial noise tensor shaped ``[N,C,H,W]``.
        """

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return only policy parameters that should be optimized."""


@runtime_checkable
class RewardEvaluator(Protocol):
    """Black-box image reward evaluator with no required gradient support."""

    def score(
        self,
        prompts: Sequence[str],
        images: torch.Tensor,
        *,
        batch_size: int,
        metadata: Sequence[Mapping[str, object] | None] | None = None,
    ) -> torch.Tensor:
        """Return one detached scalar reward per prompt-image pair.

        Args:
            prompts: Prompt strings aligned with the leading image dimension.
            images: Generated image tensor shaped ``[N,C,H,W]``.
            batch_size: Maximum evaluator batch size.
            metadata: Optional structured rows aligned with ``prompts``.

        Returns:
            Detached scalar rewards shaped ``[N]``.
        """


@runtime_checkable
class FeatureEncoder(Protocol):
    """Frozen image encoder that can retain gradients to live images."""

    def vector_features(
        self,
        images: torch.Tensor,
        features: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return named feature blocks with the image batch as dimension zero."""


@runtime_checkable
class OnlineObjective(Protocol):
    """Algorithm plug-in consumed by the generic alignment trainer."""

    @property
    def name(self) -> str:
        """Return a stable objective name used in logs and checkpoints."""

    def rollout_requests(self, config: object) -> tuple[RolloutRequest, ...]:
        """Declare all populations and enrichments needed for one step."""

    def compute(
        self,
        batch: PromptBatch,
        populations: Mapping[str, Population],
    ) -> ObjectiveResult:
        """Compute one scalar differentiable loss and logging diagnostics."""
