"""Reward and differentiable feature enrichment for generated populations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext

import torch

from .interfaces import FeatureEncoder, RewardEvaluator
from .types import Population, RolloutRequest, RolloutResult


class PopulationEnricher:
    """Apply optional black-box rewards and frozen image features."""

    def __init__(
        self,
        *,
        reward: RewardEvaluator | None,
        features: FeatureEncoder | None,
        reward_batch_size: int = 8,
        feature_batch_size: int = 4,
    ) -> None:
        """Configure reusable population enrichment components.

        Args:
            reward: Optional detached reward evaluator.
            features: Optional frozen differentiable image encoder.
            reward_batch_size: Maximum images per reward forward pass.
            feature_batch_size: Maximum images per feature forward pass.

        Returns:
            Nothing. Components are retained for future rollout enrichment.
        """
        if reward_batch_size < 1 or feature_batch_size < 1:
            raise ValueError("Enricher batch sizes must be positive")
        self.reward = reward
        self.features = features
        self.reward_batch_size = reward_batch_size
        self.feature_batch_size = feature_batch_size

    def enrich(
        self,
        rollout: RolloutResult,
        request: RolloutRequest,
        *,
        metadata: Sequence[Mapping[str, object] | None] | None = None,
    ) -> Population:
        """Add exactly the rewards and feature blocks requested by an objective.

        Args:
            rollout: Generated population whose image dimension is ``N``.
            request: Population declaration controlling enrichment and gradient
                behavior.
            metadata: Optional structured reward rows aligned with the generated
                population, shape ``[N]``.

        Returns:
            Population containing the rollout and requested enrichments.

        Raises:
            RuntimeError: If a requested reward or feature component is absent.
        """
        rewards = None
        if request.requires_rewards:
            if self.reward is None:
                raise RuntimeError(f"Population {request.name!r} requires a reward evaluator")
            rewards = self.reward.score(
                rollout.prompts,
                rollout.images.detach(),
                batch_size=self.reward_batch_size,
                metadata=metadata,
            ).detach().to(device=rollout.images.device, dtype=torch.float32)

        blocks: dict[str, list[torch.Tensor]] = {
            name: [] for name in request.feature_names
        }
        if request.feature_names:
            if self.features is None:
                raise RuntimeError(f"Population {request.name!r} requires a feature encoder")
            gradient_context = nullcontext() if request.trainable else torch.no_grad()
            with gradient_context:
                for chunk in rollout.images.split(self.feature_batch_size):
                    values = self.features.vector_features(
                        chunk,
                        request.feature_names,
                    )
                    for name in request.feature_names:
                        blocks[name].append(values[name])
        features = {name: torch.cat(values, dim=0) for name, values in blocks.items()}
        return Population(rollout=rollout, rewards=rewards, features=features)
