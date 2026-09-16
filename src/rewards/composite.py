"""Standardized weighted composition of canonical reward evaluators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from src.alignment.interfaces import RewardEvaluator

from .base import Metadata, as_reward_tensor, validate_reward_inputs


class StandardizedWeightedReward:
    """Combine fixed reward evaluators after explicit affine standardization."""

    def __init__(
        self,
        evaluators: Mapping[str, RewardEvaluator],
        weights: Mapping[str, float],
        *,
        centers: Mapping[str, float] | None = None,
        scales: Mapping[str, float] | None = None,
        normalize_weights: bool = False,
    ) -> None:
        """Configure a deterministic weighted sum of standardized rewards.

        Args:
            evaluators: Named canonical reward evaluators.
            weights: Finite scalar weight for every evaluator name.
            centers: Optional fixed means subtracted from component rewards;
                omitted names use zero.
            scales: Optional fixed positive standard deviations dividing component
                rewards; omitted names use one.
            normalize_weights: Whether to divide the result by the sum of absolute
                component weights.

        Returns:
            Nothing. Component objects and immutable numeric settings are retained.

        Raises:
            ValueError: If names mismatch, settings are non-finite, a scale is not
                positive, or normalized weights have zero total magnitude.
        """
        if not evaluators:
            raise ValueError("at least one evaluator is required")
        if set(evaluators) != set(weights):
            raise ValueError("evaluator and weight names must match exactly")
        centers = centers or {}
        scales = scales or {}
        unknown = (set(centers) | set(scales)) - set(evaluators)
        if unknown:
            raise ValueError(f"statistics provided for unknown rewards: {sorted(unknown)}")
        self.evaluators = dict(evaluators)
        self.weights = {name: float(weights[name]) for name in evaluators}
        self.centers = {name: float(centers.get(name, 0.0)) for name in evaluators}
        self.scales = {name: float(scales.get(name, 1.0)) for name in evaluators}
        settings = [*self.weights.values(), *self.centers.values(), *self.scales.values()]
        if not all(torch.isfinite(torch.tensor(value)).item() for value in settings):
            raise ValueError("weights, centers, and scales must be finite")
        if any(scale <= 0 for scale in self.scales.values()):
            raise ValueError("every standardization scale must be positive")
        magnitude = sum(abs(weight) for weight in self.weights.values())
        if normalize_weights and magnitude == 0:
            raise ValueError("normalized component weights cannot all be zero")
        self.weight_divisor = magnitude if normalize_weights else 1.0
        self.last_components: dict[str, torch.Tensor] = {}

    def score(
        self,
        prompts: Sequence[str],
        images: torch.Tensor,
        *,
        batch_size: int,
        metadata: Metadata = None,
    ) -> torch.Tensor:
        """Evaluate, standardize, and combine all configured component rewards.

        Args:
            prompts: Prompt strings aligned one-to-one with images.
            images: Floating image tensor shaped ``[N,3,H,W]`` in ``[-1,1]``.
            batch_size: Maximum batch size forwarded to every component.
            metadata: Optional aligned metadata forwarded to every component.

        Returns:
            Detached CPU float32 weighted rewards shaped ``[N]``. Standardized
            component tensors are also exposed through ``last_components``.
        """
        count = validate_reward_inputs(prompts, images, batch_size, metadata)
        total = torch.zeros(count, dtype=torch.float32)
        components: dict[str, torch.Tensor] = {}
        for name, evaluator in self.evaluators.items():
            raw = as_reward_tensor(
                evaluator.score(
                    prompts, images, batch_size=batch_size, metadata=metadata
                ),
                count,
            )
            standardized = (raw - self.centers[name]) / self.scales[name]
            components[name] = standardized
            total.add_(standardized, alpha=self.weights[name])
        self.last_components = components
        return total.div(self.weight_divisor).detach().to(dtype=torch.float32, device="cpu")
