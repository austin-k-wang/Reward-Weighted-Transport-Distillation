"""Fixed-temperature RWTD objective with selectable transport coupling."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..config import AlignmentConfig
from ..types import ObjectiveResult, Population, PromptBatch, RolloutRequest
from .base import register_objective


@dataclass(frozen=True)
class RWTDPreparedTargets:
    """Store detached regression targets for chunked differentiable regeneration.

    Args:
        targets: Per-feature regression targets shaped ``[B*K,D]``, where
            ``B`` is the prompt batch size and ``K`` is the current population
            size.
        metrics: Detached diagnostics computed from the complete populations.
        sample_count: Total number ``B*K`` of current samples represented by
            each target tensor.
    """

    targets: dict[str, torch.Tensor]
    metrics: dict[str, torch.Tensor | float]
    sample_count: int


def fixed_temperature_reward_masses(
    rewards: torch.Tensor,
    *,
    temperature: float = 0.5,
    reward_mean: float = 0.0,
    reward_scale: float = 1.0,
    mass_floor: float = 0.0,
) -> torch.Tensor:
    """Convert calibrated rewards into fixed-temperature probability masses.

    Args:
        rewards: Scalar rewards shaped ``[...,K]``.
        temperature: Positive fixed softmax temperature. No ESS adaptation is
            performed.
        reward_mean: Frozen global reward calibration mean.
        reward_scale: Positive frozen global reward calibration scale.
        mass_floor: Uniform mixture fraction in ``[0,1)``.

    Returns:
        Probability masses shaped like ``rewards`` and summing to one over the
        final dimension.
    """
    if temperature <= 0 or reward_scale <= 0:
        raise ValueError("Reward temperature and scale must be positive")
    if not 0 <= mass_floor < 1:
        raise ValueError("mass_floor must be in [0, 1)")
    calibrated = (rewards.float() - reward_mean) / reward_scale
    soft = torch.softmax(calibrated / temperature, dim=-1)
    if mass_floor == 0:
        return soft
    return (1 - mass_floor) * soft + mass_floor / rewards.shape[-1]


@torch.no_grad()
def sinkhorn_plan(
    source_mass: torch.Tensor,
    target_mass: torch.Tensor,
    cost: torch.Tensor,
    epsilon: torch.Tensor | float,
    *,
    iterations: int,
    tolerance: float | None,
) -> torch.Tensor:
    """Solve batched entropic optimal transport in the log domain.

    Args:
        source_mass: Positive source masses shaped ``[B,K]``.
        target_mass: Non-negative target masses shaped ``[B,L]`` with positive
            row sums. Zero entries allow current-only or reference-only support.
        cost: Detached non-negative transport costs shaped ``[B,K,L]``.
        epsilon: Positive scalar or one regularization value per batch item.
        iterations: Maximum Sinkhorn dual updates.
        tolerance: Optional maximum dual-update stopping threshold.

    Returns:
        Detached transport plan shaped ``[B,K,L]`` with approximately the
        requested row and column marginals.
    """
    if iterations < 1:
        raise ValueError("Sinkhorn iterations must be positive")
    epsilon_tensor = torch.as_tensor(
        epsilon,
        device=cost.device,
        dtype=cost.dtype,
    )
    if epsilon_tensor.ndim == 0:
        epsilon_tensor = epsilon_tensor.expand(cost.shape[0])
    if epsilon_tensor.shape != (cost.shape[0],) or torch.any(epsilon_tensor <= 0):
        raise ValueError("epsilon must be positive and scalar or shaped [B]")
    log_kernel = -cost / epsilon_tensor[:, None, None]
    log_a = source_mass.clamp_min(1e-30).log()
    log_b = target_mass.clamp_min(1e-30).log()
    dual_source = torch.zeros_like(log_a)
    dual_target = torch.zeros_like(log_b)
    for _ in range(iterations):
        next_source = log_a - torch.logsumexp(
            log_kernel + dual_target[:, None, :],
            dim=2,
        )
        next_target = log_b - torch.logsumexp(
            log_kernel + next_source[:, :, None],
            dim=1,
        )
        if tolerance is not None:
            delta = torch.maximum(
                (next_source - dual_source).abs().amax(),
                (next_target - dual_target).abs().amax(),
            )
            dual_source, dual_target = next_source, next_target
            if float(delta) <= tolerance:
                break
        else:
            dual_source, dual_target = next_source, next_target
    plan = torch.exp(
        dual_source[:, :, None] + log_kernel + dual_target[:, None, :]
    )
    # A short marginal projection removes residual error from sharply peaked
    # kernels without differentiating through the transport solver.
    projection_tolerance = tolerance if tolerance is not None else 1e-6
    for projection_step in range(500):
        plan = plan * (
            source_mass / plan.sum(dim=2).clamp_min(1e-30)
        )[:, :, None]
        plan = plan * (
            target_mass / plan.sum(dim=1).clamp_min(1e-30)
        )[:, None, :]
        if projection_step % 10 == 9:
            row_error = (plan.sum(dim=2) - source_mass).abs().amax()
            column_error = (plan.sum(dim=1) - target_mass).abs().amax()
            if float(torch.maximum(row_error, column_error)) <= projection_tolerance:
                break
    return plan


def _effective_sample_size(mass: torch.Tensor) -> torch.Tensor:
    """Compute batchwise effective sample size for normalized masses.

    Args:
        mass: Probability mass tensor shaped ``[B,K]``.

    Returns:
        ESS tensor shaped ``[B]``.
    """
    return mass.square().sum(dim=-1).clamp_min(1e-30).reciprocal()


@torch.no_grad()
def sample_random_coupling_indices(
    target_mass: torch.Tensor,
    *,
    source_count: int,
) -> torch.Tensor:
    """Sample independent reward-weighted targets for each source particle.

    Args:
        target_mass: Per-prompt target probabilities shaped ``[B,L]``, where
            ``B`` is the prompt batch size and ``L`` is the combined current
            and reference support size.
        source_count: Number ``K`` of current source particles per prompt.

    Returns:
        Integer support indices shaped ``[B,K]`` on the same device as
        ``target_mass``. Sampling is categorical with replacement and occurs
        independently inside each prompt row.
    """
    if target_mass.ndim != 2:
        raise ValueError("Random-coupling target masses must be shaped [B,L]")
    if source_count < 1:
        raise ValueError("Random-coupling source_count must be positive")
    if not torch.isfinite(target_mass).all() or torch.any(target_mass < 0):
        raise ValueError("Random-coupling target masses must be finite and non-negative")
    if torch.any(target_mass.sum(dim=1) <= 0):
        raise ValueError("Random-coupling target masses must have positive row sums")
    return torch.multinomial(
        target_mass,
        num_samples=source_count,
        replacement=True,
    )


@torch.no_grad()
def sample_sinkhorn_coupling_indices(
    plan: torch.Tensor,
    source_mass: torch.Tensor,
) -> torch.Tensor:
    """Sample one target from each conditional row of a Sinkhorn plan.

    Args:
        plan: Non-negative transport coupling shaped ``[B,K,L]`` whose source
            marginal is ``source_mass``.
        source_mass: Positive source particle masses shaped ``[B,K]``.

    Returns:
        Integer support indices shaped ``[B,K]``. Row probabilities are formed
        as ``plan[b,i,:] / source_mass[b,i]`` and renormalized to absorb small
        numerical marginal errors before categorical sampling.
    """
    if plan.ndim != 3:
        raise ValueError("Sinkhorn coupling plan must be shaped [B,K,L]")
    if source_mass.shape != plan.shape[:2]:
        raise ValueError("Sinkhorn source masses must match plan shape [B,K]")
    if not torch.isfinite(plan).all() or torch.any(plan < 0):
        raise ValueError("Sinkhorn coupling plan must be finite and non-negative")
    if not torch.isfinite(source_mass).all() or torch.any(source_mass <= 0):
        raise ValueError("Sinkhorn source masses must be finite and positive")
    row_probabilities = plan / source_mass[:, :, None]
    row_sums = row_probabilities.sum(
        dim=2,
        keepdim=True,
    )
    if torch.any(row_sums <= 0):
        raise ValueError("Every Sinkhorn coupling row must have positive mass")
    row_probabilities = row_probabilities / row_sums
    sampled = torch.multinomial(
        row_probabilities.reshape(-1, plan.shape[-1]),
        num_samples=1,
        replacement=True,
    )
    return sampled.reshape(plan.shape[0], plan.shape[1])


def _load_stats(path: str | None) -> dict[str, dict[str, torch.Tensor]]:
    """Load optional frozen per-coordinate feature whitening statistics.

    Args:
        path: JSON or PyTorch artifact containing ``means`` and ``scales``
            mappings, or ``None`` for identity whitening.

    Returns:
        Dictionary with tensor-valued ``means`` and ``scales`` mappings.
    """
    if path is None:
        return {"means": {}, "scales": {}}
    stats_path = Path(path).expanduser()
    if not stats_path.is_file():
        raise FileNotFoundError(f"RWTD feature statistics do not exist: {stats_path}")
    if stats_path.suffix.lower() == ".json":
        raw: Any = json.loads(stats_path.read_text(encoding="utf-8"))
    else:
        raw = torch.load(stats_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise TypeError("RWTD feature statistics must be a mapping")
    means = raw.get("means", raw.get("feature_means", {}))
    scales = raw.get("scales", raw.get("feature_scales", {}))
    if not isinstance(means, Mapping) or not isinstance(scales, Mapping):
        raise TypeError("RWTD feature means and scales must be mappings")
    return {
        "means": {str(key): torch.as_tensor(value).float() for key, value in means.items()},
        "scales": {str(key): torch.as_tensor(value).float() for key, value in scales.items()},
    }


class RWTDObjective:
    """Regress toward reward-weighted support using OT or random coupling."""

    def __init__(self, config: AlignmentConfig) -> None:
        """Resolve RWTD populations, feature weights, and frozen statistics.

        Args:
            config: Validated alignment configuration with ``objective=rwtd``.

        Returns:
            Nothing. Immutable algorithm settings are retained.
        """
        self.config = config
        self.feature_names = tuple(config.features.names)
        if config.objective.feature_weights:
            if len(config.objective.feature_weights) != len(self.feature_names):
                raise ValueError("feature_weights must align with features.names")
            self.feature_weights = tuple(
                float(value) for value in config.objective.feature_weights
            )
        else:
            self.feature_weights = tuple(1.0 for _ in self.feature_names)
        if any(value < 0 for value in self.feature_weights) or sum(self.feature_weights) <= 0:
            raise ValueError("RWTD feature weights must be non-negative with positive sum")
        self.stats = _load_stats(config.rwtd.feature_stats_path)

    @property
    def name(self) -> str:
        """Return the stable objective registry name.

        Returns:
            ``rwtd``.
        """
        return "rwtd"

    def rollout_requests(self, config: object) -> tuple[RolloutRequest, ...]:
        """Request independent scored current and frozen-reference populations.

        Args:
            config: Generic trainer argument retained for protocol symmetry.

        Returns:
            Live current and detached reference population requests.
        """
        del config
        return (
            RolloutRequest(
                name="current",
                samples_per_prompt=self.config.objective.current_samples,
                trainable=True,
                requires_rewards=True,
                feature_names=self.feature_names,
            ),
            RolloutRequest(
                name="reference",
                samples_per_prompt=self.config.objective.reference_samples,
                trainable=False,
                requires_rewards=True,
                feature_names=self.feature_names,
            ),
        )

    def _whiten(
        self,
        name: str,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Flatten and whiten one feature block with frozen statistics.

        Args:
            name: Feature block key.
            values: Features shaped ``[B,N,...]``.

        Returns:
            Whitened features shaped ``[B,N,D]``.
        """
        flat = values.flatten(start_dim=2).float()
        mean = self.stats["means"].get(name)
        scale = self.stats["scales"].get(name)
        if mean is None and scale is None:
            return flat
        if mean is None or scale is None:
            raise ValueError(f"RWTD feature statistics for {name!r} are incomplete")
        mean = mean.to(device=flat.device, dtype=flat.dtype).flatten()
        scale = scale.to(device=flat.device, dtype=flat.dtype).flatten()
        if mean.numel() != flat.shape[-1] or scale.numel() != flat.shape[-1]:
            raise ValueError(
                f"RWTD feature statistics for {name!r} have dimension "
                f"{mean.numel()}/{scale.numel()}, expected {flat.shape[-1]}"
            )
        if torch.any(scale <= 0):
            raise ValueError(f"RWTD feature scales for {name!r} must be positive")
        return (flat - mean) / scale.clamp_min(1e-6)

    def compute(
        self,
        batch: PromptBatch,
        populations: Mapping[str, Population],
    ) -> ObjectiveResult:
        """Compute fixed-temperature shared-plan multi-feature RWTD.

        Args:
            batch: Prompt conditions defining the leading batch dimension.
            populations: Scored current and reference populations with all
                configured feature blocks.

        Returns:
            Differentiable detached-target regression loss and RWTD diagnostics.
        """
        prepared = self.prepare_targets(batch, populations)
        loss = self.chunk_loss(
            populations["current"].features,
            prepared,
            start=0,
            stop=prepared.sample_count,
        )
        return ObjectiveResult(loss=loss, metrics=prepared.metrics)

    @torch.no_grad()
    def prepare_targets(
        self,
        batch: PromptBatch,
        populations: Mapping[str, Population],
    ) -> RWTDPreparedTargets:
        """Compute detached coupled targets from complete sampled populations.

        Args:
            batch: Prompt conditions defining leading population groups.
            populations: Current source and reference populations containing
                detached rewards and feature blocks. Current features are
                shaped ``[B*K,...]`` and reference features ``[B*R,...]``.

        Returns:
            Detached per-current-sample feature targets and full-population
            diagnostics suitable for a second, chunked gradient pass.
        """
        current = populations["current"]
        reference = populations["reference"]
        if current.rewards is None or reference.rewards is None:
            raise ValueError("RWTD requires current and reference rewards")
        batch_size = len(batch.prompts)
        current_count = self.config.objective.current_samples
        reference_count = self.config.objective.reference_samples
        current_rewards = current.rewards.reshape(batch_size, current_count)
        reference_rewards = reference.rewards.reshape(batch_size, reference_count)
        rwtd = self.config.rwtd
        current_mass = fixed_temperature_reward_masses(
            current_rewards,
            temperature=rwtd.reward_temperature,
            reward_mean=rwtd.reward_mean,
            reward_scale=rwtd.reward_scale,
            mass_floor=rwtd.mass_floor,
        )
        reference_mass = fixed_temperature_reward_masses(
            reference_rewards,
            temperature=rwtd.reward_temperature,
            reward_mean=rwtd.reward_mean,
            reward_scale=rwtd.reward_scale,
        )
        source_mass = current_rewards.new_full(
            (batch_size, current_count),
            1.0 / current_count,
        )
        target_mass = torch.cat(
            (
                (1 - rwtd.reference_fraction) * current_mass,
                rwtd.reference_fraction * reference_mass,
            ),
            dim=1,
        )

        live_blocks: dict[str, torch.Tensor] = {}
        target_blocks: dict[str, torch.Tensor] = {}
        cost = None
        total_feature_weight = sum(self.feature_weights)
        for name, feature_weight in zip(self.feature_names, self.feature_weights):
            live = self._whiten(
                name,
                current.features[name].reshape(batch_size, current_count, *current.features[name].shape[1:]),
            )
            frozen_source = live.detach()
            frozen_reference = self._whiten(
                name,
                reference.features[name].reshape(
                    batch_size,
                    reference_count,
                    *reference.features[name].shape[1:],
                ),
            ).detach()
            support = torch.cat(
                (frozen_source, frozen_reference),
                dim=1,
            )
            block_cost = (
                frozen_source[:, :, None, :] - support[:, None, :, :]
            ).square().mean(dim=-1)
            weighted_cost = feature_weight * block_cost / total_feature_weight
            cost = weighted_cost if cost is None else cost + weighted_cost
            live_blocks[name] = live
            target_blocks[name] = support
        if cost is None:
            raise ValueError("RWTD requires at least one feature block")

        detached_cost = cost.detach()
        plan = None
        epsilon = None
        sampled_indices = None
        if rwtd.coupling == "sinkhorn":
            epsilon = (
                rwtd.ot_regularization_scale
                * detached_cost.flatten(1).median(dim=1).values
            ).clamp_min(rwtd.minimum_ot_epsilon)
            plan = sinkhorn_plan(
                source_mass.detach(),
                target_mass.detach(),
                detached_cost,
                epsilon,
                iterations=rwtd.sinkhorn_iterations,
                tolerance=rwtd.sinkhorn_tolerance,
            )
            if rwtd.sinkhorn_target == "sampled":
                sampled_indices = sample_sinkhorn_coupling_indices(
                    plan,
                    source_mass.detach(),
                )
        else:
            sampled_indices = sample_random_coupling_indices(
                target_mass.detach(),
                source_count=current_count,
            )

        losses = []
        displacement_norms = []
        prepared_targets: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor | float] = {}
        metric_prefix = "rwtd" if rwtd.coupling == "sinkhorn" else "rwr"
        transport_mask = None
        if rwtd.stochastic_transport:
            transport_mask = (
                torch.rand(
                    (batch_size, current_count, 1),
                    device=source_mass.device,
                )
                < rwtd.transport_step
            )
            metrics[f"{metric_prefix}/transport_fraction"] = (
                transport_mask.float().mean().detach()
            )
        for name, feature_weight in zip(self.feature_names, self.feature_weights):
            live = live_blocks[name]
            if plan is not None and sampled_indices is None:
                coupled_target = torch.einsum(
                    "bkl,bld->bkd",
                    plan,
                    target_blocks[name],
                ) / source_mass[:, :, None]
            else:
                if sampled_indices is None:
                    raise RuntimeError("Sampled coupling did not produce target indices")
                coupled_target = torch.gather(
                    target_blocks[name],
                    dim=1,
                    index=sampled_indices[:, :, None].expand(
                        -1,
                        -1,
                        target_blocks[name].shape[-1],
                    ),
                )
            displacement = coupled_target - live.detach()
            if rwtd.displacement_clip is not None:
                norm = displacement.square().mean(dim=-1, keepdim=True).sqrt()
                displacement = displacement * (
                    rwtd.displacement_clip / norm.clamp_min(1e-12)
                ).clamp_max(1.0)
            if transport_mask is None:
                target = live.detach() + rwtd.transport_step * displacement
            else:
                target = live.detach() + transport_mask * displacement
            target = target.detach()
            block_loss = 0.5 * (live - target).square().mean()
            losses.append(feature_weight * block_loss)
            prepared_targets[name] = target.reshape(
                batch_size * current_count,
                target.shape[-1],
            )
            displacement_norms.append(
                displacement.square().mean(dim=-1).sqrt().mean().detach()
            )
            metrics[f"{metric_prefix}/{name}_loss"] = block_loss.detach()
        loss = torch.stack(losses).sum() / total_feature_weight

        metrics.update(
            {
                f"{metric_prefix}/loss": loss.detach(),
                f"{metric_prefix}/reward_temperature": rwtd.reward_temperature,
                f"{metric_prefix}/current_reward_mean": current_rewards.mean().detach(),
                f"{metric_prefix}/reference_reward_mean": reference_rewards.mean().detach(),
                f"{metric_prefix}/current_ess": _effective_sample_size(
                    current_mass
                ).mean().detach(),
                f"{metric_prefix}/reference_ess": _effective_sample_size(
                    reference_mass
                ).mean().detach(),
                f"{metric_prefix}/current_weight_max": current_mass.max(
                    dim=1
                ).values.mean().detach(),
                f"{metric_prefix}/reference_weight_max": reference_mass.max(
                    dim=1
                ).values.mean().detach(),
                f"{metric_prefix}/displacement_norm": torch.stack(
                    displacement_norms
                ).mean(),
                f"{metric_prefix}/reference_mass": rwtd.reference_fraction,
            }
        )
        if plan is not None:
            if epsilon is None:
                raise RuntimeError("Sinkhorn coupling did not produce epsilon")
            row_error = (plan.sum(dim=2) - source_mass).abs().amax()
            column_error = (plan.sum(dim=1) - target_mass).abs().amax()
            row_probabilities = plan / source_mass[:, :, None]
            row_entropy = -(
                row_probabilities.clamp_min(1e-30).log() * row_probabilities
            ).sum(dim=2).mean()
            metrics.update(
                {
                    "rwtd/ot_epsilon": epsilon.mean().detach(),
                    "rwtd/ot_row_error": row_error.detach(),
                    "rwtd/ot_column_error": column_error.detach(),
                    "rwtd/ot_row_entropy": row_entropy.detach(),
                }
            )
            if sampled_indices is not None:
                sampled_cost = torch.gather(
                    detached_cost,
                    dim=2,
                    index=sampled_indices[:, :, None],
                ).squeeze(2)
                unique_counts = torch.tensor(
                    [torch.unique(row).numel() for row in sampled_indices],
                    device=sampled_indices.device,
                    dtype=detached_cost.dtype,
                )
                metrics.update(
                    {
                        "rwtd/sampled_coupling_cost": sampled_cost.mean().detach(),
                        "rwtd/sampled_unique_targets": unique_counts.mean().detach(),
                    }
                )
        else:
            if sampled_indices is None:
                raise RuntimeError("Random coupling did not produce target indices")
            sampled_cost = torch.gather(
                detached_cost,
                dim=2,
                index=sampled_indices[:, :, None],
            ).squeeze(2)
            unique_counts = torch.tensor(
                [torch.unique(row).numel() for row in sampled_indices],
                device=sampled_indices.device,
                dtype=detached_cost.dtype,
            )
            metrics.update(
                {
                    "rwr/coupling_cost": sampled_cost.mean().detach(),
                    "rwr/unique_targets": unique_counts.mean().detach(),
                    "rwr/target_ess": _effective_sample_size(
                        target_mass
                    ).mean().detach(),
                }
            )
        return RWTDPreparedTargets(
            targets=prepared_targets,
            metrics=metrics,
            sample_count=batch_size * current_count,
        )

    def chunk_loss(
        self,
        features: Mapping[str, torch.Tensor],
        prepared: RWTDPreparedTargets,
        *,
        start: int,
        stop: int,
    ) -> torch.Tensor:
        """Compute one sample-weighted differentiable RWTD loss chunk.

        Args:
            features: Live regenerated feature blocks shaped ``[C,...]`` for
                the contiguous current-sample range ``[start,stop)``.
            prepared: Detached full-population targets from
                :meth:`prepare_targets`.
            start: Inclusive flattened current-sample index.
            stop: Exclusive flattened current-sample index.

        Returns:
            Scalar loss weighted so summing all non-overlapping chunks exactly
            recovers the complete-population mean loss.
        """
        if not 0 <= start < stop <= prepared.sample_count:
            raise ValueError(
                f"Invalid RWTD chunk [{start},{stop}) for "
                f"{prepared.sample_count} samples"
            )
        chunk_count = stop - start
        total_feature_weight = sum(self.feature_weights)
        losses = []
        for name, feature_weight in zip(self.feature_names, self.feature_weights):
            if name not in features or name not in prepared.targets:
                raise KeyError(f"Missing RWTD feature block {name!r}")
            values = features[name]
            if values.shape[0] != chunk_count:
                raise ValueError(
                    f"RWTD feature {name!r} has {values.shape[0]} samples, "
                    f"expected {chunk_count}"
                )
            live = self._whiten(name, values.unsqueeze(0)).squeeze(0)
            target = prepared.targets[name][start:stop]
            if live.shape != target.shape:
                raise ValueError(
                    f"RWTD feature {name!r} live/target shapes differ: "
                    f"{tuple(live.shape)} vs {tuple(target.shape)}"
                )
            block_loss = 0.5 * (live - target).square().mean()
            losses.append(feature_weight * block_loss)
        loss = torch.stack(losses).sum() / total_feature_weight
        return loss * (chunk_count / prepared.sample_count)


@register_objective("rwtd")
def _build_rwtd(config: AlignmentConfig) -> RWTDObjective:
    """Build the registered fixed-temperature RWTD objective.

    Args:
        config: Validated complete alignment configuration.

    Returns:
        Configured RWTD objective.
    """
    return RWTDObjective(config)
