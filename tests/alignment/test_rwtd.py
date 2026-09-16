"""Tests for fixed-temperature Reward-Weighted Transport Distillation."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from src.alignment.config import (
    AlignmentConfig,
    FeatureConfig,
    ObjectiveConfig,
    RWTDConfig,
    RewardConfig,
)
from src.alignment.objectives import build_objective
from src.alignment.objectives.rwtd import (
    fixed_temperature_reward_masses,
    sample_random_coupling_indices,
    sample_sinkhorn_coupling_indices,
    sinkhorn_plan,
)
from src.alignment.types import Population, PromptBatch, RolloutResult


def _population(
    name: str,
    features: torch.Tensor,
    rewards: torch.Tensor,
) -> Population:
    """Construct a scored one-block population for RWTD tests.

    Args:
        name: Population identifier.
        features: Feature tensor shaped ``[N,1,D]``.
        rewards: Scalar rewards shaped ``[N]``.

    Returns:
        Population carrying aligned dummy rollout tensors.
    """
    images = features.transpose(1, 2).unsqueeze(-1)
    return Population(
        rollout=RolloutResult(
            name=name,
            prompts=tuple("prompt" for _ in range(features.shape[0])),
            initial_noise=images.detach(),
            denoised_latents=images,
            images=images,
        ),
        rewards=rewards,
        features={"cls": features},
    )


def test_fixed_temperature_reward_masses_use_requested_temperature() -> None:
    """Verify fixed softmax temperature and uniform mass-floor behavior.

    Returns:
        Nothing. The test checks normalization, floor, and concentration.
    """
    rewards = torch.tensor([[0.0, 1.0, 2.0]])

    masses = fixed_temperature_reward_masses(
        rewards,
        temperature=0.5,
        mass_floor=0.1,
    )
    hotter = fixed_temperature_reward_masses(rewards, temperature=1.0)

    torch.testing.assert_close(masses.sum(dim=-1), torch.ones(1))
    assert torch.all(masses >= 0.1 / 3)
    assert masses[0, -1] > hotter[0, -1]


def test_log_sinkhorn_matches_requested_marginals() -> None:
    """Verify the detached OT solver satisfies row and column capacities.

    Returns:
        Nothing. The test checks both Sinkhorn marginal constraints.
    """
    source = torch.tensor([[0.5, 0.5]])
    target = torch.tensor([[0.2, 0.3, 0.5]])
    cost = torch.tensor([[[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]])

    plan = sinkhorn_plan(
        source,
        target,
        cost,
        0.2,
        iterations=200,
        tolerance=1e-7,
    )

    torch.testing.assert_close(plan.sum(dim=2), source, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(plan.sum(dim=1), target, atol=1e-5, rtol=1e-5)
    assert not plan.requires_grad


@pytest.mark.parametrize(
    "target",
    [
        torch.tensor([[0.2, 0.8, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.3, 0.7]]),
    ],
)
def test_log_sinkhorn_supports_zero_mass_population_blocks(
    target: torch.Tensor,
) -> None:
    """Verify endpoint reference fractions yield valid sparse OT marginals.

    Args:
        target: Target probability tensor shaped ``[1,4]`` with either its
            current-support half or frozen-reference-support half set to zero.

    Returns:
        Nothing. The test checks finite plans and both requested marginals.
    """
    source = torch.tensor([[0.5, 0.5]])
    cost = torch.tensor(
        [[[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]]]
    )

    plan = sinkhorn_plan(
        source,
        target,
        cost,
        0.2,
        iterations=200,
        tolerance=1e-7,
    )

    assert torch.isfinite(plan).all()
    torch.testing.assert_close(plan.sum(dim=2), source, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(plan.sum(dim=1), target, atol=1e-5, rtol=1e-5)


def test_random_coupling_samples_independently_within_prompt_rows() -> None:
    """Verify categorical sampling cannot select support from another prompt.

    Returns:
        Nothing. One-hot row masses make the expected per-prompt indices exact.
    """
    target_mass = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    indices = sample_random_coupling_indices(target_mass, source_count=4)

    torch.testing.assert_close(indices[0], torch.zeros(4, dtype=torch.long))
    torch.testing.assert_close(indices[1], torch.full((4,), 2, dtype=torch.long))
    assert indices.device == target_mass.device


def test_random_coupling_is_reproducible_with_fixed_rng_state() -> None:
    """Verify categorical target indices replay from the same torch RNG seed.

    Returns:
        Nothing. Repeated non-degenerate sampling must return equal indices
        when the global RNG state is reset.
    """
    target_mass = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    torch.manual_seed(123)
    first = sample_random_coupling_indices(target_mass, source_count=20)
    torch.manual_seed(123)
    second = sample_random_coupling_indices(target_mass, source_count=20)

    torch.testing.assert_close(first, second)
    assert torch.unique(first).numel() > 1


def test_sinkhorn_coupling_samples_from_each_conditional_row() -> None:
    """Verify sampled RWTD normalizes and samples each OT row independently.

    Returns:
        Nothing. Deterministic rows with non-uniform source masses must select
        their sole supported target.
    """
    source_mass = torch.tensor([[0.25, 0.75]])
    plan = torch.tensor(
        [
            [
                [0.25, 0.0, 0.0],
                [0.0, 0.0, 0.75],
            ]
        ]
    )

    indices = sample_sinkhorn_coupling_indices(plan, source_mass)

    torch.testing.assert_close(indices, torch.tensor([[0, 2]]))
    assert indices.device == plan.device


def test_random_coupling_reuses_sampled_particles_across_feature_blocks() -> None:
    """Verify RWR gathers one physical target index set for every feature block.

    Returns:
        Nothing. The test checks multi-prompt target values, target metadata,
        RWR diagnostics, and that Sinkhorn is never called.
    """
    config = AlignmentConfig(
        reward=RewardConfig(enabled=True),
        features=FeatureConfig(names=("cls", "patch_mean")),
        objective=ObjectiveConfig(
            name="rwtd",
            current_samples=2,
            reference_samples=2,
            feature_weights=(1.0, 1.0),
        ),
        rwtd=RWTDConfig(
            coupling="random",
            mass_floor=0.0,
            reference_fraction=0.5,
            transport_step=1.0,
        ),
    )
    config.validate()
    objective = build_objective(config)
    current_cls = torch.tensor([[[0.0]], [[1.0]], [[2.0]], [[3.0]]])
    reference_cls = torch.tensor([[[10.0]], [[11.0]], [[12.0]], [[13.0]]])
    current = _population(
        "current",
        current_cls,
        torch.tensor([0.0, 1.0, 2.0, 3.0]),
    )
    reference = _population(
        "reference",
        reference_cls,
        torch.tensor([1.0, 0.0, 3.0, 2.0]),
    )
    current.features["patch_mean"] = current_cls + 100.0
    reference.features["patch_mean"] = reference_cls + 100.0
    sampled_indices = torch.tensor([[0, 3], [2, 1]])

    with (
        patch(
            "src.alignment.objectives.rwtd.sample_random_coupling_indices",
            return_value=sampled_indices,
        ),
        patch(
            "src.alignment.objectives.rwtd.sinkhorn_plan",
            side_effect=AssertionError("RWR must not invoke Sinkhorn"),
        ),
    ):
        prepared = objective.prepare_targets(
            PromptBatch(("prompt-a", "prompt-b")),
            {"current": current, "reference": reference},
        )

    expected_cls = torch.tensor([[0.0], [11.0], [12.0], [3.0]])
    torch.testing.assert_close(prepared.targets["cls"], expected_cls)
    torch.testing.assert_close(
        prepared.targets["patch_mean"],
        expected_cls + 100.0,
    )
    assert prepared.targets["cls"].shape == (4, 1)
    assert prepared.targets["cls"].dtype == current_cls.dtype
    assert prepared.targets["cls"].device == current_cls.device
    assert not prepared.targets["cls"].requires_grad
    assert "rwr/coupling_cost" in prepared.metrics
    assert "rwr/target_ess" in prepared.metrics
    assert not any(key.startswith("rwtd/ot_") for key in prepared.metrics)


def test_sampled_rwtd_reuses_conditional_targets_across_feature_blocks() -> None:
    """Verify sampled RWTD gathers one OT-conditional particle in all blocks.

    Returns:
        Nothing. The test checks sampled support values, detached target
        metadata, and preservation of Sinkhorn diagnostics.
    """
    config = AlignmentConfig(
        reward=RewardConfig(enabled=True),
        features=FeatureConfig(names=("cls", "patch_mean")),
        objective=ObjectiveConfig(
            name="rwtd",
            current_samples=2,
            reference_samples=2,
            feature_weights=(1.0, 1.0),
        ),
        rwtd=RWTDConfig(
            coupling="sinkhorn",
            sinkhorn_target="sampled",
            mass_floor=0.0,
            reference_fraction=0.5,
            transport_step=1.0,
            sinkhorn_iterations=200,
        ),
    )
    config.validate()
    objective = build_objective(config)
    current_cls = torch.tensor([[[0.0]], [[1.0]], [[2.0]], [[3.0]]])
    reference_cls = torch.tensor([[[10.0]], [[11.0]], [[12.0]], [[13.0]]])
    current = _population(
        "current",
        current_cls,
        torch.tensor([0.0, 1.0, 2.0, 3.0]),
    )
    reference = _population(
        "reference",
        reference_cls,
        torch.tensor([1.0, 0.0, 3.0, 2.0]),
    )
    current.features["patch_mean"] = current_cls + 100.0
    reference.features["patch_mean"] = reference_cls + 100.0
    sampled_indices = torch.tensor([[0, 3], [2, 1]])

    with patch(
        "src.alignment.objectives.rwtd.sample_sinkhorn_coupling_indices",
        return_value=sampled_indices,
    ):
        prepared = objective.prepare_targets(
            PromptBatch(("prompt-a", "prompt-b")),
            {"current": current, "reference": reference},
        )

    expected_cls = torch.tensor([[0.0], [11.0], [12.0], [3.0]])
    torch.testing.assert_close(prepared.targets["cls"], expected_cls)
    torch.testing.assert_close(
        prepared.targets["patch_mean"],
        expected_cls + 100.0,
    )
    assert prepared.targets["cls"].shape == (4, 1)
    assert prepared.targets["cls"].dtype == current_cls.dtype
    assert prepared.targets["cls"].device == current_cls.device
    assert not prepared.targets["cls"].requires_grad
    assert "rwtd/ot_row_error" in prepared.metrics
    assert "rwtd/sampled_coupling_cost" in prepared.metrics
    assert "rwtd/sampled_unique_targets" in prepared.metrics
    assert not any(key.startswith("rwr/") for key in prepared.metrics)


def test_stochastic_transport_reuses_self_mask_across_feature_blocks() -> None:
    """Verify stochastic transport gates full conditional targets per sample.

    Returns:
        Nothing. The test checks that ``transport_step`` is used as the
        transport probability and that one Bernoulli decision is shared by all
        feature blocks belonging to a physical sample.
    """
    config = AlignmentConfig(
        reward=RewardConfig(enabled=True),
        features=FeatureConfig(names=("cls", "patch_mean")),
        objective=ObjectiveConfig(
            name="rwtd",
            current_samples=2,
            reference_samples=2,
            feature_weights=(1.0, 1.0),
        ),
        rwtd=RWTDConfig(
            sinkhorn_target="sampled",
            stochastic_transport=True,
            mass_floor=0.0,
            reference_fraction=0.5,
            transport_step=0.25,
            sinkhorn_iterations=200,
        ),
    )
    config.validate()
    objective = build_objective(config)
    current_cls = torch.tensor([[[0.0]], [[1.0]], [[2.0]], [[3.0]]])
    reference_cls = torch.tensor([[[10.0]], [[11.0]], [[12.0]], [[13.0]]])
    current = _population(
        "current",
        current_cls,
        torch.tensor([0.0, 1.0, 2.0, 3.0]),
    )
    reference = _population(
        "reference",
        reference_cls,
        torch.tensor([1.0, 0.0, 3.0, 2.0]),
    )
    current.features["patch_mean"] = current_cls + 100.0
    reference.features["patch_mean"] = reference_cls + 100.0

    with (
        patch(
            "src.alignment.objectives.rwtd.sample_sinkhorn_coupling_indices",
            return_value=torch.tensor([[2, 3], [2, 3]]),
        ),
        patch(
            "src.alignment.objectives.rwtd.torch.rand",
            return_value=torch.tensor([[[0.10], [0.90]], [[0.20], [0.80]]]),
        ),
    ):
        prepared = objective.prepare_targets(
            PromptBatch(("prompt-a", "prompt-b")),
            {"current": current, "reference": reference},
        )

    expected_cls = torch.tensor([[10.0], [1.0], [12.0], [3.0]])
    torch.testing.assert_close(prepared.targets["cls"], expected_cls)
    torch.testing.assert_close(
        prepared.targets["patch_mean"],
        expected_cls + 100.0,
    )
    assert prepared.metrics["rwtd/transport_fraction"] == pytest.approx(0.5)


def test_rwtd_objective_backpropagates_only_through_current_features() -> None:
    """Verify RWTD produces a live loss from detached reward/transport targets.

    Returns:
        Nothing. The test checks requests, diagnostics, and gradient boundaries.
    """
    config = AlignmentConfig(
        reward=RewardConfig(enabled=True),
        features=FeatureConfig(names=("cls",)),
        objective=ObjectiveConfig(
            name="rwtd",
            current_samples=3,
            reference_samples=2,
            feature_weights=(1.0,),
        ),
        rwtd=RWTDConfig(
            reward_temperature=0.5,
            mass_floor=0.1,
            reference_fraction=0.2,
            transport_step=0.5,
            sinkhorn_iterations=200,
        ),
    )
    config.validate()
    objective = build_objective(config)
    current_features = torch.tensor(
        [[[0.0, 0.0]], [[1.0, 0.0]], [[2.0, 0.0]]],
        requires_grad=True,
    )
    reference_features = torch.tensor([[[0.5, 1.0]], [[1.5, 1.0]]])
    populations = {
        "current": _population(
            "current",
            current_features,
            torch.tensor([0.0, 1.0, 2.0]),
        ),
        "reference": _population(
            "reference",
            reference_features,
            torch.tensor([1.5, 0.5]),
        ),
    }

    result = objective.compute(PromptBatch(("prompt",)), populations)
    result.loss.backward()

    requests = objective.rollout_requests(config)
    assert requests[0].requires_rewards and requests[1].requires_rewards
    assert result.loss > 0
    assert current_features.grad is not None
    assert torch.isfinite(current_features.grad).all()
    assert not reference_features.requires_grad
    assert result.metrics["rwtd/reward_temperature"] == 0.5
    assert result.metrics["rwtd/ot_row_error"] < 1e-4
    assert result.metrics["rwtd/ot_column_error"] < 1e-4


def test_rwr_objective_backpropagates_only_through_current_features() -> None:
    """Verify random-coupling targets stay detached from the live regression.

    Returns:
        Nothing. The test checks finite current-feature gradients and RWR
        diagnostics under fixed-seed categorical sampling.
    """
    config = AlignmentConfig(
        reward=RewardConfig(enabled=True),
        features=FeatureConfig(names=("cls",)),
        objective=ObjectiveConfig(
            name="rwtd",
            current_samples=3,
            reference_samples=2,
            feature_weights=(1.0,),
        ),
        rwtd=RWTDConfig(
            coupling="random",
            mass_floor=0.1,
            reference_fraction=0.2,
            transport_step=0.5,
        ),
    )
    config.validate()
    objective = build_objective(config)
    current_features = torch.tensor(
        [[[0.0, 0.0]], [[1.0, 0.0]], [[2.0, 0.0]]],
        requires_grad=True,
    )
    reference_features = torch.tensor([[[0.5, 1.0]], [[1.5, 1.0]]])
    populations = {
        "current": _population(
            "current",
            current_features,
            torch.tensor([0.0, 1.0, 2.0]),
        ),
        "reference": _population(
            "reference",
            reference_features,
            torch.tensor([1.5, 0.5]),
        ),
    }
    torch.manual_seed(7)

    result = objective.compute(PromptBatch(("prompt",)), populations)
    result.loss.backward()

    assert result.loss > 0
    assert current_features.grad is not None
    assert torch.isfinite(current_features.grad).all()
    assert not reference_features.requires_grad
    assert result.metrics["rwr/reward_temperature"] == 0.5
    assert "rwr/unique_targets" in result.metrics
    assert "rwtd/ot_row_error" not in result.metrics


@pytest.mark.parametrize(
    ("coupling", "sinkhorn_target"),
    (
        ("sinkhorn", "barycentric"),
        ("sinkhorn", "sampled"),
        ("random", "barycentric"),
    ),
)
def test_chunked_rwtd_loss_matches_full_loss_and_gradients(
    coupling: str,
    sinkhorn_target: str,
) -> None:
    """Verify detached-target chunks reproduce the complete regression.

    Args:
        coupling: Coupling mode used to prepare the detached targets.
        sinkhorn_target: Barycentric or sampled Sinkhorn target construction.

    Returns:
        Nothing. Summed chunk losses and gradients must match a full-population
        differentiable loss using the same prepared coupled targets.
    """
    config = AlignmentConfig(
        reward=RewardConfig(enabled=True),
        features=FeatureConfig(names=("cls",)),
        objective=ObjectiveConfig(
            name="rwtd",
            current_samples=3,
            reference_samples=2,
            feature_weights=(1.0,),
        ),
        rwtd=RWTDConfig(
            coupling=coupling,
            sinkhorn_target=sinkhorn_target,
            reference_fraction=0.2,
            transport_step=0.5,
            sinkhorn_iterations=200,
        ),
    )
    config.validate()
    objective = build_objective(config)
    values = torch.tensor(
        [[[0.0, 0.0]], [[1.0, 0.5]], [[2.0, 1.0]]],
    )
    populations = {
        "current": _population(
            "current",
            values,
            torch.tensor([0.0, 1.0, 2.0]),
        ),
        "reference": _population(
            "reference",
            torch.tensor([[[0.5, 1.0]], [[1.5, 0.0]]]),
            torch.tensor([1.5, 0.5]),
        ),
    }
    prepared = objective.prepare_targets(PromptBatch(("prompt",)), populations)

    full_features = values.clone().requires_grad_(True)
    full_loss = objective.chunk_loss(
        {"cls": full_features},
        prepared,
        start=0,
        stop=3,
    )
    full_loss.backward()

    chunk_gradients = []
    chunk_losses = []
    for start, stop in ((0, 1), (1, 3)):
        chunk_features = values[start:stop].clone().requires_grad_(True)
        loss = objective.chunk_loss(
            {"cls": chunk_features},
            prepared,
            start=start,
            stop=stop,
        )
        loss.backward()
        chunk_losses.append(loss.detach())
        chunk_gradients.append(chunk_features.grad)

    torch.testing.assert_close(torch.stack(chunk_losses).sum(), full_loss.detach())
    torch.testing.assert_close(torch.cat(chunk_gradients), full_features.grad)
