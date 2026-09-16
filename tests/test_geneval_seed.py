"""Tests for process-count-invariant GenEval generation seeds."""

from __future__ import annotations

import torch

from Sana.scripts.geneval_seed import (
    SANA_SPRINT_PAPER_4_STEP_TIMESTEPS,
    advance_official_geneval_generator,
    geneval_sample_seed,
    validate_geneval_timesteps,
)


def _seeds_for_shards(shard_count: int) -> dict[tuple[int, int], int]:
    """Simulate prompt sharding and collect global image seeds.

    Args:
        shard_count: Number of contiguous prompt shards to simulate.

    Returns:
        Mapping from ``(prompt_index,image_index)`` to deterministic seed for
        all 553 prompts and four images per prompt.
    """
    seeds: dict[tuple[int, int], int] = {}
    for rank in range(shard_count):
        start = rank * 553 // shard_count
        end = (rank + 1) * 553 // shard_count
        for prompt_index in range(start, end):
            for image_index in range(4):
                seeds[(prompt_index, image_index)] = geneval_sample_seed(
                    42,
                    prompt_index,
                    image_index,
                )
    return seeds


def test_geneval_seeds_are_invariant_to_prompt_sharding() -> None:
    """Verify one- and eight-process generation assign identical image seeds.

    Returns:
        Nothing. Complete seed mappings are compared exactly.
    """
    assert _seeds_for_shards(1) == _seeds_for_shards(8)


def test_geneval_seed_changes_for_each_image() -> None:
    """Verify neighboring prompts and images receive distinct seeds.

    Returns:
        Nothing. Seed uniqueness is checked for a representative subset.
    """
    values = {
        geneval_sample_seed(0, prompt_index, image_index)
        for prompt_index in range(10)
        for image_index in range(4)
    }
    assert len(values) == 40


def test_official_generator_advancement_matches_unsharded_stream() -> None:
    """Verify a shard starts at the official unsharded generator position.

    Returns:
        Nothing. The next latent batch after advancing is compared exactly
        against the corresponding batch from one continuously consumed stream.
    """
    batch_shape = (4, 2, 3, 3)
    unsharded = torch.Generator(device="cpu").manual_seed(0)
    expected = None
    for prompt_index in range(8):
        batch = torch.randn(*batch_shape, generator=unsharded)
        if prompt_index == 5:
            expected = batch

    sharded = torch.Generator(device="cpu").manual_seed(0)
    advance_official_geneval_generator(
        sharded,
        prompt_count=5,
        batch_size=batch_shape[0],
        latent_channels=batch_shape[1],
        latent_size=batch_shape[2],
        device="cpu",
    )
    actual = torch.randn(*batch_shape, generator=sharded)

    assert expected is not None
    assert torch.equal(actual, expected)


def test_paper_four_step_schedule_has_five_boundaries() -> None:
    """Verify Appendix F.2's schedule is accepted for four transitions.

    Returns:
        Nothing. The validated schedule is compared with the published
        optimized timestep boundaries.
    """
    actual = validate_geneval_timesteps(
        4,
        SANA_SPRINT_PAPER_4_STEP_TIMESTEPS,
    )

    assert actual == [
        1.5682963320032104,
        1.3,
        1.1,
        0.6,
        0.0,
    ]


def test_default_geneval_schedule_remains_implicit() -> None:
    """Verify legacy generation still delegates schedule construction.

    Returns:
        Nothing. A missing explicit schedule remains ``None`` for backwards
        compatibility with one- and two-step inference.
    """
    assert validate_geneval_timesteps(1, None) is None
    assert validate_geneval_timesteps(2, None) is None


def test_geneval_schedule_rejects_transition_count_as_boundary_count() -> None:
    """Verify four-step schedules cannot omit the final timestep boundary.

    Returns:
        Nothing. Validation is expected to raise for a four-value schedule,
        which cannot encode all four denoising transitions.
    """
    try:
        validate_geneval_timesteps(4, [1.5682963320032104, 1.3, 1.1, 0.0])
    except ValueError as error:
        assert "require 5 timestep boundaries" in str(error)
    else:
        raise AssertionError("Expected an invalid boundary count to raise")
