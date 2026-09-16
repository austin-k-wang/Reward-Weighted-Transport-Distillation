"""Tests for alignment configuration and objective contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.alignment.config import (
    AlignmentConfig,
    FeatureConfig,
    ObjectiveConfig,
    RWTDConfig,
    RewardConfig,
    apply_alignment_overrides,
    load_alignment_config,
)
from src.alignment.objectives import build_objective


def test_load_alignment_config_resolves_nested_defaults(tmp_path: Path) -> None:
    """Verify a partial YAML file becomes a validated complete configuration.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Nothing. The test checks values and immutable tuple conversion.
    """
    config_path = tmp_path / "alignment.yaml"
    config_path.write_text(
        """
model:
  resolution: 512
objective:
  current_samples: 3
  reference_samples: 3
features:
  names: [cls]
runtime:
  max_train_steps: 2
logging:
  checkpointing_steps: 2
""",
        encoding="utf-8",
    )

    config = load_alignment_config(config_path)

    assert config.model.resolution == 512
    assert config.objective.current_samples == 3
    assert config.features.names == ("cls",)
    assert config.lora.rank == 32
    assert config.lora.alpha == 32
    assert config.objective.name == "rwtd"
    assert config.reward.enabled


def test_alignment_overrides_parse_fixed_rwtd_temperature() -> None:
    """Verify shell-style overrides preserve types and validate the result.

    Returns:
        Nothing. The test checks scalar and tuple-valued dotted assignments.
    """
    config = AlignmentConfig()

    updated = apply_alignment_overrides(
        config,
        [
            "rwtd.reward_temperature=0.5",
            "features.names=[cls,patch_mean]",
            "runtime.max_train_steps=7",
        ],
    )

    assert updated.rwtd.reward_temperature == 0.5
    assert updated.features.names == ("cls", "patch_mean")
    assert updated.runtime.max_train_steps == 7


@pytest.mark.parametrize("reference_fraction", [0.0, 1.0])
def test_rwtd_reference_fraction_accepts_closed_interval_endpoints(
    reference_fraction: float,
) -> None:
    """Verify RWTD permits current-only and frozen-reference-only target mass.

    Args:
        reference_fraction: Endpoint target-mass fraction, either ``0`` for
            current support only or ``1`` for frozen-reference support only.

    Returns:
        Nothing. Successful validation confirms that both endpoint controls are
        available to reference-mass sweeps.
    """
    config = AlignmentConfig(
        reward=RewardConfig(enabled=True),
        features=FeatureConfig(enabled=True),
        objective=ObjectiveConfig(name="rwtd"),
        rwtd=RWTDConfig(reference_fraction=reference_fraction),
    )

    config.validate()


@pytest.mark.parametrize("reference_fraction", [-0.01, 1.01])
def test_rwtd_reference_fraction_rejects_values_outside_closed_interval(
    reference_fraction: float,
) -> None:
    """Verify RWTD rejects reference target mass outside probability bounds.

    Args:
        reference_fraction: Invalid target-mass fraction below zero or above
            one.

    Returns:
        Nothing. The test asserts configuration validation raises ``ValueError``.
    """
    config = AlignmentConfig(
        reward=RewardConfig(enabled=True),
        features=FeatureConfig(enabled=True),
        objective=ObjectiveConfig(name="rwtd"),
        rwtd=RWTDConfig(reference_fraction=reference_fraction),
    )

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        config.validate()


def test_reward_provider_validation_accepts_wrappers_and_composite() -> None:
    """Verify alignment configuration accepts every unified reward provider.

    Returns:
        Nothing. The test validates individual providers and a minimally
        configured standardized composite reward.
    """
    for provider in (
        "pickscore",
        "geneval",
        "imagereward",
        "hpsv2",
        "clip",
        "laion_aesthetic",
    ):
        AlignmentConfig(reward=RewardConfig(provider=provider)).validate()

    AlignmentConfig(
        reward=RewardConfig(
            provider="composite",
            components=(
                {
                    "provider": "clip",
                    "weight": 1.0,
                    "mean": 0.0,
                    "scale": 1.0,
                },
            ),
        )
    ).validate()


def test_composite_reward_validation_rejects_invalid_calibration() -> None:
    """Verify composite components require fixed positive calibration scales.

    Returns:
        Nothing. The test asserts invalid nested component configuration is
        rejected before any reward checkpoints are loaded.
    """
    config = AlignmentConfig(
        reward=RewardConfig(
            provider="composite",
            components=(
                {
                    "provider": "clip",
                    "weight": 1.0,
                    "mean": 0.0,
                    "scale": 0.0,
                },
            ),
        )
    )

    with pytest.raises(ValueError, match="scale must be positive"):
        config.validate()


def test_checked_in_hpsv2_rwtd_config_uses_one_step_endpoint() -> None:
    """Verify the HPS v2 training recipe resolves its critical model settings.

    Returns:
        Nothing. The test checks reward provider/checkpoints, one-step
        generation at the pure-noise endpoint, and independent PickScore
        evaluation.
    """
    root = Path(__file__).resolve().parents[2]
    config = load_alignment_config(
        root / "configs/online_alignment/sana_sprint_rwtd_hpsv2.yaml"
    )

    assert config.reward.provider == "hpsv2"
    assert config.features.provider == "hpsv2"
    assert config.features.names == ("hps",)
    assert config.objective.feature_weights == (1.0,)
    assert config.reward.base_model_path.endswith("open_clip_pytorch_model.bin")
    assert config.reward.checkpoint_path.endswith("HPS_v2.1_compressed.pt")
    assert config.model.num_inference_steps == 1
    assert config.model.max_timesteps == pytest.approx(1.57080)
    assert config.evaluation.provider == "pickscore"
    assert config.rwtd.coupling == "sinkhorn"
    assert config.rwtd.sinkhorn_target == "barycentric"
    assert config.rwtd.reward_mean == pytest.approx(0.3031447375430634)
    assert config.rwtd.reward_scale == pytest.approx(0.03406016468398281)


def test_rwtd_coupling_validation_is_backward_compatible() -> None:
    """Verify random coupling bypasses only Sinkhorn-specific validation.

    Returns:
        Nothing. Existing defaults remain Sinkhorn, unknown modes fail, and
        random coupling accepts unused non-positive Sinkhorn parameters.
    """
    assert RWTDConfig().coupling == "sinkhorn"
    assert RWTDConfig().sinkhorn_target == "barycentric"
    assert not RWTDConfig().stochastic_transport
    common = {
        "reward": RewardConfig(enabled=True),
        "features": FeatureConfig(enabled=True),
        "objective": ObjectiveConfig(name="rwtd"),
    }

    with pytest.raises(ValueError, match="rwtd.coupling"):
        AlignmentConfig(
            **common,
            rwtd=RWTDConfig(coupling="unknown"),
        ).validate()

    with pytest.raises(ValueError, match="rwtd.sinkhorn_target"):
        AlignmentConfig(
            **common,
            rwtd=RWTDConfig(sinkhorn_target="unknown"),
        ).validate()

    AlignmentConfig(
        **common,
        rwtd=RWTDConfig(sinkhorn_target="sampled"),
    ).validate()

    AlignmentConfig(
        **common,
        rwtd=RWTDConfig(
            coupling="random",
            ot_regularization_scale=0.0,
            minimum_ot_epsilon=0.0,
            sinkhorn_iterations=0,
            sinkhorn_tolerance=0.0,
        ),
    ).validate()

    with pytest.raises(ValueError, match="ot_regularization_scale"):
        AlignmentConfig(
            **common,
            rwtd=RWTDConfig(
                coupling="sinkhorn",
                ot_regularization_scale=0.0,
            ),
        ).validate()
