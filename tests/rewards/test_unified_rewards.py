"""Mock-based tests for unified rewards with no model or network access."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from src.alignment.interfaces import RewardEvaluator
from src.geneval.reward import GenEvalReward as ExistingGenEvalReward
from src.rewards.aesthetics import LAIONAestheticReward
from src.rewards.base import images_to_pil, images_to_unit_range, validate_reward_inputs
from src.rewards.clip import CLIPReward
from src.rewards.composite import StandardizedWeightedReward
from src.rewards import factory as factory_module
from src.rewards.factory import (
    available_rewards,
    create_reward,
    create_reward_from_config,
    register_reward,
)
from src.rewards.geneval import GenEvalReward
from src.rewards.hps import HPSv21Reward
from src.rewards.imagereward import ImageRewardEvaluator
from src.rewards.pickscore import PickScoreReward


def canonical_images() -> torch.Tensor:
    """Build two tiny canonical red and green image tensors.

    Returns:
        Float32 tensor shaped ``[2,3,1,1]`` with values in ``[-1,1]``.
    """
    return torch.tensor(
        [
            [[[1.0]], [[-1.0]], [[-1.0]]],
            [[[-1.0]], [[1.0]], [[-1.0]]],
        ]
    )


class FakeProcessor:
    """Return deterministic processor tensors for image and text inputs."""

    def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Encode the requested batch as simple sequential tensors.

        Args:
            **kwargs: Processor inputs containing images and optionally text.

        Returns:
            Floating pixels and, for joint calls, integer token IDs.
        """
        count = len(kwargs["images"])
        result = {"pixel_values": torch.arange(count, dtype=torch.float32).unsqueeze(1)}
        if "text" in kwargs:
            result["input_ids"] = torch.arange(count, dtype=torch.long).unsqueeze(1)
        return result


class FakeCLIPModel:
    """Produce diagonal CLIP similarities and deterministic image features."""

    def __call__(self, **inputs: torch.Tensor) -> SimpleNamespace:
        """Return a square logit matrix based on input batch size.

        Args:
            **inputs: Processor tensors whose first dimension is the batch.

        Returns:
            Namespace with ``logits_per_image`` shaped ``[N,N]``.
        """
        count = inputs["pixel_values"].shape[0]
        return SimpleNamespace(logits_per_image=torch.eye(count) * 100)

    def get_image_features(self, **inputs: torch.Tensor) -> torch.Tensor:
        """Return two-dimensional embeddings for aesthetic tests.

        Args:
            **inputs: Processor tensors with image batch dimension ``N``.

        Returns:
            Feature tensor shaped ``[N,2]``.
        """
        count = inputs["pixel_values"].shape[0]
        return torch.tensor([[3.0, 4.0]]).repeat(count, 1)


class FakePickScorer:
    """Record canonical PickScore forwarding and return fixed values."""

    def score(self, prompts: Any, images: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Return one sequential score per forwarded image.

        Args:
            prompts: Forwarded aligned prompts.
            images: Forwarded canonical images shaped ``[N,3,H,W]``.
            **kwargs: Legacy batch, progress, and metadata options.

        Returns:
            Float tensor shaped ``[N]``.
        """
        assert len(prompts) == images.shape[0]
        assert kwargs["show_progress"] is False
        return torch.arange(images.shape[0], device=images.device)


class FakeImageRewardModel:
    """Return a square pairwise matrix in ImageReward's legacy format."""

    def inference_rank(self, prompts: list[str], images: list[Any]) -> tuple[None, list[float]]:
        """Produce flattened diagonal-dominant scores.

        Args:
            prompts: Prompt chunk of length ``N``.
            images: RGB PIL image chunk of length ``N``.

        Returns:
            ``(None, values)`` where values flatten an ``[N,N]`` matrix.
        """
        assert len(prompts) == len(images)
        return None, (torch.eye(len(prompts)) * 2).reshape(-1).tolist()


class FakeHPSModel:
    """Expose HPS visual settings and deterministic normalized features."""

    def __init__(self) -> None:
        """Initialize a small square HPS visual configuration.

        Returns:
            Nothing. ``visual.image_size`` is set to two pixels.
        """
        self.visual = SimpleNamespace(
            image_size=2,
            image_mean=(0.0, 0.0, 0.0),
            image_std=(1.0, 1.0, 1.0),
        )

    def __call__(self, images: torch.Tensor, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return matching identity image and text features.

        Args:
            images: Preprocessed image batch shaped ``[N,3,2,2]``.
            tokens: Token tensor with leading dimension ``N``.

        Returns:
            Mapping containing identity features shaped ``[N,N]``.
        """
        assert images.shape[0] == tokens.shape[0]
        features = torch.eye(images.shape[0])
        return {"image_features": features, "text_features": features}


class ConstantReward:
    """Canonical fake evaluator returning a fixed scalar for every pair."""

    def __init__(self, value: float) -> None:
        """Store the scalar emitted for every image.

        Args:
            value: Constant reward value.

        Returns:
            Nothing.
        """
        self.value = value

    def score(
        self,
        prompts: Any,
        images: torch.Tensor,
        *,
        batch_size: int,
        metadata: Any = None,
    ) -> torch.Tensor:
        """Return constant detached CPU rewards.

        Args:
            prompts: Aligned prompts.
            images: Canonical image tensor shaped ``[N,3,H,W]``.
            batch_size: Forwarded positive batch size.
            metadata: Optional forwarded metadata.

        Returns:
            CPU float32 tensor shaped ``[N]``.
        """
        del prompts, batch_size, metadata
        return torch.full((images.shape[0],), self.value)


def test_base_conversion_and_validation() -> None:
    """Verify canonical range conversion, PIL quantization, and shape checks.

    Returns:
        Nothing. Exact unit-range values and RGB pixels are asserted.
    """
    images = canonical_images()

    assert validate_reward_inputs(("red", "green"), images, 2) == 2
    torch.testing.assert_close(
        images_to_unit_range(images)[:, :, 0, 0],
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    pil_images = images_to_pil(images)
    assert [image.getpixel((0, 0)) for image in pil_images] == [(255, 0, 0), (0, 255, 0)]

    overshot = images.mul(1.05)
    assert validate_reward_inputs(("red", "green"), overshot, 2) == 2
    torch.testing.assert_close(
        images_to_unit_range(overshot)[:, :, 0, 0],
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )


def test_pickscore_adapter_has_canonical_output() -> None:
    """Verify the PickScore adapter forwards canonical options and detaches output.

    Returns:
        Nothing. Canonical values, dtype, and device are asserted.
    """
    reward = PickScoreReward.__new__(PickScoreReward)
    reward.scorer = FakePickScorer()

    result = reward.score(("red", "green"), canonical_images(), batch_size=1)

    torch.testing.assert_close(result, torch.tensor([0.0, 1.0]))
    assert result.dtype == torch.float32 and result.device.type == "cpu"


def test_geneval_is_existing_adapter_reexport() -> None:
    """Verify the unified package preserves the existing GenEval class identity.

    Returns:
        Nothing. Exact class identity is asserted.
    """
    assert GenEvalReward is ExistingGenEvalReward


def test_imagereward_extracts_pairwise_diagonal_in_chunks() -> None:
    """Verify ImageReward converts PIL batches and extracts aligned pair scores.

    Returns:
        Nothing. The extracted canonical reward tensor is asserted.
    """
    reward = ImageRewardEvaluator.__new__(ImageRewardEvaluator)
    reward.model = FakeImageRewardModel()

    result = reward.score(("red", "green"), canonical_images(), batch_size=2)

    torch.testing.assert_close(result, torch.tensor([2.0, 2.0]))


def test_clip_reward_scores_corresponding_pairs() -> None:
    """Verify Hugging Face CLIP uses diagonal scaled logits across chunks.

    Returns:
        Nothing. Pairwise scaled similarities are asserted.
    """
    reward = CLIPReward.__new__(CLIPReward)
    reward.device = torch.device("cpu")
    reward.dtype = torch.float32
    reward.processor = FakeProcessor()
    reward.model = FakeCLIPModel()
    reward.logit_scale_divisor = 100.0

    result = reward.score(("red", "green"), canonical_images(), batch_size=1)

    torch.testing.assert_close(result, torch.ones(2))


def test_hps_v21_reward_scores_corresponding_features() -> None:
    """Verify HPS preprocessing and diagonal similarities remain canonical.

    Returns:
        Nothing. HPS pair scores are asserted after fake preprocessing.
    """
    reward = HPSv21Reward.__new__(HPSv21Reward)
    reward.device = torch.device("cpu")
    reward.dtype = torch.float32
    reward.model = FakeHPSModel()
    reward.tokenizer = lambda prompts: torch.arange(len(prompts)).unsqueeze(1)
    reward.image_size = 2
    reward.mean = (0.0, 0.0, 0.0)
    reward.std = (1.0, 1.0, 1.0)

    result = reward.score(("red", "green"), canonical_images(), batch_size=2)

    torch.testing.assert_close(result, torch.ones(2))


def test_laion_aesthetic_reward_uses_normalized_clip_features() -> None:
    """Verify aesthetic predictions consume normalized image embeddings only.

    Returns:
        Nothing. Predictions from known normalized embeddings are asserted.
    """
    reward = LAIONAestheticReward.__new__(LAIONAestheticReward)
    reward.device = torch.device("cpu")
    reward.dtype = torch.float32
    reward.processor = FakeProcessor()
    reward.clip_model = FakeCLIPModel()
    reward.predictor = lambda features: features.sum(dim=1, keepdim=True)

    result = reward.score(("ignored", "ignored"), canonical_images(), batch_size=2)

    torch.testing.assert_close(result, torch.tensor([1.4, 1.4]))


def test_composite_applies_fixed_standardization_and_weights() -> None:
    """Verify explicit centers, scales, and normalized weights are deterministic.

    Returns:
        Nothing. Composite and retained component values are asserted.
    """
    reward = StandardizedWeightedReward(
        {"a": ConstantReward(5.0), "b": ConstantReward(1.0)},
        {"a": 2.0, "b": -1.0},
        centers={"a": 1.0},
        scales={"a": 2.0},
        normalize_weights=True,
    )

    result = reward.score(("red", "green"), canonical_images(), batch_size=2)

    torch.testing.assert_close(result, torch.ones(2))
    torch.testing.assert_close(reward.last_components["a"], torch.full((2,), 2.0))
    assert isinstance(reward, RewardEvaluator)


def test_factory_registers_and_constructs_without_backend_imports() -> None:
    """Verify custom construction and built-in registry names are available.

    Returns:
        Nothing. Constructed type and expected lazy names are asserted.
    """
    register_reward("unit_test_constant", ConstantReward, replace=True)

    reward = create_reward("UNIT_TEST_CONSTANT", value=3.0)

    assert isinstance(reward, ConstantReward)
    assert "imagereward" in available_rewards()
    assert "hpsv2.1" in available_rewards()


def test_factory_builds_hps_from_generic_reward_config(monkeypatch: Any) -> None:
    """Verify generic checkpoint fields map to the HPS constructor contract.

    Args:
        monkeypatch: Pytest fixture replacing registry construction.

    Returns:
        Nothing. The test checks the selected provider and translated paths.
    """
    calls = []

    def fake_create(name: str, **kwargs: Any) -> object:
        """Record one factory construction request.

        Args:
            name: Registered provider name.
            **kwargs: Translated provider-specific constructor options.

        Returns:
            Opaque object representing the constructed evaluator.
        """
        calls.append((name, kwargs))
        return object()

    monkeypatch.setattr(factory_module, "create_reward", fake_create)
    create_reward_from_config(
        {
            "provider": "hpsv2",
            "base_model_path": "base.bin",
            "checkpoint_path": "hps.pt",
        },
        device="cpu",
    )

    assert calls[0][0] == "hpsv2"
    assert calls[0][1]["model_checkpoint_path"] == "base.bin"
    assert calls[0][1]["hps_checkpoint_path"] == "hps.pt"
