"""Tests for reusable HPS feature representations."""

from __future__ import annotations

import pytest
import torch

from src.rewards.hps import HPSv21FeatureEncoder, HPSv21Reward


class _TinyHPS(torch.nn.Module):
    """Provide a deterministic three-dimensional image encoder."""

    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        """Average normalized image pixels spatially.

        Args:
            pixels: HPS-preprocessed images shaped ``[N,3,H,W]``.

        Returns:
            Features shaped ``[N,3]``.
        """

        return pixels.mean(dim=(2, 3))

def test_hps_public_image_features_are_normalized_and_differentiable() -> None:
    """Verify shared HPS image features retain input gradients."""

    reward = HPSv21Reward.__new__(HPSv21Reward)
    reward.device = torch.device("cpu")
    reward.dtype = torch.float32
    reward.model = _TinyHPS()
    reward.tokenizer = lambda prompts: torch.arange(3).expand(len(prompts), -1)
    reward.image_size = 4
    reward.mean = (0.0, 0.0, 0.0)
    reward.std = (1.0, 1.0, 1.0)
    images = torch.zeros(2, 3, 4, 4, requires_grad=True)

    image_features = reward.encode_image_features(images)
    image_features.sum().backward()

    torch.testing.assert_close(image_features.norm(dim=-1), torch.ones(2))
    assert image_features.shape == (2, 3)
    assert images.grad is not None


def test_hps_feature_encoder_exposes_one_differentiable_rwtd_block() -> None:
    """Verify the RWTD adapter reuses HPS and rejects unsupported block names."""

    reward = HPSv21Reward.__new__(HPSv21Reward)
    reward.device = torch.device("cpu")
    reward.dtype = torch.float32
    reward.model = _TinyHPS()
    reward.image_size = 4
    reward.mean = (0.0, 0.0, 0.0)
    reward.std = (1.0, 1.0, 1.0)
    encoder = HPSv21FeatureEncoder(reward)
    images = torch.ones(2, 3, 4, 4, requires_grad=True)

    features = encoder.vector_features(images, ("hps",))
    features["hps"].sum().backward()

    assert features["hps"].shape == (2, 3)
    torch.testing.assert_close(
        features["hps"].norm(dim=-1),
        torch.full((2,), 32.0),
    )
    assert images.grad is not None
    with pytest.raises(ValueError, match="only the 'hps'"):
        encoder.vector_features(images, ("cls",))
