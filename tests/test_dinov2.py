"""Tests for the DINOv2 feature wrapper without checkpoint downloads."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image

from src import dinov2 as dinov2_module


class FakeImageProcessor:
    """Convert RGB PIL images into tiny normalized image tensors."""

    image_mean = [0.0, 0.0, 0.0]
    image_std = [1.0, 1.0, 1.0]
    crop_size = {"height": 2, "width": 2}

    def __call__(
        self,
        *,
        images: list[Image.Image],
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        """Convert each image's first pixel into a constant 2x2 RGB tensor.

        Args:
            images: Batch of RGB PIL images.
            return_tensors: Expected tensor output convention; must be ``pt``.

        Returns:
            Mapping containing ``pixel_values`` shaped ``(B,3,2,2)``.
        """
        assert return_tensors == "pt"
        values = []
        for image in images:
            rgb = torch.tensor(image.getpixel((0, 0)), dtype=torch.float32) / 255
            values.append(rgb.view(3, 1, 1).expand(3, 2, 2))
        return {"pixel_values": torch.stack(values)}


class FakeDINOv2Model(torch.nn.Module):
    """Produce deterministic CLS and patch tokens from input RGB means."""

    def __init__(self) -> None:
        """Initialize one parameter so device and dtype properties are defined.

        Returns:
            Nothing. The fake model is ready for frozen feature extraction.
        """
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        """Construct one CLS token and two patch tokens per image.

        Args:
            pixel_values: Image tensor shaped ``(B,3,H,W)``.

        Returns:
            Namespace whose ``last_hidden_state`` has shape ``(B,3,3)``.
        """
        rgb = pixel_values.mean(dim=(-2, -1)) * self.scale
        tokens = torch.stack((rgb, rgb + 1, rgb + 3), dim=1)
        return SimpleNamespace(last_hidden_state=tokens)


def build_encoder(monkeypatch: Any) -> dinov2_module.DINOv2FeatureEncoder:
    """Build a CPU DINOv2 wrapper backed by deterministic fake components.

    Args:
        monkeypatch: Pytest fixture used to replace Hugging Face auto-loading.

    Returns:
        Frozen DINOv2 wrapper with three-dimensional fake token features.
    """
    monkeypatch.setattr(
        dinov2_module,
        "AutoImageProcessor",
        SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: FakeImageProcessor()
        ),
    )
    monkeypatch.setattr(
        dinov2_module,
        "AutoModel",
        SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: FakeDINOv2Model()
        ),
    )
    return dinov2_module.DINOv2FeatureEncoder(device="cpu")


def test_encode_makes_single_image_feature_extraction_simple(
    monkeypatch: Any,
) -> None:
    """Verify one PIL image produces one CPU CLS feature vector.

    Args:
        monkeypatch: Pytest fixture used to install fake model components.

    Returns:
        Nothing. The test validates the convenient inference API.
    """
    encoder = build_encoder(monkeypatch)
    image = Image.new("RGB", (4, 4), color=(255, 0, 0))

    features = encoder.encode(image)

    assert features.shape == (1, 3)
    assert features.device.type == "cpu"
    assert features.dtype == torch.float32
    torch.testing.assert_close(features, torch.tensor([[1.0, 0.0, 0.0]]))


def test_encode_batches_images_and_combines_feature_blocks(
    monkeypatch: Any,
) -> None:
    """Verify batched tensors support concatenated CLS and patch statistics.

    Args:
        monkeypatch: Pytest fixture used to install fake model components.

    Returns:
        Nothing. The test checks output shape and values across mini-batches.
    """
    encoder = build_encoder(monkeypatch)
    images = torch.tensor(
        [
            [[[1.0]], [[-1.0]], [[-1.0]]],
            [[[-1.0]], [[1.0]], [[-1.0]]],
        ]
    )

    features = encoder.encode(
        images,
        features=("cls", "patch_mean", "patch_std"),
        batch_size=1,
    )

    assert features.shape == (2, 9)
    torch.testing.assert_close(
        features[0],
        torch.tensor([1.0, 0.0, 0.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0]),
    )
    torch.testing.assert_close(
        features[1],
        torch.tensor([0.0, 1.0, 0.0, 2.0, 3.0, 2.0, 1.0, 1.0, 1.0]),
    )


def test_forward_preserves_image_gradients(monkeypatch: Any) -> None:
    """Verify RWTD can backpropagate through frozen DINOv2 image features.

    Args:
        monkeypatch: Pytest fixture used to install fake model components.

    Returns:
        Nothing. The test checks feature block shapes and input gradients.
    """
    encoder = build_encoder(monkeypatch)
    images = torch.zeros((2, 3, 2, 2), requires_grad=True)

    blocks = encoder.vector_features(
        images,
        features=("cls", "patch_mean", "patch_std", "patch"),
    )
    loss = blocks["cls"].sum() + blocks["patch_mean"].sum()
    loss.backward()

    assert blocks["cls"].shape == (2, 1, 3)
    assert blocks["patch_mean"].shape == (2, 1, 3)
    assert blocks["patch_std"].shape == (2, 1, 3)
    assert blocks["patch"].shape == (2, 2, 3)
    assert images.grad is not None
    assert torch.count_nonzero(images.grad) == images.numel()


def test_invalid_feature_name_is_rejected(monkeypatch: Any) -> None:
    """Verify unsupported feature requests fail with an actionable message.

    Args:
        monkeypatch: Pytest fixture used to install fake model components.

    Returns:
        Nothing. The test checks feature-selection validation.
    """
    encoder = build_encoder(monkeypatch)

    try:
        encoder.encode(Image.new("RGB", (1, 1)), features=("unknown",))
    except ValueError as exc:
        assert "Unsupported DINOv2 features" in str(exc)
    else:
        raise AssertionError("Expected an invalid DINOv2 feature to raise")
