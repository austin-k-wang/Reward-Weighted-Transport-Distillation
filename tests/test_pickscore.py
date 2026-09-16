"""Tests for the PickScore inference wrapper without network model downloads."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image

from src import pickscore as pickscore_module


class FakeProcessor:
    """Encode simple red/green test images and matching words as RGB vectors."""

    def __call__(
        self,
        *,
        images: list[Image.Image] | None = None,
        text: list[str] | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Convert fake processor inputs to deterministic feature tensors.

        Args:
            images: Optional RGB PIL image batch.
            text: Optional prompt batch containing ``red`` or ``green``.
            **_: Additional processor options ignored by this fake.

        Returns:
            A dictionary containing ``pixel_values`` for images or
            ``input_ids`` for text, with shape ``(N, 3)``.
        """
        if images is not None:
            values = [
                torch.tensor(image.getpixel((0, 0)), dtype=torch.float32) / 255
                for image in images
            ]
            return {"pixel_values": torch.stack(values)}
        assert text is not None
        vectors = {
            "red": torch.tensor([1.0, 0.0, 0.0]),
            "green": torch.tensor([0.0, 1.0, 0.0]),
        }
        return {"input_ids": torch.stack([vectors[prompt] for prompt in text])}


class FakePickScoreModel(torch.nn.Module):
    """Expose processor vectors directly as normalized-model feature inputs."""

    def __init__(self) -> None:
        """Initialize a fixed PickScore logit scale of ten.

        Args:
            None.

        Returns:
            Nothing. The fake model contains one frozen-compatible parameter.
        """
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.tensor(10.0).log())

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return image vectors unchanged.

        Args:
            pixel_values: Fake image feature tensor with shape ``(N, 3)``.

        Returns:
            The same tensor with shape ``(N, 3)``.
        """
        return pixel_values

    def get_text_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return text vectors unchanged.

        Args:
            input_ids: Fake text feature tensor with shape ``(N, 3)``.

        Returns:
            The same tensor with shape ``(N, 3)``.
        """
        return input_ids


def build_scorer(monkeypatch: Any) -> pickscore_module.PickScore:
    """Construct a CPU PickScore wrapper backed by deterministic fake objects.

    Args:
        monkeypatch: Pytest fixture used to replace Hugging Face auto-loading,
            preventing network and checkpoint access.

    Returns:
        A ready ``PickScore`` instance whose embeddings are RGB vectors.
    """
    monkeypatch.setattr(
        pickscore_module,
        "AutoProcessor",
        SimpleNamespace(from_pretrained=lambda _: FakeProcessor()),
    )
    monkeypatch.setattr(
        pickscore_module,
        "AutoModel",
        SimpleNamespace(from_pretrained=lambda _: FakePickScoreModel()),
    )
    return pickscore_module.PickScore(device="cpu")


def test_score_accepts_sana_tensor_batch(monkeypatch: Any) -> None:
    """Verify SANA-range tensors produce one CPU float32 score per prompt.

    Args:
        monkeypatch: Pytest fixture used to install the fake processor/model.

    Returns:
        Nothing. The test asserts output shape, device, dtype, and raw logits
        for a ``(N, C, H, W)`` tensor batch.
    """
    scorer = build_scorer(monkeypatch)
    images = torch.tensor(
        [
            [[[1.0]], [[-1.0]], [[-1.0]]],
            [[[-1.0]], [[1.0]], [[-1.0]]],
        ]
    )

    scores = scorer.score(["red", "green"], images, show_progress=False)

    assert scores.shape == (2,)
    assert scores.device.type == "cpu"
    assert scores.dtype == torch.float32
    torch.testing.assert_close(scores, torch.tensor([10.0, 10.0]))


def test_score_broadcasts_prompt_and_returns_preferences(monkeypatch: Any) -> None:
    """Verify one prompt broadcasts and softmax ranks matching candidates.

    Args:
        monkeypatch: Pytest fixture used to install the fake processor/model.

    Returns:
        Nothing. The test checks that probabilities have shape ``(2,)``, sum
        to one, and favor the red image for the prompt ``red``.
    """
    scorer = build_scorer(monkeypatch)
    images = torch.tensor(
        [
            [[[1.0]], [[-1.0]], [[-1.0]]],
            [[[-1.0]], [[1.0]], [[-1.0]]],
        ]
    )

    probabilities = scorer.preference_probabilities(
        "red",
        images,
        batch_size=1,
        show_progress=False,
    )

    assert probabilities.shape == (2,)
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))
    assert probabilities[0] > probabilities[1]
