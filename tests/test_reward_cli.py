"""Tests for standalone unified reward scoring and calibration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from scripts.calibrate_rewards import calibration_statistics
from scripts.score_samples import score_rows


class MeanReward:
    """Return the canonical image mean as a deterministic fake reward."""

    def score(
        self,
        prompts: list[str],
        images: torch.Tensor,
        *,
        batch_size: int,
        metadata: Any = None,
    ) -> torch.Tensor:
        """Score each image using its mean canonical pixel value.

        Args:
            prompts: Prompt strings aligned with the image batch.
            images: CPU float tensor shaped ``[N,3,H,W]`` in ``[-1,1]``.
            batch_size: Positive caller batch limit.
            metadata: Optional aligned metadata.

        Returns:
            CPU float rewards shaped ``[N]``.
        """
        del batch_size, metadata
        assert len(prompts) == images.shape[0]
        return images.mean(dim=(1, 2, 3))


def test_score_rows_loads_relative_images_and_preserves_rows(tmp_path: Path) -> None:
    """Verify JSONL-style rows receive aligned scores from relative image paths.

    Args:
        tmp_path: Temporary directory containing two tiny image fixtures.

    Returns:
        Nothing. Preserved fields and exact canonical black/white scores are
        asserted.
    """
    Image.new("RGB", (2, 2), color=(0, 0, 0)).save(tmp_path / "black.png")
    Image.new("RGB", (2, 2), color=(255, 255, 255)).save(tmp_path / "white.png")
    rows = [
        {"prompt": "black", "image": "black.png", "id": 1},
        {"prompt": "white", "image": "white.png", "id": 2},
    ]

    scored = score_rows(
        rows,
        input_dir=tmp_path,
        reward=MeanReward(),
        batch_size=2,
    )

    assert [row["id"] for row in scored] == [1, 2]
    assert [row["reward"] for row in scored] == pytest.approx([-1.0, 1.0])


def test_calibration_statistics_reports_population_scale() -> None:
    """Verify fixed calibration uses population standard deviation.

    Returns:
        Nothing. Mean, scale, count, and median are checked on known values.
    """
    statistics = calibration_statistics(torch.tensor([1.0, 3.0]))

    assert statistics["count"] == 2
    assert statistics["mean"] == pytest.approx(2.0)
    assert statistics["scale"] == pytest.approx(1.0)
    assert statistics["q50"] == pytest.approx(2.0)
