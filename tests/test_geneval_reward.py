"""Unit tests for metadata-aware online GenEval reward integration."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.alignment.data import PromptFileDataset, collate_prompts
from src.geneval.metadata import MetadataError, validate_metadata
from src.geneval.protocol import receive_frame, send_frame
from src.geneval.reward import GenEvalReward
from src.geneval.scoring import score_detections


def _single_object_row() -> dict[str, Any]:
    """Build one valid single-object metadata row.

    Returns:
        Structured metadata requesting exactly one detected car.
    """
    return {
        "prompt": "a photo of a car",
        "tag": "single_object",
        "include": [{"class": "car", "count": 1}],
        "exclude": [],
    }


class FakeClient:
    """Record adapter requests and return deterministic dense rewards."""

    def __init__(self) -> None:
        """Initialize request and diagnostic traces.

        Returns:
            Nothing. Empty mutable traces are retained by the fake.
        """
        self.requests: list[tuple[np.ndarray, list[dict[str, Any]], str, float]] = []
        self.last_diagnostics: dict[str, Any] = {}

    def score(
        self,
        images: np.ndarray,
        metadata: list[dict[str, Any]],
        *,
        reward_mode: str,
        binary_bonus: float,
    ) -> list[float]:
        """Return image means while retaining exact request inputs.

        Args:
            images: RGB uint8 batch shaped ``[N,H,W,3]``.
            metadata: Structured rows aligned with ``images``.
            reward_mode: Requested GenEval reward mode.
            binary_bonus: Requested hybrid correctness bonus.

        Returns:
            Mean normalized pixel intensity for each image, shape ``[N]``.
        """
        self.requests.append(
            (images.copy(), list(metadata), reward_mode, binary_bonus)
        )
        self.last_diagnostics = {"count": len(images)}
        return images.astype(np.float32).mean(axis=(1, 2, 3)).tolist()

    def close(self) -> None:
        """Release no-op fake client resources.

        Returns:
            Nothing.
        """


def test_metadata_jsonl_dataset_preserves_structured_rows(tmp_path: Path) -> None:
    """Verify JSONL loading and collation retain nested GenEval metadata.

    Args:
        tmp_path: Temporary directory receiving a one-row JSONL file.

    Returns:
        Nothing. Prompt and metadata alignment are asserted.
    """
    path = tmp_path / "train.jsonl"
    row = _single_object_row()
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    dataset = PromptFileDataset(path)
    batch = collate_prompts([dataset[0]])

    assert batch.prompts == (row["prompt"],)
    assert batch.metadata[0] == validate_metadata(row)


def test_metadata_validation_rejects_missing_structure() -> None:
    """Verify prompt-only mappings cannot enter GenEval scoring.

    Returns:
        Nothing. Invalid metadata must raise an actionable error.
    """
    try:
        validate_metadata({"prompt": "a car"})
    except MetadataError as exc:
        assert "tag" in str(exc)
    else:
        raise AssertionError("prompt-only metadata unexpectedly passed validation")


def test_protocol_round_trip_preserves_header_and_payload() -> None:
    """Verify versioned socket framing round-trips image bytes.

    Returns:
        Nothing. Header authority and payload identity are asserted.
    """
    sender, receiver = socket.socketpair()
    try:
        send_frame(sender, {"op": "score"}, b"rgb")
        frame = receive_frame(receiver)
    finally:
        sender.close()
        receiver.close()

    assert frame.header["op"] == "score"
    assert frame.header["payload_length"] == 3
    assert frame.payload == b"rgb"


def test_hybrid_scoring_combines_dense_and_binary_credit() -> None:
    """Verify a correct detection receives dense credit plus the bonus.

    Returns:
        Nothing. Exact binary and hybrid arithmetic are asserted.
    """
    score = score_detections(
        _single_object_row(),
        {"car": np.asarray([[1, 2, 20, 30, 0.8]], dtype=np.float32)},
        image_size=(32, 32),
        reward_mode="hybrid",
        binary_bonus=0.25,
    )

    assert score.official_correct
    assert np.isclose(score.dense, 0.8)
    assert np.isclose(score.reward, 1.05)


def test_reward_adapter_converts_images_and_chunks_requests() -> None:
    """Verify SANA tensors become aligned uint8 RGB GenEval requests.

    Returns:
        Nothing. Shape, range, chunking, metadata, and reward shape are checked.
    """
    reward = GenEvalReward.__new__(GenEvalReward)
    reward.reward_mode = "hybrid"
    reward.binary_bonus = 0.25
    reward.tensor_value_range = (-1.0, 1.0)
    reward.client = FakeClient()
    reward.last_diagnostics = {}
    rows = [_single_object_row(), _single_object_row()]
    images = torch.stack(
        (
            torch.full((3, 2, 2), -1.0),
            torch.full((3, 2, 2), 1.0),
        )
    )

    values = reward.score(
        (rows[0]["prompt"], rows[1]["prompt"]),
        images,
        batch_size=1,
        metadata=rows,
    )

    assert values.shape == (2,)
    assert len(reward.client.requests) == 2
    assert reward.client.requests[0][0].shape == (1, 2, 2, 3)
    assert reward.client.requests[0][0].dtype == np.uint8
    assert reward.client.requests[0][0].max() == 0
    assert reward.client.requests[1][0].min() == 255
    assert reward.client.requests[0][1][0]["tag"] == "single_object"

    reward.score_with_mode(
        (rows[0]["prompt"], rows[1]["prompt"]),
        images,
        batch_size=2,
        metadata=rows,
        reward_mode="binary",
        binary_bonus=0.0,
    )
    assert reward.client.requests[-1][2:] == ("binary", 0.0)
