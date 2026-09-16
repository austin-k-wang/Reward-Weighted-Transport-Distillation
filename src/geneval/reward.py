"""Training-facing adapter for the persistent GenEval reward service."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .client import GenEvalRewardClient


class GenEvalReward:
    """Score generated image tensors through a rank-local GenEval server.

    Args:
        socket_path: Unix-domain socket exposed by the local scorer process.
        timeout: Per-request socket timeout in seconds.
        startup_timeout: Maximum readiness wait in seconds during construction.
        reward_mode: One of ``binary``, ``dense``, or ``hybrid``.
        binary_bonus: Non-negative correctness bonus used by hybrid rewards.
        warmup: Whether to request an additional server warmup after readiness.
        tensor_value_range: Minimum and maximum represented by floating image
            tensors. SANA decoded images use ``(-1, 1)``.

    Returns:
        A reusable evaluator whose :meth:`score` method returns detached CPU
        float32 rewards shaped ``[N]``.
    """

    def __init__(
        self,
        socket_path: str,
        *,
        timeout: float = 300.0,
        startup_timeout: float = 900.0,
        reward_mode: str = "hybrid",
        binary_bonus: float = 0.25,
        warmup: bool = False,
        tensor_value_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        """Initialize the client and wait for the scorer to become ready.

        Args:
            socket_path: Rank-local Unix-domain socket path.
            timeout: Per-request socket timeout in seconds.
            startup_timeout: Maximum server-readiness wait in seconds.
            reward_mode: ``binary``, ``dense``, or ``hybrid``.
            binary_bonus: Correct-image bonus used by hybrid mode.
            warmup: Whether to request warmup after the health check.
            tensor_value_range: Inclusive floating tensor value range.

        Returns:
            Nothing. The instance retains a connected, reusable client.
        """
        if reward_mode not in {"binary", "dense", "hybrid"}:
            raise ValueError("reward_mode must be binary, dense, or hybrid")
        if binary_bonus < 0:
            raise ValueError("binary_bonus must be non-negative")
        range_min, range_max = tensor_value_range
        if range_max <= range_min:
            raise ValueError("tensor_value_range maximum must exceed its minimum")
        self.reward_mode = reward_mode
        self.binary_bonus = float(binary_bonus)
        self.tensor_value_range = (float(range_min), float(range_max))
        self.client = GenEvalRewardClient(socket_path, timeout=timeout)
        self.client.wait_until_ready(timeout=startup_timeout)
        if warmup:
            self.client.warmup()
        self.last_diagnostics: dict[str, Any] = {}

    def _to_uint8_rgb(self, images: torch.Tensor) -> np.ndarray:
        """Convert decoded image tensors to contiguous GenEval input arrays.

        Args:
            images: Floating image tensor shaped ``[N,3,H,W]`` in the configured
                value range.

        Returns:
            Contiguous RGB NumPy array shaped ``[N,H,W,3]`` with dtype uint8.
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"GenEval images must have shape [N,3,H,W], got {tuple(images.shape)}"
            )
        range_min, range_max = self.tensor_value_range
        scaled = (
            images.detach()
            .float()
            .clamp(range_min, range_max)
            .sub(range_min)
            .div(range_max - range_min)
            .mul(255.0)
            .round()
            .to(dtype=torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
        )
        return np.ascontiguousarray(scaled)

    def score(
        self,
        prompts: Sequence[str],
        images: torch.Tensor,
        *,
        batch_size: int,
        metadata: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> torch.Tensor:
        """Return one detached GenEval reward per prompt-image pair.

        Args:
            prompts: Prompt strings aligned one-to-one with ``images``.
            images: Decoded image tensor shaped ``[N,3,H,W]``.
            batch_size: Maximum number of images sent in one server request.
            metadata: Structured GenEval rows aligned with ``prompts`` and
                ``images``. Every entry must be present.

        Returns:
            CPU float32 reward tensor shaped ``[N]``.

        Raises:
            ValueError: If cardinalities, metadata, or image shapes are invalid.
        """
        return self.score_with_mode(
            prompts,
            images,
            batch_size=batch_size,
            metadata=metadata,
            reward_mode=self.reward_mode,
            binary_bonus=self.binary_bonus,
        )

    def score_with_mode(
        self,
        prompts: Sequence[str],
        images: torch.Tensor,
        *,
        batch_size: int,
        metadata: Sequence[Mapping[str, Any] | None] | None,
        reward_mode: str,
        binary_bonus: float,
    ) -> torch.Tensor:
        """Score images with an explicit mode on the existing server connection.

        This permits periodic binary GenEval evaluation to reuse the rank-zero
        training scorer even when online training uses dense or hybrid rewards.

        Args:
            prompts: Prompt strings aligned one-to-one with ``images``.
            images: Decoded image tensor shaped ``[N,3,H,W]``.
            batch_size: Maximum images sent in one server request.
            metadata: Non-null structured GenEval rows aligned with images.
            reward_mode: ``binary``, ``dense``, or ``hybrid``.
            binary_bonus: Non-negative exact-correctness bonus for hybrid mode.

        Returns:
            CPU float32 reward tensor shaped ``[N]``.

        Raises:
            ValueError: If the mode, bonus, cardinalities, metadata, or image
                shapes are invalid.
        """
        if reward_mode not in {"binary", "dense", "hybrid"}:
            raise ValueError("reward_mode must be binary, dense, or hybrid")
        if binary_bonus < 0:
            raise ValueError("binary_bonus must be non-negative")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(prompts) != images.shape[0]:
            raise ValueError(
                f"received {len(prompts)} prompts for {images.shape[0]} images"
            )
        if metadata is None or len(metadata) != images.shape[0]:
            count = 0 if metadata is None else len(metadata)
            raise ValueError(
                f"GenEval requires one metadata row per image; received {count} "
                f"for {images.shape[0]} images"
            )
        if any(row is None for row in metadata):
            raise ValueError("GenEval metadata rows must not be null")
        arrays = self._to_uint8_rgb(images)
        rewards: list[float] = []
        diagnostics: list[dict[str, Any]] = []
        for start in range(0, arrays.shape[0], batch_size):
            stop = min(start + batch_size, arrays.shape[0])
            chunk_rows = metadata[start:stop]
            rewards.extend(
                self.client.score(
                    arrays[start:stop],
                    chunk_rows,  # type: ignore[arg-type]
                    reward_mode=reward_mode,
                    binary_bonus=binary_bonus,
                )
            )
            diagnostics.append(dict(self.client.last_diagnostics))
        self.last_diagnostics = {
            "requests": diagnostics,
            "request_count": len(diagnostics),
        }
        return torch.tensor(rewards, dtype=torch.float32)

    def close(self) -> None:
        """Close the local client connection without stopping the server.

        Returns:
            Nothing after the Unix-domain socket connection is released.
        """
        self.client.close()
