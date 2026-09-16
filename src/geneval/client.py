"""Client proxy for the persistent GenEval reward server."""

from __future__ import annotations

import math
import socket
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .metadata import validate_metadata
from .protocol import ProtocolError, receive_frame, send_frame


class GenEvalRewardClient:
    """Persistent Unix-socket client for online GenEval scoring."""

    def __init__(
        self,
        socket_path: str | Path,
        *,
        timeout: float = 120.0,
        max_payload_bytes: int = 1 << 30,
    ) -> None:
        """Initialize a disconnected client.

        Parameters:
            socket_path: Unix-domain socket exposed by one rank-local scorer.
            timeout: Connect, send, and receive timeout in seconds.
            max_payload_bytes: Maximum accepted image payload size in bytes.

        Returns:
            A client that connects lazily on the first request.
        """

        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.socket_path = str(socket_path)
        self.timeout = float(timeout)
        self.max_payload_bytes = int(max_payload_bytes)
        self._connection: socket.socket | None = None
        self._request_index = 0
        self._lock = threading.Lock()
        self.last_diagnostics: dict[str, Any] = {}

    def _connect(self) -> socket.socket:
        """Open or return the persistent Unix-domain socket.

        Parameters:
            None.

        Returns:
            Connected stream socket configured with the client timeout.
        """

        if self._connection is None:
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.settimeout(self.timeout)
            try:
                connection.connect(self.socket_path)
            except BaseException:
                connection.close()
                raise
            self._connection = connection
        return self._connection

    def close(self) -> None:
        """Close the persistent connection without stopping the server.

        Parameters:
            None.

        Returns:
            ``None`` after local socket resources are released.
        """

        connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()

    def __enter__(self) -> GenEvalRewardClient:
        """Enter a context that closes the client at exit.

        Parameters:
            None.

        Returns:
            This client instance.
        """

        return self

    def __exit__(self, *_: object) -> None:
        """Close the connection when leaving a context.

        Parameters:
            *_: Standard context-manager exception details, ignored here.

        Returns:
            ``None``.
        """

        self.close()

    def _round_trip(
        self, header: Mapping[str, Any], payload: bytes = b""
    ) -> dict[str, Any]:
        """Send one request and validate its matching response envelope.

        Parameters:
            header: Request metadata including an ``op`` field.
            payload: Optional raw uint8 image bytes.

        Returns:
            Decoded response header.
        """

        with self._lock:
            request_id = self._request_index
            self._request_index += 1
            request = dict(header)
            request["request_id"] = request_id
            try:
                connection = self._connect()
                send_frame(
                    connection,
                    request,
                    payload,
                    max_payload_bytes=self.max_payload_bytes,
                )
                response = receive_frame(
                    connection, max_payload_bytes=self.max_payload_bytes
                )
            except (OSError, ProtocolError):
                self.close()
                raise
            if response.payload:
                self.close()
                raise ProtocolError("server response unexpectedly contained a payload")
            if response.header.get("request_id") != request_id:
                self.close()
                raise ProtocolError(
                    "response request_id does not match the outstanding request"
                )
            if not response.header.get("ok", False):
                raise RuntimeError(
                    f"GenEval server rejected {header.get('op')!r}: "
                    f"{response.header.get('error', 'unknown error')}"
                )
            return response.header

    def health(self) -> dict[str, Any]:
        """Query scorer readiness without running detector inference.

        Parameters:
            None.

        Returns:
            Server health response containing readiness and process details.
        """

        return self._round_trip({"op": "health"})

    def warmup(self) -> dict[str, Any]:
        """Request one server-side detector and color-model warm-up.

        Parameters:
            None.

        Returns:
            Server acknowledgement including warm-up latency.
        """

        return self._round_trip({"op": "warmup"})

    def wait_until_ready(
        self, *, timeout: float = 300.0, poll_interval: float = 0.25
    ) -> dict[str, Any]:
        """Wait until the socket exists and the scorer reports readiness.

        Parameters:
            timeout: Overall readiness deadline in seconds.
            poll_interval: Delay between connection attempts in seconds.

        Returns:
            Successful health response from the server.
        """

        if timeout <= 0 or poll_interval <= 0:
            raise ValueError("timeout and poll_interval must be positive")
        deadline = time.monotonic() + timeout
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                response = self.health()
                if response.get("ready"):
                    return response
            except (OSError, ProtocolError, RuntimeError) as exc:
                last_error = exc
            self.close()
            time.sleep(poll_interval)
        raise TimeoutError(
            f"GenEval scorer at {self.socket_path} was not ready within {timeout}s"
        ) from last_error

    def score(
        self,
        images: np.ndarray | Sequence[np.ndarray],
        metadata: Sequence[Mapping[str, Any]],
        *,
        reward_mode: str = "hybrid",
        binary_bonus: float = 0.25,
    ) -> list[float]:
        """Score a uint8 RGB image batch with structured GenEval metadata.

        Parameters:
            images: Contiguous image batch with shape ``[B,H,W,3]`` or a
                sequence of equally sized ``[H,W,3]`` NumPy arrays.
            metadata: One GenEval metadata row per image, shape ``[B]``.
            reward_mode: ``binary``, ``dense``, or ``hybrid``.
            binary_bonus: Non-negative official-correctness bonus used only in
                hybrid mode.

        Returns:
            Detached Python rewards with shape ``[B]``. Detailed server
            diagnostics are available in :attr:`last_diagnostics`.
        """

        if reward_mode not in {"binary", "dense", "hybrid"}:
            raise ValueError("reward_mode must be binary, dense, or hybrid")
        if not math.isfinite(binary_bonus) or binary_bonus < 0:
            raise ValueError("binary_bonus must be finite and non-negative")
        batch = np.asarray(images)
        if batch.ndim == 3:
            batch = batch[None, ...]
        if batch.ndim != 4 or batch.shape[-1] != 3:
            raise ValueError(f"images must have shape [B,H,W,3], got {batch.shape}")
        if batch.dtype != np.uint8:
            raise ValueError(f"images must have dtype uint8, got {batch.dtype}")
        batch = np.ascontiguousarray(batch)
        if len(metadata) != batch.shape[0]:
            raise ValueError(
                f"received {len(metadata)} metadata rows for {batch.shape[0]} images"
            )
        normalized_metadata = [validate_metadata(row) for row in metadata]
        response = self._round_trip(
            {
                "op": "score",
                "shape": list(batch.shape),
                "dtype": "uint8",
                "metadata": normalized_metadata,
                "reward_mode": reward_mode,
                "binary_bonus": float(binary_bonus),
            },
            batch.tobytes(),
        )
        rewards = response.get("rewards")
        if not isinstance(rewards, list) or len(rewards) != batch.shape[0]:
            raise ProtocolError(
                f"server returned invalid rewards shape; expected [{batch.shape[0]}]"
            )
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in rewards
        ):
            raise ProtocolError("server returned non-finite or non-numeric rewards")
        self.last_diagnostics = dict(response.get("diagnostics", {}))
        return [float(value) for value in rewards]

    def shutdown_server(self) -> None:
        """Ask the connected scorer process to shut down gracefully.

        Parameters:
            None.

        Returns:
            ``None`` after the server acknowledges the request.
        """

        self._round_trip({"op": "shutdown"})
        self.close()
