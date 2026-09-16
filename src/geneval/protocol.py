"""Versioned, length-prefixed Unix-socket protocol for GenEval rewards."""

from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass
from typing import Any, Mapping

PROTOCOL_VERSION = 1
DEFAULT_MAX_HEADER_BYTES = 1 << 20
DEFAULT_MAX_PAYLOAD_BYTES = 1 << 30
_LENGTH = struct.Struct("!Q")


class ProtocolError(RuntimeError):
    """Raised when a GenEval IPC frame is malformed or unsupported."""


@dataclass(frozen=True)
class Frame:
    """A decoded protocol frame.

    Parameters:
        header: JSON-compatible metadata decoded from the frame header.
        payload: Optional raw binary payload associated with ``header``.

    Returns:
        Immutable frame containing the decoded header and exact payload bytes.
    """

    header: dict[str, Any]
    payload: bytes


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    """Receive exactly ``size`` bytes from a connected socket.

    Parameters:
        connection: Connected stream socket from which bytes are read.
        size: Number of bytes to receive. Must be non-negative.

    Returns:
        Byte string with shape/length ``[size]``.

    Raises:
        ProtocolError: If the peer closes before the requested bytes arrive.
    """

    if size < 0:
        raise ProtocolError(f"negative receive size: {size}")
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ProtocolError(
                f"connection closed with {remaining} of {size} bytes remaining"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(
    connection: socket.socket,
    header: Mapping[str, Any],
    payload: bytes = b"",
    *,
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> None:
    """Serialize and send one versioned protocol frame.

    Parameters:
        connection: Connected Unix stream socket.
        header: JSON-compatible frame metadata. Reserved fields ``version`` and
            ``payload_length`` are overwritten with authoritative values.
        payload: Raw binary payload, normally contiguous uint8 RGB image bytes
            with shape described by the header.
        max_header_bytes: Maximum encoded JSON header length.
        max_payload_bytes: Maximum raw payload length.

    Returns:
        ``None`` after the full frame has been written.
    """

    if len(payload) > max_payload_bytes:
        raise ProtocolError(
            f"payload is {len(payload)} bytes; limit is {max_payload_bytes}"
        )
    encoded_header = dict(header)
    encoded_header["version"] = PROTOCOL_VERSION
    encoded_header["payload_length"] = len(payload)
    try:
        header_bytes = json.dumps(
            encoded_header, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"header is not valid JSON: {exc}") from exc
    if len(header_bytes) > max_header_bytes:
        raise ProtocolError(
            f"header is {len(header_bytes)} bytes; limit is {max_header_bytes}"
        )
    connection.sendall(_LENGTH.pack(len(header_bytes)))
    connection.sendall(header_bytes)
    if payload:
        connection.sendall(payload)


def receive_frame(
    connection: socket.socket,
    *,
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> Frame:
    """Receive and validate one versioned protocol frame.

    Parameters:
        connection: Connected Unix stream socket.
        max_header_bytes: Maximum accepted encoded JSON header length.
        max_payload_bytes: Maximum accepted raw payload length.

    Returns:
        A :class:`Frame` containing a decoded header and payload bytes.
    """

    (header_length,) = _LENGTH.unpack(_recv_exact(connection, _LENGTH.size))
    if header_length == 0 or header_length > max_header_bytes:
        raise ProtocolError(
            f"invalid header length {header_length}; limit is {max_header_bytes}"
        )
    raw_header = _recv_exact(connection, header_length)
    try:
        decoded = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid JSON header: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ProtocolError("frame header must decode to a JSON object")
    if decoded.get("version") != PROTOCOL_VERSION:
        raise ProtocolError(
            f"unsupported protocol version {decoded.get('version')!r}; "
            f"expected {PROTOCOL_VERSION}"
        )
    payload_length = decoded.get("payload_length")
    if (
        not isinstance(payload_length, int)
        or isinstance(payload_length, bool)
        or payload_length < 0
        or payload_length > max_payload_bytes
    ):
        raise ProtocolError(
            f"invalid payload length {payload_length!r}; limit is {max_payload_bytes}"
        )
    return Frame(header=decoded, payload=_recv_exact(connection, payload_length))
