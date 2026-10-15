"""Binary protocol shared by the OneWM policy server and LIBERO clients."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import msgpack
import numpy as np


PROTOCOL_VERSION = 1
_ARRAY_MARKER = "__onewm_ndarray__"


class ProtocolError(ValueError):
    """Raised when a policy frame violates the wire contract."""


def _encode_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            _ARRAY_MARKER: True,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": array.tobytes(),
        }
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported protocol value: {type(value).__name__}")


def _decode_value(value: dict[str, Any]) -> Any:
    if value.get(_ARRAY_MARKER) is not True:
        return value
    try:
        dtype = np.dtype(value["dtype"])
        shape = tuple(int(dim) for dim in value["shape"])
        data = value["data"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("malformed ndarray payload") from exc
    if any(dim < 0 for dim in shape) or not isinstance(data, bytes):
        raise ProtocolError("invalid ndarray shape or data")
    expected_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(data) != expected_size:
        raise ProtocolError(
            f"ndarray byte length mismatch: expected {expected_size}, got {len(data)}"
        )
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def encode_message(message: Mapping[str, Any]) -> bytes:
    """Encode one versioned message into a WebSocket binary frame."""

    payload = dict(message)
    payload.setdefault("protocol_version", PROTOCOL_VERSION)
    frame = msgpack.packb(payload, default=_encode_value, use_bin_type=True)
    if not isinstance(frame, bytes):
        raise TypeError("msgpack.packb did not return bytes")
    return frame


def decode_message(frame: bytes | bytearray | memoryview) -> dict[str, Any]:
    """Decode and validate one WebSocket binary frame."""

    if not isinstance(frame, (bytes, bytearray, memoryview)):
        raise ProtocolError("policy messages must use binary WebSocket frames")
    try:
        payload = msgpack.unpackb(
            bytes(frame),
            raw=False,
            object_hook=_decode_value,
            strict_map_key=False,
        )
    except ProtocolError:
        raise
    except Exception as exc:
        raise ProtocolError("invalid msgpack policy frame") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("policy frame root must be a map")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError(
            f"unsupported protocol_version={payload.get('protocol_version')!r}; "
            f"expected {PROTOCOL_VERSION}"
        )
    return payload
