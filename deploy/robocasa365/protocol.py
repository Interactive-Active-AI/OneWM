"""Small JSON/zlib protocol shared by the simulator and policy environments."""

from __future__ import annotations

import base64
import zlib
from typing import Any

import numpy as np


CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_eye_in_hand",
    "video.robot0_agentview_right",
)
STATE_KEYS = (
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
)


def encode_array(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "data": base64.b64encode(zlib.compress(array.tobytes(), level=1)).decode(
            "ascii"
        ),
    }


def decode_array(payload: dict[str, Any]) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(payload["data"]))
    array = np.frombuffer(raw, dtype=np.dtype(payload["dtype"]))
    expected = int(np.prod(payload["shape"], dtype=np.int64))
    if array.size != expected:
        raise ValueError(f"array payload size mismatch: {array.size} != {expected}")
    return array.reshape(payload["shape"]).copy()


def encode_observation(observation: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        key: encode_array(observation[key]) for key in CAMERA_KEYS + STATE_KEYS
    }
    output["annotation.human.task_description"] = str(
        observation.get("annotation.human.task_description", "")
    )
    return output


def decode_observation(payload: dict[str, Any]) -> dict[str, Any]:
    instruction_key = "annotation.human.task_description"
    output: dict[str, Any] = {
        key: decode_array(value)
        for key, value in payload.items()
        if key != instruction_key
    }
    output[instruction_key] = str(payload.get(instruction_key, ""))
    return output
