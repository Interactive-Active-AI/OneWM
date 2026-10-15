"""Pure NumPy translation between LIBERO observations and policy values."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


LIBERO_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)
LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
LIBERO_MAX_STEPS = {
    "libero_spatial": 400,
    "libero_object": 400,
    "libero_goal": 400,
    "libero_10": 700,
}


def rotate_rgb_180(image: np.ndarray) -> np.ndarray:
    """Rotate a simulator RGB image to the orientation used by training data."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB image, got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array[::-1, ::-1])


def extract_camera_views(observation: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return training-oriented primary and wrist views in that order."""

    return (
        rotate_rgb_180(observation["agentview_image"]),
        rotate_rgb_180(observation["robot0_eye_in_hand_image"]),
    )


def quat_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    """Convert an xyzw quaternion into a three-dimensional axis-angle vector."""

    quat = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if quat.shape != (4,):
        raise ValueError(f"expected quaternion shape (4,), got {quat.shape}")
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(1.0 - float(quat[3] * quat[3]), 0.0))
    if math.isclose(denominator, 0.0, abs_tol=1e-8):
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.acos(float(quat[3]))
    return (quat[:3] * angle / denominator).astype(np.float32)


def extract_state(observation: dict) -> np.ndarray:
    """Build LIBERO's 8-D EEF pose and gripper proprio vector."""

    state = np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32).reshape(3),
            quat_to_axis_angle(observation["robot0_eef_quat"]),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32).reshape(2),
        )
    )
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError(f"invalid LIBERO state: shape={state.shape}")
    return state.astype(np.float32, copy=False)


def action_to_libero(action: np.ndarray) -> np.ndarray:
    """Convert dataset gripper convention into LIBERO's signed command."""

    raw = np.asarray(action, dtype=np.float32).reshape(-1)
    if raw.shape != (7,) or not np.isfinite(raw).all():
        raise ValueError(f"policy action must be finite shape (7,), got {raw.shape}")
    converted = raw.copy()
    converted[6] = 1.0 - 2.0 * float(raw[6] > 0.5)
    return converted


def max_steps_for_suite(suite: str) -> int:
    try:
        return LIBERO_MAX_STEPS[suite]
    except KeyError as exc:
        raise ValueError(f"unsupported LIBERO suite: {suite!r}") from exc


def select_shard(
    items: Sequence,
    *,
    shard_index: int,
    num_shards: int,
) -> list:
    """Select one deterministic modulo shard without copying benchmark logic."""

    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    return [
        item for index, item in enumerate(items) if index % num_shards == shard_index
    ]
