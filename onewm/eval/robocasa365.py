"""RoboCasa365 online observation history and action adapter."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch

from onewm.data.robocasa365 import (
    RoboCasa365VideoTransform,
    compose_robocasa365_views,
    robocasa365_action_to_env_dict,
    robocasa365_state_to_xiaomi14,
)


ONLINE_CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_eye_in_hand",
    "video.robot0_agentview_right",
)
ONLINE_STATE_KEYS = (
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
)


def _chw_uint8(value: Any) -> torch.Tensor:
    array = np.asarray(value)
    if array.shape != (256, 256, 3):
        raise ValueError(f"RoboCasa365 RGB must be [256,256,3], got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)


def online_raw_state(observation: dict[str, Any]) -> torch.Tensor:
    missing = [key for key in ONLINE_STATE_KEYS if key not in observation]
    if missing:
        raise KeyError(f"RoboCasa365 observation misses state keys: {missing}")
    parts = [
        torch.as_tensor(np.asarray(observation[key]), dtype=torch.float32).flatten()
        for key in ONLINE_STATE_KEYS
    ]
    raw = torch.cat(parts)
    if raw.shape != (16,):
        raise ValueError(
            f"online RoboCasa365 raw state must be 16D, got {tuple(raw.shape)}"
        )
    return raw


class RoboCasa365HistoryAdapter:
    """Maintain Xiaomi offsets ``[-6,-4,-2,0]`` for one rollout."""

    def __init__(
        self, *, separator_width: int = 16, separator_value: int = 128
    ) -> None:
        self.separator_width = int(separator_width)
        self.separator_value = int(separator_value)
        self._history: deque[dict[str, Any]] = deque(maxlen=7)
        self._center_transform = RoboCasa365VideoTransform(
            random_probability=0.0, crop_ratio=0.95
        )

    def reset(self, observation: dict[str, Any] | None = None) -> None:
        self._history.clear()
        if observation is not None:
            self.observe(observation)

    def observe(self, observation: dict[str, Any]) -> None:
        missing = [key for key in ONLINE_CAMERA_KEYS if key not in observation]
        if missing:
            raise KeyError(f"RoboCasa365 observation misses camera keys: {missing}")
        # Copy arrays because the simulator can reuse its observation buffers.
        stored = {
            key: np.array(observation[key], copy=True)
            for key in ONLINE_CAMERA_KEYS + ONLINE_STATE_KEYS
        }
        stored["annotation.human.task_description"] = str(
            observation.get("annotation.human.task_description", "")
        )
        self._history.append(stored)

    def _selected(self) -> list[dict[str, Any]]:
        if not self._history:
            raise RuntimeError(
                "observe() must be called before encoding a policy input"
            )
        rows = list(self._history)
        padded = [rows[0]] * (7 - len(rows)) + rows
        return [padded[index] for index in (0, 2, 4, 6)]

    def encode(self) -> tuple[torch.Tensor, np.ndarray, str]:
        selected = self._selected()
        cameras = [
            torch.stack([_chw_uint8(row[key]) for row in selected], dim=0)
            for key in ONLINE_CAMERA_KEYS
        ]
        cameras = self._center_transform(cameras)
        strips = compose_robocasa365_views(
            cameras[0],
            cameras[1],
            cameras[2],
            separator_width=self.separator_width,
            separator_value=self.separator_value,
        )
        # Duplicate current to keep Wan's clean prefix exactly 5 raw frames.
        strips = torch.cat((strips, strips[-1:]), dim=0)
        video = strips.float().div(255.0).mul(2.0).sub(1.0)
        video = video.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        state = torch.stack(
            [robocasa365_state_to_xiaomi14(online_raw_state(row)) for row in selected]
        ).numpy()
        instruction = selected[-1].get("annotation.human.task_description", "").strip()
        if not instruction:
            raise ValueError("RoboCasa365 observation has an empty task instruction")
        return video, state.astype(np.float32), instruction

    @staticmethod
    def action_dict(action: np.ndarray | torch.Tensor) -> dict[str, np.ndarray]:
        value = torch.as_tensor(action, dtype=torch.float32).clamp(-1.0, 1.0)
        output = robocasa365_action_to_env_dict(value)
        return {
            key: tensor.cpu().numpy().astype(np.float32)
            for key, tensor in output.items()
        }
