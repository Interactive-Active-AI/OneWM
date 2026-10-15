"""Serve a trained OneWM policy to LIBERO simulation processes."""

from __future__ import annotations

import argparse
import logging
import threading
from collections.abc import Mapping
from typing import Any

import numpy as np

from deploy.libero.model_adapter import compose_policy_image
from deploy.libero.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    decode_message,
    encode_message,
)


LOGGER = logging.getLogger("onewm.libero.policy_server")
PRIMARY_KEY = "observation.images.image"
WRIST_KEY = "observation.images.wrist_image"


class LiberoPolicyEndpoint:
    """Hide LIBERO image preparation and OneWM inference behind one request method."""

    def __init__(self, policy: Any, *, replan_steps: int = 10) -> None:
        self.policy = policy
        self.config = policy.config
        self.replan_steps = max(
            1,
            min(int(replan_steps), int(policy.action_horizon)),
        )
        self._inference_lock = threading.Lock()

        framework = self.config.framework
        data = self.config.data
        self.action_dim = int(framework.action_dim)
        self.proprio_dim = int(framework.proprio_dim or 0)
        view_size = tuple(int(value) for value in data.video_size)
        if len(view_size) != 2 or min(view_size) <= 0:
            raise ValueError(f"invalid per-view image size: {view_size}")
        self.view_size: tuple[int, int] = (view_size[0], view_size[1])
        self.concat_mode = str(data.concat_multi_camera)
        self.video_keys = list(data.video_keys or [data.video_key])

        if self.action_dim != 7:
            raise ValueError(f"LIBERO requires action_dim=7, got {self.action_dim}")
        if self.proprio_dim != 8:
            raise ValueError(f"LIBERO requires proprio_dim=8, got {self.proprio_dim}")
        if self.concat_mode != "horizontal":
            raise ValueError(
                "LIBERO recipe must use horizontal camera concatenation, "
                f"got {self.concat_mode!r}"
            )
        if self.video_keys != [PRIMARY_KEY, WRIST_KEY]:
            raise ValueError(
                f"LIBERO camera order must be primary then wrist, got {self.video_keys}"
            )

    def metadata(self) -> dict[str, Any]:
        height, width = self.view_size
        return {
            "ok": True,
            "op": "metadata",
            "protocol_version": PROTOCOL_VERSION,
            "action_dim": self.action_dim,
            "proprio_dim": self.proprio_dim,
            "action_horizon": int(self.policy.action_horizon),
            "replan_steps": self.replan_steps,
            "camera_order": ["primary", "wrist"],
            "concat_mode": self.concat_mode,
            "view_size": [height, width],
            "policy_image_size": [height, width * 2],
        }

    def _predict(self, request: Mapping[str, Any]) -> np.ndarray:
        primary = np.asarray(request.get("primary"))
        wrist = np.asarray(request.get("wrist"))
        state = np.asarray(request.get("state"), dtype=np.float32).reshape(-1)
        instruction = str(request.get("instruction", "")).strip()

        if state.shape != (self.proprio_dim,) or not np.isfinite(state).all():
            raise ProtocolError(
                f"state must be finite shape ({self.proprio_dim},), got {state.shape}"
            )
        if not instruction:
            raise ProtocolError("instruction must be a non-empty string")

        image = compose_policy_image(
            primary,
            wrist,
            view_size=self.view_size,
            concat_mode=self.concat_mode,
        )
        with self._inference_lock:
            actions = self.policy.predict_chunk(image, state, instruction)
        actions = np.asarray(actions, dtype=np.float32)
        expected = (int(self.policy.action_horizon), self.action_dim)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise RuntimeError(
                f"policy returned invalid actions {actions.shape}; expected {expected}"
            )
        return actions

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        op = request.get("op")
        request_id = request.get("request_id")
        if op == "metadata":
            response = self.metadata()
        elif op == "predict":
            response = {
                "ok": True,
                "op": "predict",
                "actions": self._predict(request),
            }
        else:
            raise ProtocolError(f"unsupported operation: {op!r}")
        response["request_id"] = request_id
        return response


def _serve_connection(connection: Any, endpoint: LiberoPolicyEndpoint) -> None:
    for frame in connection:
        request_id = None
        try:
            request = decode_message(frame)
            request_id = request.get("request_id")
            response = endpoint.handle(request)
        except Exception as exc:
            LOGGER.exception("LIBERO policy request failed")
            response = {
                "ok": False,
                "request_id": request_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        connection.send(encode_message(response))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="LIBERO OneWM recipe")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Recipe overrides in dot.notation=value form",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from onewm.eval.policy import OneWMPolicy

    policy = OneWMPolicy(
        args.config,
        args.checkpoint,
        overrides=args.override,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        replan_steps=args.replan_steps,
        seed=args.seed,
    )
    endpoint = LiberoPolicyEndpoint(policy, replan_steps=args.replan_steps)

    from websockets.sync.server import serve

    LOGGER.info(
        "serving LIBERO policy on ws://%s:%d image=%s action_horizon=%d replan=%d",
        args.host,
        args.port,
        endpoint.metadata()["policy_image_size"],
        policy.action_horizon,
        endpoint.replan_steps,
    )
    with serve(
        lambda connection: _serve_connection(connection, endpoint),
        args.host,
        args.port,
        max_size=8 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=20,
    ) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
