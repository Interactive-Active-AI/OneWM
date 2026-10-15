"""Run the GPU-resident OneWM policy independently of the MuJoCo environment."""

from __future__ import annotations

import argparse
import asyncio
import json

from deploy.robocasa365.protocol import decode_observation
from onewm.eval.policy import OneWMPolicy
from onewm.eval.robocasa365 import RoboCasa365HistoryAdapter


class Service:
    def __init__(self, args: argparse.Namespace) -> None:
        self.policy = OneWMPolicy(
            args.config,
            args.checkpoint,
            device=args.device,
            num_inference_steps=10,
            replan_steps=8,
            seed=args.seed,
            overrides=args.override,
        )
        self.adapters: dict[str, RoboCasa365HistoryAdapter] = {}

    def handle(self, request: dict) -> dict:
        rollout_id = str(request.get("rollout_id", "default"))
        operation = request.get("op")
        if operation == "health":
            return {"ok": True, "contract": "rc365_xiaomi14_action12_v1"}
        if operation == "reset":
            self.adapters[rollout_id] = RoboCasa365HistoryAdapter()
            return {"ok": True}
        if operation not in {"observe", "predict"}:
            raise ValueError(f"unsupported operation: {operation!r}")
        adapter = self.adapters.setdefault(rollout_id, RoboCasa365HistoryAdapter())
        adapter.observe(decode_observation(request["observation"]))
        if operation == "observe":
            return {"ok": True}
        image, state, instruction = adapter.encode()
        chunk = self.policy.predict_chunk(image, state, instruction)[:8]
        return {"ok": True, "actions": chunk.tolist()}


async def main_async(args: argparse.Namespace) -> None:
    import websockets

    service = Service(args)

    async def handler(socket):
        async for message in socket:
            try:
                response = service.handle(json.loads(message))
            except Exception as error:  # noqa: BLE001
                response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            await socket.send(json.dumps(response))

    async with websockets.serve(
        handler, args.host, args.port, max_size=8 * 1024 * 1024
    ):
        print(
            f"RoboCasa365 OneWM policy server listening on ws://{args.host}:{args.port}",
            flush=True,
        )
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
