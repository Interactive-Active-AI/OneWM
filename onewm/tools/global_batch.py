"""Exact global-batch geometry for benchmark launchers."""

from __future__ import annotations

import argparse


def gradient_accumulation_steps(
    global_batch: int, world_size: int, micro_batch: int
) -> int:
    values = (int(global_batch), int(world_size), int(micro_batch))
    if any(value <= 0 for value in values):
        raise ValueError("global_batch, world_size, and micro_batch must be positive")
    denominator = values[1] * values[2]
    if values[0] % denominator:
        raise ValueError(
            f"global batch {values[0]} is not divisible by world_size*micro_batch "
            f"({values[1]}*{values[2]}={denominator})"
        )
    return values[0] // denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-batch", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--micro-batch", type=int, required=True)
    args = parser.parse_args()
    print(
        gradient_accumulation_steps(
            args.global_batch, args.world_size, args.micro_batch
        )
    )


if __name__ == "__main__":
    main()
