"""Launch OneWM with exact global batch size on a chosen number of GPUs."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from onewm.config import load_config
from onewm.tools.global_batch import gradient_accumulation_steps
from onewm.utils.config_cli import apply_overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument(
        "--accelerate-config", default="configs/accelerate/deepspeed_zero2.yaml"
    )
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.override)
    expected = config.training.expected_global_batch_size
    if expected is None:
        parser.error("training.expected_global_batch_size is required")
    accumulation = gradient_accumulation_steps(
        expected, args.num_processes, config.training.batch_size
    )
    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        args.accelerate_config,
        "--num_processes",
        str(args.num_processes),
        "--module",
        "onewm.training.train",
        "--config",
        str(Path(args.config)),
        "--override",
        *args.override,
        f"training.gradient_accumulation_steps={accumulation}",
    ]
    print(
        f"OneWM global batch = {config.training.batch_size} × {args.num_processes} × {accumulation} = {expected}",
        flush=True,
    )
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
