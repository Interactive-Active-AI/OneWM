"""Portable output locations for OneWM training."""

from pathlib import Path
import os
from typing import Any


def resolve_training_output_dir(requested=None, *, checkpoint_root=None) -> Path:
    path = Path(requested or "outputs").expanduser()
    root = checkpoint_root or os.environ.get("ONEWM_CHECKPOINT_ROOT")
    if root and not path.is_absolute():
        if path.parts and path.parts[0] == "outputs":
            path = Path(*path.parts[1:])
        path = Path(root).expanduser() / path
    return path.resolve()


def enforce_training_output_dir(config: Any, *, checkpoint_root=None) -> Path:
    path = resolve_training_output_dir(
        config.training.output_dir, checkpoint_root=checkpoint_root
    )
    config.training.output_dir = str(path)
    return path
