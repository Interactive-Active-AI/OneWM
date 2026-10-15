"""Atomic records used by Franka data preparation."""

import json
import os
from pathlib import Path
import time

ASSETS = Path("data/franka/assets")


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default


def status(path, **values):
    atomic(path, {"time": time.time(), **values})
