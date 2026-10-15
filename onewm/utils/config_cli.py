"""Small helpers for config command-line overrides."""

from __future__ import annotations

import ast
from typing import Any


def coerce_override_value(value: str) -> Any:
    value = value.strip()
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    if value.startswith("[") and value.endswith("]"):
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, list):
            raise ValueError(
                f"Override list value must parse to list, got: {type(parsed).__name__}"
            )
        return parsed
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def apply_overrides(cfg: Any, overrides: list[str]) -> Any:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be 'key=value', got: {override}")
        key, value = override.split("=", 1)
        parts = key.split(".")
        target = cfg
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise ValueError(f"Unknown override path: {key}")
            target = getattr(target, part)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise ValueError(f"Unknown override path: {key}")
        setattr(target, leaf, coerce_override_value(value))
    return cfg
