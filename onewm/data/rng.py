"""Random-number isolation helpers for dataset preprocessing."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def isolated_cpu_rng(seed: int) -> Iterator[None]:
    """Temporarily seed the default CPU RNG without touching device RNGs."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(generator.get_state())
        yield
