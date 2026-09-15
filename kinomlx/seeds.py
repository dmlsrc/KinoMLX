"""Ordered seed selection shared by every generation family."""

from typing import Literal

SeedMode = Literal["sequential", "station", "resident"]
SEED_MODES = ("sequential", "station", "resident")


def resolve_seeds(seed: int, seeds: tuple[int, ...] | None, count: int = 1) -> tuple[int, ...]:
    """Resolve explicit selections or consecutive variations without deduplication."""
    if type(count) is not int or count < 1:
        raise ValueError("images_per_prompt must be a positive integer")
    if seeds is not None:
        if not isinstance(seeds, (tuple, list)) or not seeds:
            raise ValueError("seeds must be a nonempty sequence of integers")
        if count != 1:
            raise ValueError("Explicit seeds cannot be combined with images_per_prompt > 1")
        values = tuple(seeds)
    else:
        if type(seed) is not int:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if not 0 <= seed < 2**64 or seed + count - 1 >= 2**64:
            raise ValueError("seeds must be unsigned 64-bit integers")
        values = tuple(seed + index for index in range(count))
    if any(type(value) is not int or not 0 <= value < 2**64 for value in values):
        raise ValueError("seeds must be unsigned 64-bit integers")
    return values
