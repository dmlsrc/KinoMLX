"""Low-overhead allocator counters for durable run diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class _MLXMemoryCounters(Protocol):
    def get_active_memory(self) -> int: ...

    def get_cache_memory(self) -> int: ...

    def get_peak_memory(self) -> int: ...

    def reset_peak_memory(self) -> None: ...


def capture_mlx_memory_counters(
    mx: _MLXMemoryCounters,
    *,
    reset_peak: bool = False,
) -> dict[str, int]:
    """Read MLX allocator counters without evaluating arrays or synchronizing.

    When ``reset_peak`` is true, the returned peak covers the interval ending at
    this sample. The reset happens after all counters have been read.
    """
    counters = {
        "active_bytes": max(0, int(mx.get_active_memory())),
        "cache_bytes": max(0, int(mx.get_cache_memory())),
        "peak_bytes": max(0, int(mx.get_peak_memory())),
    }
    if reset_peak:
        mx.reset_peak_memory()
    return counters


def create_mlx_memory_sampler() -> Callable[[], dict[str, int]]:
    """Bind MLX once and return a non-synchronizing interval-peak sampler."""
    import mlx.core as mx

    return _create_interval_sampler(mx)


def _create_interval_sampler(mx: _MLXMemoryCounters) -> Callable[[], dict[str, int]]:
    """Create an interval sampler against an MLX-compatible counter surface."""
    # Establish the baseline when diagnostics begin so the first checkpoint is
    # not polluted by allocator activity that predates the run reporter.
    mx.reset_peak_memory()
    return lambda: capture_mlx_memory_counters(mx, reset_peak=True)


__all__ = ["capture_mlx_memory_counters", "create_mlx_memory_sampler"]
