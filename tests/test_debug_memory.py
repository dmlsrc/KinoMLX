"""Low-overhead MLX allocator-counter diagnostics."""

from __future__ import annotations

import pytest

from kinomlx.debug.memory import _create_interval_sampler, capture_mlx_memory_counters


def test_memory_counter_capture_does_not_synchronize_or_touch_arrays() -> None:
    reset_calls = 0

    class _Counters:
        def get_active_memory(self) -> int:
            return 11

        def get_cache_memory(self) -> int:
            return 22

        def get_peak_memory(self) -> int:
            return 33

        def reset_peak_memory(self) -> None:
            nonlocal reset_calls
            reset_calls += 1

        def synchronize(self) -> None:
            pytest.fail("allocator counter capture must not synchronize")

    assert capture_mlx_memory_counters(_Counters(), reset_peak=True) == {
        "active_bytes": 11,
        "cache_bytes": 22,
        "peak_bytes": 33,
    }
    assert reset_calls == 1


def test_interval_sampler_resets_at_creation_and_after_each_sample() -> None:
    reset_calls = 0

    class _Counters:
        def get_active_memory(self) -> int:
            return 11

        def get_cache_memory(self) -> int:
            return 22

        def get_peak_memory(self) -> int:
            return 33

        def reset_peak_memory(self) -> None:
            nonlocal reset_calls
            reset_calls += 1

    sample = _create_interval_sampler(_Counters())
    assert reset_calls == 1
    assert sample()["peak_bytes"] == 33
    assert reset_calls == 2
    assert sample()["peak_bytes"] == 33
    assert reset_calls == 3
