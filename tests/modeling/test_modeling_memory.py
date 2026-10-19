"""Shared Metal budget and macOS reclaimable-page accounting."""

import ctypes

import pytest

from kinomlx.modeling import memory

GIB = 1024**3


def test_reclaimable_pages_do_not_double_count_speculative_or_inactive() -> None:
    stats = memory._VMStatistics64()
    stats.free_count = 100
    stats.speculative_count = 20
    stats.external_page_count = 70
    stats.purgeable_count = 30
    stats.inactive_count = 500
    stats.internal_page_count = 400
    stats.compressor_page_count = 300

    assert memory._reclaimable_bytes(stats, 16384) == (80 + 70 + 30) * 16384


def test_native_host_vm_statistics_layout_and_query() -> None:
    assert ctypes.sizeof(memory._VMStatistics64) == 38 * 4
    available = memory.system_reclaimable_memory()
    assert available is not None
    assert available > 0


@pytest.mark.parametrize(
    ("active", "reclaimable", "expected"),
    [(20, 30, 30), (20, 50, 40), (61, 30, 0), (0, 0, 0), (5, None, 55)],
)
def test_budget_takes_separate_ceilings_without_double_subtraction(active, reclaimable, expected):
    budget = memory.DecodeMemoryBudget(
        limit_bytes=60 * GIB,
        active_bytes=active * GIB,
        cache_bytes=0,
        system_reclaimable_bytes=None if reclaimable is None else reclaimable * GIB,
    )
    assert budget.available_bytes == expected * GIB


@pytest.mark.parametrize(
    ("system_gib", "expected_gib", "limiter"),
    [(59, 58, "GPU-limited"), (40, 40, "system-limited")],
)
def test_capture_reclaims_own_cache_and_uses_full_reported_wired_limit(
    monkeypatch, caplog, system_gib, expected_gib, limiter
) -> None:
    caplog.set_level("INFO", logger=memory.__name__)
    counters = {"cache": 6 * GIB}
    monkeypatch.setattr(
        memory.mx,
        "device_info",
        lambda: {
            "memory_size": 64 * GIB,
            "max_recommended_working_set_size": 60 * GIB,
            "max_buffer_length": 39 * GIB,
        },
    )
    monkeypatch.setattr(memory.mx, "clear_cache", lambda: counters.update(cache=0))
    monkeypatch.setattr(memory.mx, "get_active_memory", lambda: 2 * GIB)
    monkeypatch.setattr(memory.mx, "get_cache_memory", lambda: counters["cache"])
    monkeypatch.setattr(memory, "system_reclaimable_memory", lambda: system_gib * GIB)

    budget = memory.capture_decode_memory_budget()

    assert budget.available_bytes == expected_gib * GIB
    assert budget.limit_bytes == 60 * GIB
    assert budget.cache_bytes == 0
    assert budget.max_buffer_bytes == 39 * GIB
    assert caplog.messages == [
        f"VAE memory: decode headroom={expected_gib:.2f} GiB ({limiter}), "
        "GPU limit=60.00 GiB, live MLX=2.00 GiB"
    ]


def test_missing_system_sample_does_not_invent_a_small_fixed_budget(monkeypatch) -> None:
    monkeypatch.setattr(memory.mx, "device_info", lambda: {"memory_size": 64 * GIB})
    monkeypatch.setattr(memory.mx, "clear_cache", lambda: None)
    monkeypatch.setattr(memory.mx, "get_active_memory", lambda: 3 * GIB)
    monkeypatch.setattr(memory.mx, "get_cache_memory", lambda: 0)
    monkeypatch.setattr(memory, "system_reclaimable_memory", lambda: None)

    assert memory.capture_decode_memory_budget().available_bytes == 61 * GIB


def test_missing_all_device_limits_is_reported_instead_of_guessed(monkeypatch) -> None:
    monkeypatch.setattr(memory.mx, "device_info", dict)
    with pytest.raises(RuntimeError, match="did not report a GPU memory limit"):
        memory.capture_decode_memory_budget()
