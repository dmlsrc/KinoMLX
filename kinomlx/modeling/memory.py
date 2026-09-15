"""Shared decode headroom from Metal's limit and reclaimable macOS memory."""

import ctypes
import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import mlx.core as mx

_log = logging.getLogger(__name__)


class _VMStatistics64(ctypes.Structure):
    """The public HOST_VM_INFO64 revision-1 layout from mach/vm_statistics.h."""

    _fields_ = [
        ("free_count", ctypes.c_uint32),
        ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32),
        ("wire_count", ctypes.c_uint32),
        ("page_events", ctypes.c_uint64 * 9),
        ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32),
        ("compression_events", ctypes.c_uint64 * 4),
        ("compressor_page_count", ctypes.c_uint32),
        ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32),
        ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    ]


@lru_cache(maxsize=1)
def _mach_library() -> ctypes.CDLL:
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    library.mach_host_self.argtypes = []
    library.mach_host_self.restype = ctypes.c_uint32
    library.host_statistics64.argtypes = [
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    library.host_statistics64.restype = ctypes.c_int
    library.mach_port_deallocate.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    library.mach_port_deallocate.restype = ctypes.c_int
    return library


def _reclaimable_bytes(stats: _VMStatistics64, page_size: int) -> int:
    # Speculative pages are included in BOTH free_count and external_page_count.
    # Do not add inactive_count: it overlaps these categories and includes
    # anonymous application pages that require compression or swap to reclaim.
    free = max(0, int(stats.free_count) - int(stats.speculative_count))
    return (free + int(stats.external_page_count) + int(stats.purgeable_count)) * page_size


def system_reclaimable_memory() -> int | None:
    """Estimate free plus reclaimable memory without counting swap capacity.

    Pageable file-backed pages may need writeback or be faulted in again. This
    is a current capacity estimate, not a reservation or an allocation guarantee.
    """
    try:
        library = _mach_library()
        host = library.mach_host_self()
        task = ctypes.c_uint32.in_dll(library, "mach_task_self_").value
        try:
            stats = _VMStatistics64()
            count = ctypes.c_uint32(ctypes.sizeof(stats) // ctypes.sizeof(ctypes.c_int32))
            result = library.host_statistics64(host, 4, ctypes.byref(stats), ctypes.byref(count))
            if result != 0 or count.value < 38:
                raise OSError(f"host_statistics64 failed: {result}, count={count.value}")
            return _reclaimable_bytes(stats, os.sysconf("SC_PAGE_SIZE"))
        finally:
            library.mach_port_deallocate(task, host)
    except AttributeError, OSError, ValueError:
        _log.warning("macOS reclaimable memory is unavailable; using the Metal budget alone")
        return None


@dataclass(frozen=True)
class DecodeMemoryBudget:
    """Additional allocation headroom, not total free RAM or a new MLX limit."""

    limit_bytes: int
    active_bytes: int
    cache_bytes: int
    total_bytes: int | None = None
    recommended_working_set_bytes: int | None = None
    system_reclaimable_bytes: int | None = None
    max_buffer_bytes: int | None = None

    @property
    def available_bytes(self) -> int:
        remaining = max(0, self.limit_bytes - self.active_bytes)
        if self.system_reclaimable_bytes is not None:
            remaining = min(remaining, max(0, self.system_reclaimable_bytes))
        return remaining

    @property
    def available_gb(self) -> float:
        return self.available_bytes / 1_000_000_000

    def to_dict(self) -> dict[str, int | None]:
        return {
            "limit_bytes": self.limit_bytes,
            "active_bytes": self.active_bytes,
            "cache_bytes": self.cache_bytes,
            "system_reclaimable_bytes": self.system_reclaimable_bytes,
            "available_bytes": self.available_bytes,
            "max_buffer_bytes": self.max_buffer_bytes,
        }


def _positive_bytes(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def capture_decode_memory_budget() -> DecodeMemoryBudget:
    """Release unused MLX buffers and sample the shared VAE planning budget.

    Metal's recommended working set reflects iogpu.wired_limit_mb. The system
    estimate already excludes live allocations, so it is a separate ceiling:
    live MLX bytes must not be subtracted from it a second time.
    """
    info = mx.device_info()
    total = _positive_bytes(info.get("memory_size"))
    recommended = _positive_bytes(info.get("max_recommended_working_set_size"))
    limits = [value for value in (total, recommended) if value is not None]
    if not limits:
        raise RuntimeError("MLX did not report a GPU memory limit for automatic VAE tiling")
    mx.clear_cache()
    budget = DecodeMemoryBudget(
        limit_bytes=min(limits),
        active_bytes=int(mx.get_active_memory()),
        cache_bytes=int(mx.get_cache_memory()),
        total_bytes=total,
        recommended_working_set_bytes=recommended,
        system_reclaimable_bytes=system_reclaimable_memory(),
        max_buffer_bytes=_positive_bytes(info.get("max_buffer_length")),
    )
    system = budget.system_reclaimable_bytes
    limiter = (
        "system-limited"
        if system is not None and system < max(0, budget.limit_bytes - budget.active_bytes)
        else "GPU-limited"
    )
    _log.info(
        "VAE memory: decode headroom=%.2f GiB (%s), GPU limit=%.2f GiB, live MLX=%.2f GiB",
        budget.available_bytes / 1024**3,
        limiter,
        budget.limit_bytes / 1024**3,
        budget.active_bytes / 1024**3,
    )
    return budget


__all__ = [
    "DecodeMemoryBudget",
    "capture_decode_memory_budget",
    "system_reclaimable_memory",
]
