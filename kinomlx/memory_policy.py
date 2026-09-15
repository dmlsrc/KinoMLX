"""Invocation-owned MLX allocator and wired-memory policy."""

import logging
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Protocol, cast

import mlx.core as mx

from kinomlx.settings import Settings

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MLXMemoryPolicy:
    """Process-wide limits selected for one invocation, expressed in GiB.

    A None cache limit preserves the host's allocator setting. A None wired
    limit selects the current device recommendation; zero disables wiring.
    """

    cache_limit_gb: float | None = 1.0
    wired_limit_gb: float | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> MLXMemoryPolicy:
        return cls(settings.mlx_cache_limit_gb, settings.mlx_wired_limit_gb)


class MemoryPolicyOwner(Protocol):
    @property
    def memory_policy(self) -> MLXMemoryPolicy: ...


_active_policy: ContextVar[MLXMemoryPolicy | None] = ContextVar("mlx_memory_policy", default=None)


def apply_memory_policy(policy: MLXMemoryPolicy | Settings) -> None:
    """Apply both limits before standalone model work, while MLX is idle.

    This changes process-wide state and does not restore previous limits.
    Call once before low-level component use; runners and public recipes
    establish their own invocation scope.
    """
    selected = MLXMemoryPolicy.from_settings(policy) if isinstance(policy, Settings) else policy
    active = _active_policy.get()
    if active is not None:
        raise RuntimeError("cannot change MLX memory policy during an active invocation")
    _apply(selected)


def _apply(policy: MLXMemoryPolicy) -> None:
    for name, value in (("cache", policy.cache_limit_gb), ("wired", policy.wired_limit_gb)):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"MLX {name} limit must be finite and non-negative")
    cache = None if policy.cache_limit_gb is None else int(policy.cache_limit_gb * 1024**3)
    wired = None
    if mx.metal.is_available():
        recommended = mx.device_info().get("max_recommended_working_set_size")
        if isinstance(recommended, int) and not isinstance(recommended, bool) and recommended > 0:
            wired = (
                recommended
                if policy.wired_limit_gb is None
                else int(policy.wired_limit_gb * 1024**3)
            )
            if wired > recommended:
                raise ValueError(
                    f"MLX wired limit exceeds the system wired limit of "
                    f"{recommended / 1024**3:.1f} GiB (iogpu.wired_limit_mb)"
                )
        elif policy.wired_limit_gb is not None:
            raise ValueError("MLX device does not report a recommended wired limit")
    # Validate the entire policy before changing either control.
    if cache is not None:
        mx.set_cache_limit(cache)
    if wired is not None:
        mx.set_wired_limit(wired)
        _log.info(
            "MLX memory: allocator cache=%s, wired=%.1f GiB",
            "host" if cache is None else f"{cache / 1024**3:g} GiB",
            wired / 1024**3,
        )


@contextmanager
def memory_policy_scope(policy: MLXMemoryPolicy) -> Iterator[None]:
    """Apply a policy once across nested runner/recipe calls.

    Sequential invocations always reapply their selected policy. Nested calls
    must agree; changing global limits mid-generation is an error. Hosts must
    serialize MLX invocations and consume lazy outputs before starting another.
    """
    active = _active_policy.get()
    if active is not None:
        if active != policy:
            raise ValueError("nested MLX invocations require the same memory policy")
        yield
        return
    _apply(policy)
    token = _active_policy.set(policy)
    try:
        yield
    finally:
        _active_policy.reset(token)


def memory_managed[**P, ResultT](
    operation: Callable[P, ResultT],
) -> Callable[P, ResultT]:
    """Manage a public recipe whose second parameter is named resources."""

    @wraps(operation)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> ResultT:
        resources = cast(
            MemoryPolicyOwner, kwargs["resources"] if "resources" in kwargs else args[1]
        )
        with memory_policy_scope(resources.memory_policy):
            return operation(*args, **kwargs)

    return wrapped
