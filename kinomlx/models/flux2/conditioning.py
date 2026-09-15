"""Generation-owned velocity callables with invariant conditioning prepared."""

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import mlx.core as mx

from .kv import ReferenceKV

type PreparedVelocity = Callable[[mx.array, mx.array, ReferenceKV | None], mx.array]


@runtime_checkable
class PreparableVelocityModel(Protocol):
    def prepare_velocities(
        self, text: mx.array, image_ids: tuple[mx.array, ...], text_ids: mx.array
    ) -> tuple[PreparedVelocity, ...]: ...
