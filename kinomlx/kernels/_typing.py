"""Internal callable contracts missing from MLX's dynamic kernel API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import mlx.core as mx


class MetalKernel(Protocol):
    """Typed surface returned by ``mx.fast.metal_kernel``."""

    def __call__(
        self,
        *,
        inputs: Sequence[object],
        output_shapes: Sequence[tuple[int, ...]],
        output_dtypes: Sequence[mx.Dtype],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        template: Sequence[tuple[str, bool | int | mx.Dtype]] | None = None,
        init_value: float | None = None,
        verbose: bool = False,
        stream: mx.StreamOrDevice = None,
    ) -> Sequence[mx.array]: ...


__all__ = ["MetalKernel"]
