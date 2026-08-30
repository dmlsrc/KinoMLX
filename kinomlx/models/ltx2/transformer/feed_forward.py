"""LTX-2 GELU feed-forward network with cache-baked layouts."""

from __future__ import annotations

from typing import cast

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.kernels import gelu_approx

from .attention import _linear_shell, _projection


class GELUApprox(nn.Module):
    """Checkpoint-compatible input projection followed by approximate GELU."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.proj = _linear_shell(bias=bias)

    def __call__(self, value: mx.array) -> mx.array:
        if value.shape[-1] != self.dim_in:
            raise ValueError(f"GELU projection expects width {self.dim_in}, got {value.shape[-1]}")
        projected = self.proj(value)
        if projected.shape[-1] != self.dim_out:
            raise ValueError(
                f"GELU projection produced width {projected.shape[-1]}, expected {self.dim_out}"
            )
        return gelu_approx(projected)


class FeedForward(nn.Module):
    """Linear-GELU-linear MLP used by each audio and video block."""

    def __init__(
        self,
        dim: int,
        *,
        dim_out: int | None = None,
        mult: int = 4,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if mult <= 0:
            raise ValueError("feed-forward multiplier must be positive")
        self.dim = dim
        self.output_dim = dim if dim_out is None else dim_out
        inner_dim = dim * mult
        self.project_in = GELUApprox(dim, inner_dim, bias=bias)
        self.project_out = _linear_shell(bias=bias)
        self._project_in_weight_t: mx.array | None = None
        self._project_out_weight_t: mx.array | None = None

    def _target_dtype(self) -> mx.Dtype | None:
        if self._project_in_weight_t is not None:
            return self._project_in_weight_t.dtype
        if self._project_out_weight_t is not None:
            return self._project_out_weight_t.dtype
        return None

    def _project_in(self, value: mx.array) -> mx.array:
        projected = _projection(
            self.project_in.proj,
            value,
            self._project_in_weight_t,
        )
        return gelu_approx(projected)

    def _project_out(self, value: mx.array) -> mx.array:
        return _projection(self.project_out, value, self._project_out_weight_t)

    def __call__(self, value: mx.array) -> mx.array:
        if value.shape[-1] != self.dim:
            raise ValueError(f"feed-forward expects width {self.dim}, got {value.shape[-1]}")
        target_dtype = self._target_dtype()
        if target_dtype is None or target_dtype == value.dtype:
            output = self._project_out(self._project_in(value))
            if output.shape[-1] != self.output_dim:
                raise ValueError(
                    f"feed-forward produced width {output.shape[-1]}, expected {self.output_dim}"
                )
            return output
        residual_dtype = value.dtype
        output = self._project_out(self._project_in(value.astype(target_dtype)))
        if output.shape[-1] != self.output_dim:
            raise ValueError(
                f"feed-forward produced width {output.shape[-1]}, expected {self.output_dim}"
            )
        return output.astype(residual_dtype)

    def _pretranspose(self, target: str) -> list[mx.array]:
        if target == "project_in":
            layer = self.project_in.proj
            attribute = "_project_in_weight_t"
        elif target == "project_out":
            layer = self.project_out
            attribute = "_project_out_weight_t"
        else:
            raise ValueError(f"unsupported feed-forward layout target: {target}")
        cached = getattr(self, attribute)
        if cached is None:
            if "weight" not in layer:
                raise ValueError(f"{target} weight is unavailable for pretranspose")
            cached = mx.contiguous(layer.weight.T)
            setattr(self, attribute, cached)
        arrays = [cached]
        bias = cast(mx.array | None, layer.get("bias"))
        if bias is not None:
            arrays.append(bias)
        return arrays

    def apply_layouts(
        self,
        specs: tuple[tuple[str, str], ...],
    ) -> list[mx.array]:
        """Materialize selected same-math FF layouts."""
        arrays: list[mx.array] = []
        for target, layout in specs:
            if layout != "pretranspose":
                raise ValueError(f"unsupported feed-forward layout: {target}:{layout}")
            arrays.extend(self._pretranspose(target))
        return arrays

    def drop_layout_sources(self, specs: tuple[tuple[str, str], ...]) -> None:
        """Release dense source weights replaced by baked layouts."""
        for target, layout in specs:
            if target == "project_in":
                layer = self.project_in.proj
                cached = self._project_in_weight_t
            elif target == "project_out":
                layer = self.project_out
                cached = self._project_out_weight_t
            else:
                raise ValueError(f"unsupported feed-forward layout: {target}:{layout}")
            if layout != "pretranspose":
                raise ValueError(f"unsupported feed-forward layout: {target}:{layout}")
            if cached is not None and "weight" in layer:
                del layer.weight


__all__ = ["FeedForward", "GELUApprox"]
