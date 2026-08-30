"""MLX attention backends used by the LTX-2 diffusion transformer."""

from __future__ import annotations

from typing import Protocol, cast

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.kernels import rms_norm as precise_rms_norm
from kinomlx.kernels.steel_attention import maybe_steel_attention

from .rope import LTXRopeType, apply_rotary_emb


class _Projection(Protocol):
    def __call__(self, value: mx.array, /) -> mx.array: ...

    def __contains__(self, key: object) -> bool: ...

    def __getitem__(self, key: str) -> object: ...


def _linear_shell(*, bias: bool) -> nn.Linear:
    """Create an update-ready zero-sized Linear shell."""
    layer = nn.Linear.__new__(nn.Linear)
    nn.Module.__init__(layer)
    layer.weight = mx.zeros((0, 0), dtype=mx.float32)
    if bias:
        layer.bias = mx.zeros((0,), dtype=mx.float32)
    return layer


def _norm_weight_shell() -> mx.array:
    """Create an update-ready zero-sized RMSNorm scale shell."""
    return mx.zeros((0,), dtype=mx.float32)


def _projection(
    layer: _Projection,
    value: mx.array,
    weight_t: mx.array | None,
) -> mx.array:
    """Apply a normal/quantized Linear or its baked transpose."""
    if weight_t is None:
        return layer(value)
    if "bias" not in layer:
        return value @ weight_t
    return mx.addmm(cast(mx.array, layer["bias"]), value, weight_t)


def rms_norm(
    value: mx.array,
    weight: mx.array | None = None,
    eps: float = 1e-6,
) -> mx.array:
    """Apply central RMSNorm precision policy without widening stored output."""
    return precise_rms_norm(value, weight, eps)


class RMSNorm(nn.Module):
    """RMS normalization with a checkpoint-backed scale vector."""

    def __init__(self, dims: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.weight = _norm_weight_shell()

    def __call__(self, value: mx.array) -> mx.array:
        if value.shape[-1] != self.dims:
            raise ValueError(f"RMSNorm expects width {self.dims}, got {value.shape[-1]}")
        return rms_norm(value, self.weight, self.eps)


def _prepare_attention_mask(mask: mx.array, dtype: mx.Dtype) -> mx.array:
    while mask.ndim in (2, 3):
        mask = mx.expand_dims(mask, axis=0 if mask.ndim == 2 else 1)
    if mask.dtype != mx.bool_:
        mask = mask.astype(dtype)
    return mask


def _attention_backend(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    scale: float,
    mask: mx.array | None,
    use_steel_attention: bool,
    steel_attention_d64: bool,
    steel_attention_probe: bool,
    attention_compiled: bool,
) -> mx.array:
    if use_steel_attention:
        output = maybe_steel_attention(
            query,
            key,
            value,
            scale=scale,
            mask=mask,
            enable_d64=steel_attention_d64,
            probe=steel_attention_probe,
            probe_compiled=attention_compiled,
            inputs_last_axis_contiguous=True,
        )
        if output is not None:
            return output
    return mx.fast.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=scale,
        mask=mask,
    )


def _attention_no_mask(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    heads: int,
    dim_head: int,
    use_steel_attention: bool,
    steel_attention_d64: bool,
    steel_attention_probe: bool,
    attention_compiled: bool,
) -> mx.array:
    batch, query_tokens, _ = query.shape
    key_tokens = key.shape[1]
    query = query.reshape(batch, query_tokens, heads, dim_head).transpose(0, 2, 1, 3)
    key = key.reshape(batch, key_tokens, heads, dim_head).transpose(0, 2, 1, 3)
    value = value.reshape(batch, key_tokens, heads, dim_head).transpose(0, 2, 1, 3)
    output = _attention_backend(
        query,
        key,
        value,
        scale=dim_head**-0.5,
        mask=None,
        use_steel_attention=use_steel_attention,
        steel_attention_d64=steel_attention_d64,
        steel_attention_probe=steel_attention_probe,
        attention_compiled=attention_compiled,
    )
    return output.transpose(0, 2, 1, 3).reshape(
        batch,
        query_tokens,
        heads * dim_head,
    )


def _attention_with_mask(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    heads: int,
    dim_head: int,
    mask: mx.array,
    use_steel_attention: bool,
    steel_attention_d64: bool,
    steel_attention_probe: bool,
    attention_compiled: bool,
) -> mx.array:
    batch, query_tokens, _ = query.shape
    key_tokens = key.shape[1]
    query = query.reshape(batch, query_tokens, heads, dim_head).transpose(0, 2, 1, 3)
    key = key.reshape(batch, key_tokens, heads, dim_head).transpose(0, 2, 1, 3)
    value = value.reshape(batch, key_tokens, heads, dim_head).transpose(0, 2, 1, 3)
    mask = _prepare_attention_mask(mask, query.dtype)
    output = _attention_backend(
        query,
        key,
        value,
        scale=dim_head**-0.5,
        mask=mask,
        use_steel_attention=use_steel_attention,
        steel_attention_d64=steel_attention_d64,
        steel_attention_probe=steel_attention_probe,
        attention_compiled=attention_compiled,
    )
    return output.transpose(0, 2, 1, 3).reshape(
        batch,
        query_tokens,
        heads * dim_head,
    )


_compiled_attention_no_mask = mx.compile(_attention_no_mask)
_compiled_attention_with_mask = mx.compile(_attention_with_mask)


def scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    heads: int,
    dim_head: int,
    mask: mx.array | None = None,
    use_steel_attention: bool = True,
    compile_attention: bool = True,
    steel_attention_d64: bool = True,
    steel_attention_probe: bool = False,
) -> mx.array:
    """Run the selected MLX SDPA backend on flattened-head inputs.

    ``compile_attention`` controls graph compilation independently of STEEL.
    Set both it and ``use_steel_attention`` false to reproduce the original
    uncompiled stock-SDPA path during a backend bisection.
    """
    no_mask = _compiled_attention_no_mask if compile_attention else _attention_no_mask
    with_mask = _compiled_attention_with_mask if compile_attention else _attention_with_mask
    if mask is None:
        return no_mask(
            query,
            key,
            value,
            heads,
            dim_head,
            use_steel_attention,
            steel_attention_d64,
            steel_attention_probe,
            compile_attention,
        )
    return with_mask(
        query,
        key,
        value,
        heads,
        dim_head,
        mask,
        use_steel_attention,
        steel_attention_d64,
        steel_attention_probe,
        compile_attention,
    )


class Attention(nn.Module):
    """Multi-head attention with Q/K RMSNorm, split RoPE, and V2 head gates."""

    _LAYOUT_TARGETS = {
        "to_q": "_to_q_weight_t",
        "to_k": "_to_k_weight_t",
        "to_v": "_to_v_weight_t",
        "to_out": "_to_out_weight_t",
        "to_gate_logits": "_to_gate_logits_weight_t",
    }

    def __init__(
        self,
        query_dim: int,
        *,
        heads: int = 8,
        dim_head: int = 64,
        context_dim: int | None = None,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        norm_eps: float = 1e-6,
        apply_gated_attention: bool = False,
        use_steel_attention: bool = True,
        compile_attention: bool = True,
        steel_attention_d64: bool = True,
        steel_attention_probe: bool = False,
    ) -> None:
        super().__init__()
        if heads <= 0 or dim_head <= 0:
            raise ValueError("attention heads and head dimension must be positive")
        self.heads = heads
        self.dim_head = dim_head
        self.query_dim = query_dim
        self.context_dim = query_dim if context_dim is None else context_dim
        self.rope_type = rope_type
        self.use_steel_attention = use_steel_attention
        self.compile_attention = compile_attention
        self.steel_attention_d64 = steel_attention_d64
        self.steel_attention_probe = steel_attention_probe
        inner_dim = heads * dim_head
        self.q_norm = RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = RMSNorm(inner_dim, eps=norm_eps)
        self.to_q = _linear_shell(bias=True)
        self.to_k = _linear_shell(bias=True)
        self.to_v = _linear_shell(bias=True)
        self.to_out = _linear_shell(bias=True)
        self.to_gate_logits = _linear_shell(bias=True) if apply_gated_attention else None
        self._to_q_weight_t: mx.array | None = None
        self._to_k_weight_t: mx.array | None = None
        self._to_v_weight_t: mx.array | None = None
        self._to_out_weight_t: mx.array | None = None
        self._to_gate_logits_weight_t: mx.array | None = None

    def __call__(
        self,
        value: mx.array,
        *,
        context: mx.array | None = None,
        mask: mx.array | None = None,
        pe: tuple[mx.array, mx.array] | None = None,
        k_pe: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        if value.shape[-1] != self.query_dim:
            raise ValueError(
                f"attention query expects width {self.query_dim}, got {value.shape[-1]}"
            )
        gate = None
        if self.to_gate_logits is not None:
            gate = 2.0 * mx.sigmoid(
                _projection(
                    self.to_gate_logits,
                    value,
                    self._to_gate_logits_weight_t,
                )
            )

        source = value if context is None else context
        if source.shape[-1] != self.context_dim:
            raise ValueError(
                f"attention context expects width {self.context_dim}, got {source.shape[-1]}"
            )
        projected_value = _projection(self.to_v, source, self._to_v_weight_t)
        query = self.q_norm(_projection(self.to_q, value, self._to_q_weight_t))
        key = self.k_norm(_projection(self.to_k, source, self._to_k_weight_t))
        if pe is not None:
            query = apply_rotary_emb(query, pe, self.rope_type)
            key = apply_rotary_emb(key, pe if k_pe is None else k_pe, self.rope_type)

        output = scaled_dot_product_attention(
            query,
            key,
            projected_value,
            heads=self.heads,
            dim_head=self.dim_head,
            mask=mask,
            use_steel_attention=self.use_steel_attention,
            compile_attention=self.compile_attention,
            steel_attention_d64=self.steel_attention_d64,
            steel_attention_probe=self.steel_attention_probe,
        )
        if gate is not None:
            batch, tokens, _ = output.shape
            output = output.reshape(batch, tokens, self.heads, self.dim_head)
            output = output * gate[..., None]
            output = output.reshape(batch, tokens, self.heads * self.dim_head)
        return _projection(self.to_out, output, self._to_out_weight_t)

    def _pretranspose(self, target: str) -> list[mx.array]:
        if target not in self._LAYOUT_TARGETS:
            raise ValueError(f"unsupported attention layout target: {target}")
        layer = getattr(self, target)
        if layer is None:
            return []
        cache_attribute = self._LAYOUT_TARGETS[target]
        cached = getattr(self, cache_attribute)
        if cached is None:
            if "weight" not in layer:
                raise ValueError(f"{target} weight is unavailable for pretranspose")
            cached = mx.contiguous(layer.weight.T)
            setattr(self, cache_attribute, cached)
        arrays = [cached]
        bias = layer.get("bias")
        if bias is not None:
            arrays.append(bias)
        return arrays

    def apply_layouts(
        self,
        specs: tuple[tuple[str, str], ...],
    ) -> list[mx.array]:
        """Materialize supported same-math attention layouts."""
        arrays: list[mx.array] = []
        for target, layout in specs:
            if layout != "pretranspose":
                raise ValueError(f"unsupported attention layout: {target}:{layout}")
            arrays.extend(self._pretranspose(target))
        return arrays

    def drop_layout_sources(self, specs: tuple[tuple[str, str], ...]) -> None:
        """Release dense source weights replaced by baked layouts."""
        for target, layout in specs:
            if layout != "pretranspose" or target not in self._LAYOUT_TARGETS:
                raise ValueError(f"unsupported attention layout: {target}:{layout}")
            layer = getattr(self, target)
            if (
                layer is not None
                and getattr(self, self._LAYOUT_TARGETS[target]) is not None
                and "weight" in layer
            ):
                del layer.weight


__all__ = ["Attention", "RMSNorm", "rms_norm", "scaled_dot_product_attention"]
