"""MLX attention backends used by the LTX-2 diffusion transformer."""

import weakref
from typing import Protocol, cast

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.kernels import mpp_linear
from kinomlx.kernels import rms_norm as precise_rms_norm
from kinomlx.models.ltx2.kernels import dense_attention
from kinomlx.models.ltx2.kernels.sol_attention import sol_attention

from ..sparse_attention import SparseAttentionStep
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


def _converted_weight(layer: nn.Linear) -> mpp_linear.Operand | None:
    """The prepared weight a converted layer holds in place of its BF16 weight, while it does."""
    kept = cast(
        "tuple[weakref.ref[mx.array], mx.array] | None", getattr(layer, "_mpp_weight", None)
    )
    if kept is None or "weight" not in layer:
        return None
    values = kept[0]()
    if values is None or layer["weight"] is not values:
        return None
    return mpp_linear.Operand(values, kept[1])


def _mpp_projection(
    layer: _Projection,
    value: mx.array,
    weight_t: mx.array | None,
    *,
    operand: mpp_linear.Operand | None = None,
    convert: bool = False,
) -> mx.array:
    """Apply a projection through MPP where it applies, otherwise as ``_projection``.

    ``operand`` is ``mpp_linear.fp16_operand(value)``, for callers that project one input
    several times. MPP is faster for large FP16 and BF16 products and gives MLX's bits within
    the limits ``mpp_linear`` states. Only a plain Linear qualifies: a subclass may add its own
    terms to the product, as scheduled adapters do.

    With ``convert`` a BF16 ``(out, in)`` weight is prepared on its first MPP call and the
    prepared FP16 copy replaces it, so memory stays flat and later calls skip the conversion.
    A converted weight that meets a call MPP does not take (fewer rows than MPP's floor, or
    MPP switched off) is restored to BF16 for MLX's product.
    """
    if type(layer) is not nn.Linear:
        return _projection(layer, value, weight_t)
    linear = layer
    bias = cast(mx.array, linear["bias"]) if "bias" in linear else None
    converted = _converted_weight(linear) if weight_t is None else None
    if converted is not None:
        if mpp_linear.applies(value, converted, bias, transposed=True):
            return mpp_linear.linear(value, converted, bias, transposed=True, operand=operand)
        restored = mpp_linear.restore(converted)
        return value @ restored.T if bias is None else mx.addmm(bias, value, restored.T)
    weight = weight_t
    if weight is None and "weight" in linear:
        weight = cast(mx.array, linear["weight"])
    transposed = weight_t is None
    if weight is None or not mpp_linear.applies(value, weight, bias, transposed=transposed):
        return _projection(layer, value, weight_t)
    if convert and transposed and weight.dtype == mx.bfloat16:
        prepared = mpp_linear.prepare(weight)
        linear.weight = prepared.values
        # Weakly: a weight bound again later (a fresh load) must not keep this copy alive.
        object.__setattr__(linear, "_mpp_weight", (weakref.ref(prepared.values), prepared.peak))
        return mpp_linear.linear(value, prepared, bias, transposed=True, operand=operand)
    return mpp_linear.linear(value, weight, bias, transposed=transposed, operand=operand)


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


def _attention_no_mask(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    heads: int,
    dim_head: int,
) -> mx.array:
    batch, query_tokens, _ = query.shape
    key_tokens = key.shape[1]
    query = query.reshape(batch, query_tokens, heads, dim_head).transpose(0, 2, 1, 3)
    key = key.reshape(batch, key_tokens, heads, dim_head).transpose(0, 2, 1, 3)
    value = value.reshape(batch, key_tokens, heads, dim_head).transpose(0, 2, 1, 3)
    output = mx.fast.scaled_dot_product_attention(query, key, value, scale=dim_head**-0.5)
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
) -> mx.array:
    batch, query_tokens, _ = query.shape
    key_tokens = key.shape[1]
    query = query.reshape(batch, query_tokens, heads, dim_head).transpose(0, 2, 1, 3)
    key = key.reshape(batch, key_tokens, heads, dim_head).transpose(0, 2, 1, 3)
    value = value.reshape(batch, key_tokens, heads, dim_head).transpose(0, 2, 1, 3)
    mask = _prepare_attention_mask(mask, query.dtype)
    output = mx.fast.scaled_dot_product_attention(
        query, key, value, scale=dim_head**-0.5, mask=mask
    )
    return output.transpose(0, 2, 1, 3).reshape(
        batch,
        query_tokens,
        heads * dim_head,
    )


_compiled_attention_no_mask = mx.compile(_attention_no_mask)
_compiled_attention_with_mask = mx.compile(_attention_with_mask)


def dense_kernel_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    heads: int,
) -> mx.array:
    """Run the FP16 dense kernel on flattened-head ``(batch, tokens, heads * dim)`` inputs.

    Keys and values may have their own token count, so cross-attention takes it too.
    """
    outputs = [
        dense_attention.dense_attention(query[item], key[item], value[item], heads=heads)
        for item in range(query.shape[0])
    ]
    return outputs[0][None] if len(outputs) == 1 else mx.stack(outputs)


def sparse_self_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    heads: int,
    sparse: SparseAttentionStep,
) -> mx.array:
    """Run Sol-Attn self-attention on flattened-head ``(batch, tokens, heads * dim)`` inputs."""
    outputs = [
        sol_attention(
            query[item], key[item], value[item], heads=heads, tau=sparse.tau, order=sparse.order
        )
        for item in range(query.shape[0])
    ]
    return outputs[0][None] if len(outputs) == 1 else mx.stack(outputs)


def scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    heads: int,
    dim_head: int,
    mask: mx.array | None = None,
    compile_attention: bool = True,
) -> mx.array:
    """Run MLX's scaled dot-product attention on flattened-head inputs.

    ``compile_attention`` compiles the head split, the call and the merge into one graph.
    """
    if mask is None:
        no_mask = _compiled_attention_no_mask if compile_attention else _attention_no_mask
        return no_mask(query, key, value, heads, dim_head)
    with_mask = _compiled_attention_with_mask if compile_attention else _attention_with_mask
    return with_mask(query, key, value, heads, dim_head, mask)


class Attention(nn.Module):
    """Multi-head attention with Q/K RMSNorm, split RoPE, and V2 head gates."""

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
        compile_attention: bool = True,
        dense_kernel: bool = False,
    ) -> None:
        super().__init__()
        if heads <= 0 or dim_head <= 0:
            raise ValueError("attention heads and head dimension must be positive")
        self.heads = heads
        self.dim_head = dim_head
        self.query_dim = query_dim
        self.context_dim = query_dim if context_dim is None else context_dim
        self.rope_type = rope_type
        self.compile_attention = compile_attention
        self.dense_kernel = dense_kernel
        # Set by the loader for a resident transformer; see ``_mpp_projection``.
        self.convert_mpp_weights = False
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
        sparse: SparseAttentionStep | None = None,
    ) -> mx.array:
        if sparse is not None and (context is not None or mask is not None):
            raise ValueError("sparse attention applies only to unmasked self-attention")
        if value.shape[-1] != self.query_dim:
            raise ValueError(
                f"attention query expects width {self.query_dim}, got {value.shape[-1]}"
            )
        gate = None
        if self.to_gate_logits is not None:
            gate = 2.0 * mx.sigmoid(
                _mpp_projection(
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
        # One FP16 copy of each input serves all of its MPP projections; it is never computed
        # when none applies.
        value_operand = mpp_linear.fp16_operand(value)
        source_operand = value_operand if context is None else mpp_linear.fp16_operand(source)
        convert = self.convert_mpp_weights
        projected_value = _mpp_projection(
            self.to_v, source, self._to_v_weight_t, operand=source_operand, convert=convert
        )
        query = self.q_norm(
            _mpp_projection(
                self.to_q, value, self._to_q_weight_t, operand=value_operand, convert=convert
            )
        )
        key = self.k_norm(
            _mpp_projection(
                self.to_k, source, self._to_k_weight_t, operand=source_operand, convert=convert
            )
        )
        if pe is not None:
            query = apply_rotary_emb(query, pe, self.rope_type)
            key = apply_rotary_emb(key, pe if k_pe is None else k_pe, self.rope_type)

        if sparse is not None:
            output = sparse_self_attention(
                query, key, projected_value, heads=self.heads, sparse=sparse
            )
        elif (
            self.dense_kernel
            and mask is None
            and dense_attention.applies(query, key, heads=self.heads)
        ):
            output = dense_kernel_attention(query, key, projected_value, heads=self.heads)
        else:
            output = scaled_dot_product_attention(
                query,
                key,
                projected_value,
                heads=self.heads,
                dim_head=self.dim_head,
                mask=mask,
                compile_attention=self.compile_attention,
            )
        if gate is not None:
            batch, tokens, _ = output.shape
            output = output.reshape(batch, tokens, self.heads, self.dim_head)
            output = output * gate[..., None]
            output = output.reshape(batch, tokens, self.heads * self.dim_head)
        return _mpp_projection(
            self.to_out, output, self._to_out_weight_t, convert=self.convert_mpp_weights
        )


__all__ = [
    "Attention",
    "RMSNorm",
    "dense_kernel_attention",
    "rms_norm",
    "scaled_dot_product_attention",
    "sparse_self_attention",
]
