"""Allocation-light native MLX implementation of the Krea 2 MMDiT graph."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.adapters.receipt import AdapterReceipt
from kinomlx.kernels import gelu_approx, silu
from kinomlx.modeling.layers import linear_shell

from ..metadata import Krea2TransformerConfig
from ..settings import REFERENCE_FP16_ERROR
from .inputs import compact_text_groups
from .reference import ReferenceAttention

type Krea2BoundaryObserver = Callable[[str, mx.array], None]

_FLOAT_DTYPES = frozenset((mx.float16, mx.bfloat16, mx.float32))
_SEQUENCE_MULTIPLE = 256


def _one_plus(value: mx.array) -> mx.array:
    return value + mx.array(1.0, dtype=value.dtype)


class Krea2RMSNorm(nn.Module):
    """Zero-centered RMSNorm with FP32 reduction and affine arithmetic."""

    def __init__(self, dims: int, eps: float) -> None:
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.weight = mx.zeros((0,), dtype=mx.float32)

    def __call__(self, value: mx.array) -> mx.array:
        if value.shape[-1] != self.dims:
            raise ValueError(f"Krea RMSNorm expects width {self.dims}, got {value.shape[-1]}")
        source_dtype = value.dtype
        widened = value.astype(mx.float32)
        variance = mx.mean(widened * widened, axis=-1, keepdims=True)
        normalized = widened * mx.rsqrt(variance + self.eps)
        scaled = normalized * (self.weight.astype(mx.float32) + 1.0)
        return scaled.astype(source_dtype)


class Krea2RotaryEmbedding:
    """Three-axis real RoPE table with adjacent-pair frequency repetition."""

    def __init__(self, *, theta: float, axes_dims: tuple[int, ...]) -> None:
        self.theta = theta
        self.axes_dims = axes_dims

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        if position_ids.ndim != 3 or position_ids.shape[-1] != len(self.axes_dims):
            raise ValueError("Krea position_ids must have shape [batch, sequence, position_axes]")
        positions = position_ids.astype(mx.float32)
        axis_angles: list[mx.array] = []
        for axis, dims in enumerate(self.axes_dims):
            frequencies = 1.0 / (self.theta ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
            angles = positions[..., axis, None] * frequencies
            axis_angles.append(mx.repeat(angles, 2, axis=-1))
        combined = mx.concatenate(axis_angles, axis=-1)
        return mx.cos(combined), mx.sin(combined)


def apply_rotary_embedding(
    queries: mx.array,
    keys: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> tuple[mx.array, mx.array]:
    """Apply adjacent-pair real RoPE with FP32 arithmetic and narrow cast-back."""
    if queries.shape[-1] != cos.shape[-1] or keys.shape[-1] != cos.shape[-1]:
        raise ValueError("Krea rotary width must match query and key head dimensions")
    expanded_cos = cos[:, None, :, :]
    expanded_sin = sin[:, None, :, :]

    def rotate(value: mx.array) -> mx.array:
        pairs = value.reshape(*value.shape[:-1], -1, 2)
        rotated = mx.stack([-pairs[..., 1], pairs[..., 0]], axis=-1)
        return rotated.reshape(value.shape)

    def apply(value: mx.array) -> mx.array:
        source_dtype = value.dtype
        widened = value.astype(mx.float32)
        rotated = rotate(value).astype(mx.float32)
        return (widened * expanded_cos + rotated * expanded_sin).astype(source_dtype)

    return apply(queries), apply(keys)


class Krea2Attention(nn.Module):
    """Native grouped-query attention with Krea Q/K norms and output gate."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        eps: float,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("Krea attention hidden size must be divisible by query heads")
        if num_heads % num_kv_heads:
            raise ValueError("Krea attention query heads must be divisible by KV heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.to_q = linear_shell(bias=False)
        self.to_k = linear_shell(bias=False)
        self.to_v = linear_shell(bias=False)
        self.to_gate = linear_shell(bias=False)
        self.to_out = linear_shell(bias=False)
        self.norm_q = Krea2RMSNorm(self.head_dim, eps)
        self.norm_k = Krea2RMSNorm(self.head_dim, eps)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        attention_mask: mx.array | None = None,
        rotary: tuple[mx.array, mx.array] | None = None,
        reference_attention: ReferenceAttention | None = None,
    ) -> mx.array:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Krea attention expects [batch, sequence, {self.hidden_size}] hidden states"
            )
        batch, length, _width = hidden_states.shape
        query = self.to_q(hidden_states).reshape(
            batch,
            length,
            self.num_heads,
            self.head_dim,
        )
        key = self.to_k(hidden_states).reshape(
            batch,
            length,
            self.num_kv_heads,
            self.head_dim,
        )
        value = self.to_v(hidden_states).reshape(
            batch,
            length,
            self.num_kv_heads,
            self.head_dim,
        )
        gate = self.to_gate(hidden_states)

        query = self.norm_q(query).transpose(0, 2, 1, 3)
        key = self.norm_k(key).transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        if rotary is not None:
            query, key = apply_rotary_embedding(query, key, *rotary)

        valid_queries: mx.array | None = None
        key_mask: mx.array | None = None
        if attention_mask is not None:
            if attention_mask.dtype != mx.bool_:
                raise ValueError("Krea attention mask must use a boolean dtype")
            if tuple(attention_mask.shape) != (batch, length):
                raise ValueError("Krea attention mask must match batch and sequence dimensions")
            valid_queries = attention_mask
            key_mask = attention_mask[:, None, None, :]

        if reference_attention is not None:
            if key_mask is not None:
                raise ValueError("reference attention requires a compact unmasked sequence")
            attended = reference_attention.attend(query, key, value, scale=self.scale)
        else:
            attended = mx.fast.scaled_dot_product_attention(
                query,
                key,
                value,
                scale=self.scale,
                mask=key_mask,
            )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, self.hidden_size)
        sigmoid_gate = mx.sigmoid(gate.astype(mx.float32)).astype(gate.dtype)
        attended = attended * sigmoid_gate
        if valid_queries is not None:
            # MLX assigns a finite minimum to false boolean-mask entries. A row
            # with no valid keys therefore softmaxes to a nonzero distribution,
            # while the official Torch path returns exact zeros. Key masking plus
            # this explicit query selection is equivalent to Krea's
            # square outer mask and avoids a quadratic mask allocation.
            # Selection also clears nonfinite results for fully masked queries.
            attended = mx.where(valid_queries[:, :, None], attended, 0)
        return self.to_out(attended)


class Krea2SwiGLU(nn.Module):
    """Krea feed-forward block with FP32-opmath SiLU."""

    def __init__(self, hidden_size: int, *, pretranspose_down: bool = False) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.gate = linear_shell(bias=False)
        self.up = linear_shell(bias=False)
        self.down = linear_shell(bias=False, pretransposed=pretranspose_down)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Krea SwiGLU expects width {self.hidden_size}, got {hidden_states.shape[-1]}"
            )
        return self.down(silu(self.gate(hidden_states)) * self.up(hidden_states))


class Krea2TextFusionBlock(nn.Module):
    """Unmodulated pre-norm transformer block used only by text fusion."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = Krea2RMSNorm(hidden_size, eps)
        self.norm2 = Krea2RMSNorm(hidden_size, eps)
        self.attn = Krea2Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            eps=eps,
        )
        self.ff = Krea2SwiGLU(hidden_size)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
        )
        return hidden_states + self.ff(self.norm2(hidden_states))


class Krea2TextFusion(nn.Module):
    """Collapse Qwen's tapped-layer axis, then refine the text-token axis."""

    def __init__(self, config: Krea2TransformerConfig) -> None:
        super().__init__()

        def block() -> Krea2TextFusionBlock:
            return Krea2TextFusionBlock(
                hidden_size=config.text_hidden_dim,
                num_heads=config.text_num_attention_heads,
                num_kv_heads=config.text_num_key_value_heads,
                eps=config.norm_eps,
            )

        self.layerwise_blocks = [block() for _index in range(config.num_layerwise_text_blocks)]
        self.projector = linear_shell(bias=False)
        self.refiner_blocks = [block() for _index in range(config.num_refiner_text_blocks)]

    def __call__(
        self,
        encoder_hidden_states: mx.array,
        *,
        attention_mask: mx.array | None,
    ) -> mx.array:
        batch, length, num_text_layers, dims = encoder_hidden_states.shape
        hidden_states = encoder_hidden_states.reshape(batch * length, num_text_layers, dims)
        for block in self.layerwise_blocks:
            hidden_states = block(hidden_states)
        hidden_states = hidden_states.reshape(batch, length, num_text_layers, dims)
        hidden_states = hidden_states.transpose(0, 1, 3, 2)
        hidden_states = self.projector(hidden_states).squeeze(-1)
        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)
        return hidden_states


def split_six_way_modulation(
    projected: mx.array,
    scale_shift_table: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Add one block table and return the official modulation chunk order."""
    if scale_shift_table.ndim != 2 or scale_shift_table.shape[0] != 6:
        raise ValueError("Krea block modulation table must have shape [6, hidden]")
    hidden_size = scale_shift_table.shape[1]
    if projected.shape[-1] != 6 * hidden_size:
        raise ValueError("Krea projected modulation width must be six times hidden size")
    modulation = projected.reshape(*projected.shape[:-1], 6, hidden_size)
    modulation = modulation + scale_shift_table
    return (
        modulation[..., 0, :],
        modulation[..., 1, :],
        modulation[..., 2, :],
        modulation[..., 3, :],
        modulation[..., 4, :],
        modulation[..., 5, :],
    )


class Krea2TransformerBlock(nn.Module):
    """One time-modulated Krea single-stream transformer block."""

    def __init__(self, config: Krea2TransformerConfig, *, pretranspose_down: bool = False) -> None:
        super().__init__()
        hidden_size = config.hidden_dim
        self.scale_shift_table = mx.zeros((0, 0), dtype=mx.float32)
        self.norm1 = Krea2RMSNorm(hidden_size, config.norm_eps)
        self.norm2 = Krea2RMSNorm(hidden_size, config.norm_eps)
        self.attn = Krea2Attention(
            hidden_size=hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            eps=config.norm_eps,
        )
        self.ff = Krea2SwiGLU(hidden_size, pretranspose_down=pretranspose_down)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        time_modulation: mx.array,
        rotary: tuple[mx.array, mx.array],
        attention_mask: mx.array | None,
        reference_attention: ReferenceAttention | None = None,
    ) -> mx.array:
        prescale, preshift, pregate, postscale, postshift, postgate = split_six_way_modulation(
            time_modulation, self.scale_shift_table
        )
        attention_input = _one_plus(prescale) * self.norm1(hidden_states) + preshift
        attention_output = self.attn(
            attention_input,
            attention_mask=attention_mask,
            rotary=rotary,
            reference_attention=reference_attention,
        )
        hidden_states = hidden_states + pregate * attention_output
        feed_forward_input = _one_plus(postscale) * self.norm2(hidden_states) + postshift
        return hidden_states + postgate * self.ff(feed_forward_input)


class Krea2TimestepEmbedding(nn.Module):
    """Cos-first sinusoidal flow-time embedding and two-layer MLP."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.linear_1 = linear_shell(bias=True)
        self.linear_2 = linear_shell(bias=True)

    def __call__(self, timestep: mx.array, *, dtype: mx.Dtype) -> mx.array:
        if timestep.ndim != 1:
            raise ValueError("Krea timestep must have shape [batch]")
        half = self.embed_dim // 2
        frequencies = mx.exp(-math.log(10_000.0) * mx.arange(half, dtype=mx.float32) / half)
        arguments = (timestep.astype(mx.float32) * 1000.0)[:, None, None] * frequencies
        embedding = mx.concatenate([mx.cos(arguments), mx.sin(arguments)], axis=-1)
        hidden_states = self.linear_1(embedding.astype(dtype))
        return self.linear_2(gelu_approx(hidden_states))


class Krea2TextProjection(nn.Module):
    """Project fused Qwen features into the MMDiT hidden width."""

    def __init__(self, config: Krea2TransformerConfig) -> None:
        super().__init__()
        self.norm = Krea2RMSNorm(config.text_hidden_dim, config.norm_eps)
        self.linear_1 = linear_shell(bias=True)
        self.linear_2 = linear_shell(bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.linear_1(self.norm(hidden_states))
        return self.linear_2(gelu_approx(hidden_states))


class Krea2FinalLayer(nn.Module):
    """Final time-adaptive RMSNorm and image-token projection."""

    def __init__(self, config: Krea2TransformerConfig) -> None:
        super().__init__()
        self.scale_shift_table = mx.zeros((0, 0), dtype=mx.float32)
        self.norm = Krea2RMSNorm(config.hidden_dim, config.norm_eps)
        self.linear = linear_shell(bias=True)

    def __call__(self, hidden_states: mx.array, time_embedding: mx.array) -> mx.array:
        if self.scale_shift_table.shape[0] != 2:
            raise ValueError("Krea final modulation table must have shape [2, hidden]")
        modulation = time_embedding + self.scale_shift_table
        scale = modulation[:, 0:1, :]
        shift = modulation[:, 1:2, :]
        hidden_states = _one_plus(scale) * self.norm(hidden_states) + shift
        return self.linear(hidden_states)


@dataclass(frozen=True)
class Krea2TransformerOutput:
    """Image predictions and sequence geometry (largest group for unequal lengths)."""

    sample: mx.array
    combined_sequence_length: int
    padding_tokens: int
    padded_sequence_length: int


@dataclass(frozen=True)
class Krea2PreparedConditioning:
    """Fixed products owned by one inference branch after adapter fusion."""

    hidden_states: mx.array
    position_ids: mx.array
    fused_text: mx.array
    text_projection: mx.array
    rotary: tuple[mx.array, mx.array]


class Krea2Transformer(nn.Module):
    """Krea Raw/Turbo shared MMDiT graph with explicit mask and position products."""

    def __init__(self, config: Krea2TransformerConfig, *, reference_denoise: bool = False) -> None:
        super().__init__()
        self._validate_config(config)
        self.config = config
        self.reference_denoise = reference_denoise
        self.adapter_receipts: tuple[AdapterReceipt, ...] = ()
        self.img_in = linear_shell(bias=True)
        self.time_embed = Krea2TimestepEmbedding(config.timestep_embed_dim)
        self.time_mod_proj = linear_shell(bias=True)
        self.text_fusion = Krea2TextFusion(config)
        self.txt_in = Krea2TextProjection(config)
        self.rotary_emb = Krea2RotaryEmbedding(
            theta=config.rope_theta,
            axes_dims=config.axes_dims_rope,
        )
        self.transformer_blocks = [
            Krea2TransformerBlock(config, pretranspose_down=not reference_denoise)
            for _index in range(config.num_layers)
        ]
        self.final_layer = Krea2FinalLayer(config)

    @staticmethod
    def _validate_config(config: Krea2TransformerConfig) -> None:
        dimensions = (
            config.attention_head_dim,
            config.in_channels,
            config.intermediate_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.num_layers,
            config.num_layerwise_text_blocks,
            config.num_refiner_text_blocks,
            config.num_text_layers,
            config.text_hidden_dim,
            config.text_intermediate_size,
            config.text_num_attention_heads,
            config.text_num_key_value_heads,
            config.timestep_embed_dim,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("Krea transformer dimensions must be positive")
        if len(config.axes_dims_rope) != 3:
            raise ValueError("Krea RoPE requires exactly three position axes")
        if sum(config.axes_dims_rope) != config.attention_head_dim:
            raise ValueError("Krea RoPE axis dimensions must sum to the attention head dimension")
        if any(dims <= 0 or dims % 2 for dims in config.axes_dims_rope):
            raise ValueError("Krea RoPE axis dimensions must be positive and even")
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("Krea query heads must be divisible by KV heads")
        if config.text_hidden_dim % config.text_num_attention_heads:
            raise ValueError("Krea text hidden dimension must be divisible by query heads")
        if config.text_num_attention_heads % config.text_num_key_value_heads:
            raise ValueError("Krea text query heads must be divisible by KV heads")
        if config.timestep_embed_dim % 2:
            raise ValueError("Krea timestep embedding dimension must be even")
        if not math.isfinite(config.norm_eps) or config.norm_eps <= 0.0:
            raise ValueError("Krea normalization epsilon must be finite and positive")
        if not math.isfinite(config.rope_theta) or config.rope_theta <= 0.0:
            raise ValueError("Krea RoPE theta must be finite and positive")

    def _validate_inputs(
        self,
        image_tokens: mx.array,
        encoder_hidden_states: mx.array,
        timestep: mx.array,
        position_ids: mx.array,
        attention_mask: mx.array | None,
    ) -> tuple[int, int, int]:
        config = self.config
        if image_tokens.ndim != 3 or image_tokens.shape[-1] != config.in_channels:
            raise ValueError(
                f"Krea image tokens must have shape [batch, sequence, {config.in_channels}]"
            )
        if encoder_hidden_states.ndim != 4 or tuple(encoder_hidden_states.shape[2:]) != (
            config.num_text_layers,
            config.text_hidden_dim,
        ):
            raise ValueError(
                "Krea encoder hidden states must have shape "
                "[batch, text, selected_layers, text_hidden]"
            )
        batch, image_length, _channels = image_tokens.shape
        text_batch, text_length, _layers, _text_dims = encoder_hidden_states.shape
        if text_batch != batch:
            raise ValueError("Krea image and text batches must match")
        if image_tokens.dtype not in _FLOAT_DTYPES:
            raise ValueError("Krea image tokens must use a supported floating dtype")
        if encoder_hidden_states.dtype != image_tokens.dtype:
            raise ValueError("Krea image and text activations must use the same dtype")
        if timestep.dtype not in _FLOAT_DTYPES or tuple(timestep.shape) != (batch,):
            raise ValueError("Krea timestep must be floating with shape [batch]")
        combined_length = text_length + image_length
        if tuple(position_ids.shape) != (batch, combined_length, 3):
            raise ValueError("Krea position_ids must match the combined sequence and three axes")
        if position_ids.dtype not in _FLOAT_DTYPES:
            raise ValueError("Krea position_ids must use a floating dtype")
        if attention_mask is None:
            return batch, text_length, image_length
        if attention_mask.dtype != mx.bool_:
            raise ValueError("Krea attention_mask must use a boolean dtype")
        if tuple(attention_mask.shape) != (batch, combined_length):
            raise ValueError("Krea attention_mask must match the combined sequence")
        if not mx.all(attention_mask[:, text_length:]).item():
            raise ValueError("Krea attention_mask must mark every image token as valid")
        return batch, text_length, image_length

    @staticmethod
    def _observe(
        observer: Krea2BoundaryObserver | None,
        name: str,
        value: mx.array,
    ) -> None:
        if observer is not None:
            observer(name, value)

    def __call__(
        self,
        image_tokens: mx.array,
        *,
        encoder_hidden_states: mx.array,
        timestep: mx.array,
        position_ids: mx.array,
        attention_mask: mx.array | None = None,
        observer: Krea2BoundaryObserver | None = None,
        reference_attention: ReferenceAttention | None = None,
    ) -> Krea2TransformerOutput:
        batch, text_length, image_length = self._validate_inputs(
            image_tokens,
            encoder_hidden_states,
            timestep,
            position_ids,
            attention_mask,
        )
        if reference_attention is not None:
            if attention_mask is not None:
                raise ValueError("reference attention requires compact input")
            if self.reference_denoise and image_tokens.dtype == mx.float16:
                raise ValueError(REFERENCE_FP16_ERROR)
            return self._forward(
                image_tokens,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                position_ids=position_ids,
                attention_mask=None,
                observer=observer,
                reference_attention=reference_attention,
            )
        if self.reference_denoise:
            if image_tokens.dtype == mx.float16:
                raise ValueError(REFERENCE_FP16_ERROR)
            if attention_mask is None:
                attention_mask = mx.ones((batch, text_length + image_length), dtype=mx.bool_)
            return self._forward(
                image_tokens,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                position_ids=position_ids,
                attention_mask=attention_mask,
                sequence_multiple=_SEQUENCE_MULTIPLE,
                observer=observer,
            )
        if attention_mask is None:
            return self._forward(
                image_tokens,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                position_ids=position_ids,
                attention_mask=None,
                observer=observer,
            )
        groups = compact_text_groups(encoder_hidden_states, position_ids, attention_mask)
        samples: dict[int, mx.array] = {}
        max_length = 0
        for index, group in enumerate(groups):
            rows = None if len(group.rows) == batch else mx.array(group.rows)
            group_observer = observer
            if observer is not None and len(groups) > 1:

                def record_group(
                    name: str,
                    value: mx.array,
                    group_index: int = index,
                    observe: Krea2BoundaryObserver = observer,
                ) -> None:
                    observe(f"group_{group_index:02d}.{name}", value)

                group_observer = record_group

            result = self._forward(
                image_tokens if rows is None else image_tokens[rows],
                encoder_hidden_states=group.hidden_states,
                timestep=timestep if rows is None else timestep[rows],
                position_ids=group.position_ids,
                attention_mask=None,
                observer=group_observer,
            )
            if len(groups) == 1:
                return result
            max_length = max(max_length, result.combined_sequence_length)
            for group_row, original_row in enumerate(group.rows):
                samples[original_row] = result.sample[group_row]
        return Krea2TransformerOutput(
            sample=mx.stack([samples[row] for row in range(batch)]),
            combined_sequence_length=max_length,
            padding_tokens=0,
            padded_sequence_length=max_length,
        )

    def prepare_conditioning(
        self, hidden_states: mx.array, position_ids: mx.array
    ) -> Krea2PreparedConditioning:
        """Build invariant unmasked products once, without a device synchronization."""
        fused = self.text_fusion(hidden_states, attention_mask=None)
        return Krea2PreparedConditioning(
            hidden_states, position_ids, fused, self.txt_in(fused), self.rotary_emb(position_ids)
        )

    def forward_prepared(
        self,
        image_tokens: mx.array,
        *,
        conditioning: Krea2PreparedConditioning,
        timestep: mx.array,
        observer: Krea2BoundaryObserver | None = None,
        reference_attention: ReferenceAttention | None = None,
    ) -> Krea2TransformerOutput:
        """Reuse branch products while evaluating all timestep-dependent blocks."""
        if self.reference_denoise:
            raise ValueError("prepared conditioning requires optimized denoising")
        self._validate_inputs(
            image_tokens, conditioning.hidden_states, timestep, conditioning.position_ids, None
        )
        return self._forward(
            image_tokens,
            encoder_hidden_states=conditioning.hidden_states,
            timestep=timestep,
            position_ids=conditioning.position_ids,
            attention_mask=None,
            observer=observer,
            reference_attention=reference_attention,
            prepared_conditioning=conditioning,
        )

    def _forward(
        self,
        image_tokens: mx.array,
        *,
        encoder_hidden_states: mx.array,
        timestep: mx.array,
        position_ids: mx.array,
        attention_mask: mx.array | None,
        sequence_multiple: int = 1,
        observer: Krea2BoundaryObserver | None = None,
        reference_attention: ReferenceAttention | None = None,
        prepared_conditioning: Krea2PreparedConditioning | None = None,
    ) -> Krea2TransformerOutput:
        """Evaluate one rectangular group; the public entry point selects its layout."""
        batch, image_length = image_tokens.shape[:2]
        text_length = encoder_hidden_states.shape[1]

        image_projection = self.img_in(image_tokens)
        self._observe(observer, "image_projection", image_projection)
        time_embedding = self.time_embed(timestep, dtype=image_tokens.dtype)
        self._observe(observer, "time_embed", time_embedding)
        time_modulation = self.time_mod_proj(gelu_approx(time_embedding))
        self._observe(observer, "time_mod", time_modulation)

        text_mask = None if attention_mask is None else attention_mask[:, :text_length]
        if prepared_conditioning is not None:
            fused_text = prepared_conditioning.fused_text
            text_projection = prepared_conditioning.text_projection
            self._observe(observer, "text_fusion", fused_text)
        elif text_length:
            fused_text = self.text_fusion(encoder_hidden_states, attention_mask=text_mask)
            self._observe(observer, "text_fusion", fused_text)
            text_projection = self.txt_in(fused_text)
        else:
            text_projection = mx.zeros((batch, 0, self.config.hidden_dim), dtype=image_tokens.dtype)
        self._observe(observer, "text_projection", text_projection)

        combined = mx.concatenate([text_projection, image_projection], axis=1)
        combined_length = combined.shape[1]
        padding_tokens = (-combined_length) % sequence_multiple
        if padding_tokens:
            combined = mx.concatenate(
                [
                    combined,
                    mx.zeros(
                        (batch, padding_tokens, self.config.hidden_dim),
                        dtype=combined.dtype,
                    ),
                ],
                axis=1,
            )
            if attention_mask is None:
                attention_mask = mx.ones((batch, combined_length), dtype=mx.bool_)
            attention_mask = mx.concatenate(
                [attention_mask, mx.zeros((batch, padding_tokens), dtype=mx.bool_)],
                axis=1,
            )
            position_ids = mx.concatenate(
                [position_ids, mx.zeros((batch, padding_tokens, 3), dtype=position_ids.dtype)],
                axis=1,
            )
        self._observe(observer, "combined_input", combined)

        rotary = (
            self.rotary_emb(position_ids)
            if prepared_conditioning is None
            else prepared_conditioning.rotary
        )
        self._observe(observer, "rotary_cos", rotary[0])
        self._observe(observer, "rotary_sin", rotary[1])

        # This loop constructs a lazy graph. The sampler owns evaluated step
        # progress; boundary observers remain available for targeted diagnostics.
        for index, block in enumerate(self.transformer_blocks):
            combined = block(
                combined,
                time_modulation=time_modulation,
                rotary=rotary,
                attention_mask=attention_mask,
                reference_attention=reference_attention,
            )
            self._observe(observer, f"block_{index:02d}", combined)

        final = self.final_layer(combined, time_embedding)
        output = final[:, text_length : text_length + image_length, :]
        self._observe(observer, "output", output)
        return Krea2TransformerOutput(
            sample=output,
            combined_sequence_length=combined_length,
            padding_tokens=padding_tokens,
            padded_sequence_length=combined.shape[1],
        )


__all__ = [
    "Krea2Attention",
    "Krea2BoundaryObserver",
    "Krea2RMSNorm",
    "Krea2RotaryEmbedding",
    "Krea2Transformer",
    "Krea2TransformerOutput",
    "apply_rotary_embedding",
    "split_six_way_modulation",
]
