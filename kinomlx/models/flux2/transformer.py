"""Common native Klein transformer with invocation-owned reference attention."""

import math
from functools import partial

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.adapters.receipt import AdapterReceipt
from kinomlx.kernels import rms_norm, silu
from kinomlx.modeling.layers import linear_shell

from .kv import ReferenceKV, attend, blend_modulation, reference_attention
from .velocity import PreparedVelocity


def rotary_positions(ids: mx.array) -> tuple[mx.array, mx.array]:
    """Four independent axes, with adjacent real/imaginary channel pairs."""
    frequency = 2000.0 ** (-mx.arange(0, 32, 2, dtype=mx.float32) / 32)
    angles = (ids.astype(mx.float32)[..., None] * frequency).reshape(*ids.shape[:-1], 64)
    angles = mx.repeat(angles, 2, axis=-1)
    return mx.cos(angles)[:, :, None], mx.sin(angles)[:, :, None]


def rotate(value: mx.array, positions: tuple[mx.array, mx.array]) -> mx.array:
    pairs = value.astype(mx.float32).reshape(*value.shape[:-1], -1, 2)
    perpendicular = mx.stack([-pairs[..., 1], pairs[..., 0]], axis=-1).reshape(value.shape)
    cosine, sine = positions
    return (value.astype(mx.float32) * cosine + perpendicular * sine).astype(value.dtype)


_compiled_rotate = mx.compile(rotate)


def joint_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    positions: tuple[mx.array, mx.array],
    *,
    text_tokens: int = 0,
    reference_kv: ReferenceKV | None = None,
    layer: int = 0,
) -> mx.array:
    query, key = _compiled_rotate(query, positions), _compiled_rotate(key, positions)
    result = (
        attend(query, key, value)
        if reference_kv is None
        else reference_attention(
            query, key, value, text_tokens=text_tokens, state=reference_kv, layer=layer
        )
    )
    return result.reshape(*query.shape[:2], -1)


def normalized(value: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    return mx.fast.layer_norm(value, None, None, 1e-6) * (1 + scale) + shift


_compiled_normalized = mx.compile(normalized)


def swiglu(value: mx.array) -> mx.array:
    gate, content = mx.split(value, 2, axis=-1)
    return silu(gate) * content


class QKNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = mx.zeros((0,))

    def __call__(self, value: mx.array) -> mx.array:
        # Diffusers functional RMSNorm retains the reduction and scale in FP32.
        return rms_norm(value, self.weight, 1e-6)


class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_in = linear_shell(bias=False)
        self.linear_out = linear_shell(bias=False)

    def __call__(self, value: mx.array) -> mx.array:
        return self.linear_out(swiglu(self.linear_in(value)))


class DoubleAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = linear_shell(bias=False)
        self.to_k = linear_shell(bias=False)
        self.to_v = linear_shell(bias=False)
        self.add_q_proj = linear_shell(bias=False)
        self.add_k_proj = linear_shell(bias=False)
        self.add_v_proj = linear_shell(bias=False)
        self.to_out = [linear_shell(bias=False)]
        self.to_add_out = linear_shell(bias=False)
        self.norm_q, self.norm_k = QKNorm(), QKNorm()
        self.norm_added_q, self.norm_added_k = QKNorm(), QKNorm()

    def __call__(
        self,
        image: mx.array,
        text: mx.array,
        positions: tuple[mx.array, mx.array],
        reference_kv: ReferenceKV | None = None,
        layer: int = 0,
    ) -> tuple[mx.array, mx.array]:
        def heads(value: mx.array) -> mx.array:
            return value.reshape(*value.shape[:2], 32, 128)

        query = mx.concatenate(
            [self.norm_added_q(heads(self.add_q_proj(text))), self.norm_q(heads(self.to_q(image)))],
            axis=1,
        )
        key = mx.concatenate(
            [self.norm_added_k(heads(self.add_k_proj(text))), self.norm_k(heads(self.to_k(image)))],
            axis=1,
        )
        value = mx.concatenate([heads(self.add_v_proj(text)), heads(self.to_v(image))], axis=1)
        result = joint_attention(
            query,
            key,
            value,
            positions,
            text_tokens=text.shape[1],
            reference_kv=reference_kv,
            layer=layer,
        )
        return self.to_out[0](result[:, text.shape[1] :]), self.to_add_out(
            result[:, : text.shape[1]]
        )


class DoubleBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = DoubleAttention()
        self.ff, self.ff_context = FeedForward(), FeedForward()

    def __call__(
        self,
        image: mx.array,
        text: mx.array,
        image_mod: mx.array,
        text_mod: mx.array,
        positions: tuple[mx.array, mx.array],
        reference_kv: ReferenceKV | None = None,
        layer: int = 0,
    ) -> tuple[mx.array, mx.array]:
        image_mod = image_mod[:, None] if image_mod.ndim == 2 else image_mod
        shift, scale, gate, ff_shift, ff_scale, ff_gate = mx.split(image_mod, 6, axis=-1)
        ts, tc, tg, fs, fc, fg = mx.split(text_mod[:, None], 6, axis=-1)
        ai, at = self.attn(
            _compiled_normalized(image, shift, scale),
            _compiled_normalized(text, ts, tc),
            positions,
            reference_kv,
            layer,
        )
        image, text = image + gate * ai, text + tg * at
        image = image + ff_gate * self.ff(_compiled_normalized(image, ff_shift, ff_scale))
        text = text + fg * self.ff_context(_compiled_normalized(text, fs, fc))
        if text.dtype == mx.float16:
            text = mx.clip(text, -65504, 65504)
        return image, text


class SingleAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_qkv_mlp_proj = linear_shell(bias=False)
        self.to_out = linear_shell(bias=False)
        self.norm_q, self.norm_k = QKNorm(), QKNorm()

    def __call__(
        self,
        value: mx.array,
        positions: tuple[mx.array, mx.array],
        reference_kv: ReferenceKV | None = None,
        layer: int = 0,
        text_tokens: int = 0,
    ) -> mx.array:
        query, key, values, mlp = mx.split(
            self.to_qkv_mlp_proj(value), [4096, 8192, 12288], axis=-1
        )
        shape = (*value.shape[:2], 32, 128)
        attention = joint_attention(
            self.norm_q(query.reshape(shape)),
            self.norm_k(key.reshape(shape)),
            values.reshape(shape),
            positions,
            text_tokens=text_tokens,
            reference_kv=reference_kv,
            layer=layer,
        )
        return self.to_out(mx.concatenate([attention, swiglu(mlp)], axis=-1))


class SingleBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = SingleAttention()

    def __call__(
        self,
        value: mx.array,
        modulation: mx.array,
        positions: tuple[mx.array, mx.array],
        reference_kv: ReferenceKV | None = None,
        layer: int = 0,
        text_tokens: int = 0,
    ) -> mx.array:
        modulation = modulation[:, None] if modulation.ndim == 2 else modulation
        shift, scale, gate = mx.split(modulation, 3, axis=-1)
        result = value + gate * self.attn(
            _compiled_normalized(value, shift, scale), positions, reference_kv, layer, text_tokens
        )
        return mx.clip(result, -65504, 65504) if result.dtype == mx.float16 else result


class Modulation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = linear_shell(bias=False)

    def __call__(self, value: mx.array) -> mx.array:
        return self.linear(silu(value))


class TimestepEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = linear_shell(bias=False)
        self.linear_2 = linear_shell(bias=False)

    def __call__(self, timestep: mx.array) -> mx.array:
        frequencies = mx.exp(-math.log(10000) * mx.arange(128, dtype=mx.float32) / 128)
        angles = timestep.astype(mx.float32)[:, None] * frequencies
        embedding = mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1).astype(timestep.dtype)
        return self.linear_2(silu(self.linear_1(embedding)))


class TimeEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.timestep_embedder = TimestepEmbedding()


class OutputNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = linear_shell(bias=False)

    def __call__(self, value: mx.array, embedding: mx.array) -> mx.array:
        scale, shift = mx.split(self.linear(silu(embedding)), 2, axis=-1)
        return normalized(value, shift[:, None], scale[:, None])


class Flux2Transformer(nn.Module):
    """The common 9B graph; base guidance belongs to the sampling recipe."""

    def __init__(self) -> None:
        super().__init__()
        self.adapter_receipts: tuple[AdapterReceipt, ...] = ()
        self.x_embedder = linear_shell(bias=False)
        self.context_embedder = linear_shell(bias=False)
        self.time_guidance_embed = TimeEmbedding()
        self.double_stream_modulation_img = Modulation()
        self.double_stream_modulation_txt = Modulation()
        self.single_stream_modulation = Modulation()
        self.transformer_blocks = [DoubleBlock() for _ in range(8)]
        self.single_transformer_blocks = [SingleBlock() for _ in range(24)]
        self.norm_out = OutputNorm()
        self.proj_out = linear_shell(bias=False)

    def new_reference_kv(self, reference_tokens: int) -> ReferenceKV:
        return ReferenceKV(
            reference_tokens, len(self.transformer_blocks) + len(self.single_transformer_blocks)
        )

    def prepare_velocities(
        self, text: mx.array, image_ids: tuple[mx.array, ...], text_ids: mx.array
    ) -> tuple[PreparedVelocity, ...]:
        """Prepare once inside denoise; returned calls own no cross-generation state."""
        projected = self.context_embedder(text)
        positions = tuple(
            rotary_positions(mx.concatenate([text_ids, ids], axis=1)) for ids in image_ids
        )
        mx.eval(projected, positions)
        return tuple(partial(self._forward, projected, position) for position in positions)

    def __call__(
        self,
        image: mx.array,
        text: mx.array,
        timestep: mx.array,
        image_ids: mx.array,
        text_ids: mx.array,
        *,
        reference_kv: ReferenceKV | None = None,
    ) -> mx.array:
        positions = rotary_positions(mx.concatenate([text_ids, image_ids], axis=1))
        return self._forward(self.context_embedder(text), positions, image, timestep, reference_kv)

    def _forward(
        self,
        text: mx.array,
        positions: tuple[mx.array, mx.array],
        image: mx.array,
        timestep: mx.array,
        reference_kv: ReferenceKV | None,
    ) -> mx.array:
        embedding = self.time_guidance_embed.timestep_embedder(timestep.astype(image.dtype) * 1000)
        im = self.double_stream_modulation_img(embedding)
        tm = self.double_stream_modulation_txt(embedding)
        sm = self.single_stream_modulation(embedding)
        reference_tokens = 0
        if reference_kv is not None:
            if reference_kv.closed:
                raise ValueError("Reference K/V belongs to a completed generation")
            if len(reference_kv.layers) != len(self.transformer_blocks) + len(
                self.single_transformer_blocks
            ):
                raise ValueError("Reference K/V layer count does not match the transformer")
            if reference_kv.extracting:
                reference_tokens = reference_kv.reference_tokens
                if reference_tokens >= image.shape[1]:
                    raise ValueError(
                        "KV extraction requires reference tokens followed by target tokens"
                    )
                fixed = self.time_guidance_embed.timestep_embedder(
                    mx.zeros_like(timestep).astype(image.dtype)
                )
                im = blend_modulation(
                    im,
                    self.double_stream_modulation_img(fixed),
                    0,
                    reference_tokens,
                    image.shape[1],
                )
                sm = blend_modulation(
                    sm,
                    self.single_stream_modulation(fixed),
                    text.shape[1],
                    reference_tokens,
                    text.shape[1] + image.shape[1],
                )
        image = self.x_embedder(image)
        for index, block in enumerate(self.transformer_blocks):
            image, text = block(image, text, im, tm, positions, reference_kv, index)
        value = mx.concatenate([text, image], axis=1)
        for index, single in enumerate(
            self.single_transformer_blocks, len(self.transformer_blocks)
        ):
            value = single(value, sm, positions, reference_kv, index, text.shape[1])
        return self.proj_out(self.norm_out(value[:, text.shape[1] + reference_tokens :], embedding))
