"""Input preprocessing for LTX-2 transformer modalities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import mlx.core as mx

from .rope import precompute_freqs_cis
from .timestep import AdaLayerNormSingle
from .transformer import TransformerArgs
from .wrappers import Modality


def _minimum_for_dtype(dtype: mx.Dtype) -> float:
    if dtype == mx.float16:
        return -65504.0
    return -3.3895313892515355e38


def _tiny_for_dtype(dtype: mx.Dtype) -> float:
    if dtype == mx.float16:
        return 6.103515625e-5
    return 1.1754943508222875e-38


class ModalityPreprocessor:
    """Prepare one modality and its optional A/V cross-attention inputs."""

    def __init__(
        self,
        *,
        patchify_proj: Callable[[mx.array], mx.array],
        adaln: AdaLayerNormSingle,
        prompt_adaln: AdaLayerNormSingle | None,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        inner_dim: int,
        max_pos: tuple[int, ...],
        heads: int,
        cross_dim: int,
        cross_max_pos: int,
        timestep_scale: float,
        av_gate_scale: float,
        theta: float,
        compute_dtype: mx.Dtype,
        double_precision_rope: bool,
        keyframes_abs_pos_embedding: Callable[[], mx.array] | None = None,
    ) -> None:
        self.patchify_proj = patchify_proj
        self.adaln = adaln
        self.prompt_adaln = prompt_adaln
        self.cross_scale_shift_adaln = cross_scale_shift_adaln
        self.cross_gate_adaln = cross_gate_adaln
        self.inner_dim = inner_dim
        self.max_pos = max_pos
        self.heads = heads
        self.cross_dim = cross_dim
        self.cross_max_pos = cross_max_pos
        self.timestep_scale = timestep_scale
        self.av_gate_scale = av_gate_scale
        self.theta = theta
        self.compute_dtype = compute_dtype
        self.double_precision_rope = double_precision_rope
        self.keyframes_abs_pos_embedding = keyframes_abs_pos_embedding

    def _timestep(
        self,
        value: mx.array,
        adaln: AdaLayerNormSingle,
        batch: int,
    ) -> tuple[mx.array, mx.array]:
        modulation, embedded = adaln(
            (value * self.timestep_scale).reshape(-1),
            self.compute_dtype,
        )
        modulation = modulation.reshape(
            batch,
            -1,
            adaln.num_embeddings,
            adaln.embedding_dim,
        )
        embedded = embedded.reshape(batch, -1, adaln.embedding_dim)
        return modulation, embedded

    def _context_mask(self, mask: mx.array | None) -> mx.array | None:
        if mask is None:
            return None
        if mask.dtype in (mx.float16, mx.bfloat16, mx.float32):
            if mask.ndim == 2:
                mask = mask[:, None, None, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            return mask.astype(self.compute_dtype)
        return (
            (mask.astype(self.compute_dtype) - 1.0) * -_minimum_for_dtype(self.compute_dtype)
        ).reshape(mask.shape[0], 1, -1, mask.shape[-1])

    def _self_attention_mask(self, mask: mx.array | None) -> mx.array | None:
        if mask is None:
            return None
        value = mask.astype(mx.float32)
        positive = value > 0
        log_bias = mx.log(mx.maximum(value, _tiny_for_dtype(self.compute_dtype)))
        bias = mx.where(positive, log_bias, _minimum_for_dtype(self.compute_dtype))
        return bias[:, None, ...].astype(self.compute_dtype)

    def _rope(
        self,
        positions: mx.array,
        *,
        dim: int,
        max_pos: tuple[int, ...],
        heads: int,
    ) -> tuple[mx.array, mx.array]:
        return precompute_freqs_cis(
            positions,
            dim=dim,
            out_dtype=mx.float32,
            theta=self.theta,
            max_pos=max_pos,
            use_middle_indices_grid=True,
            num_attention_heads=heads,
            use_double_precision=self.double_precision_rope,
        )

    @staticmethod
    def _sigma(value: mx.array, batch: int) -> mx.array:
        if value.ndim == 0:
            return mx.broadcast_to(value[None], (batch,))
        if value.ndim != 1 or value.shape[0] != batch:
            raise ValueError("modality sigma must be scalar or have shape (batch,)")
        return value

    def prepare(
        self,
        modality: Modality,
        cross_modality: Modality | None,
    ) -> TransformerArgs:
        hidden = self.patchify_proj(modality.latent)
        keyframes_mask = modality.keyframes_mask
        if keyframes_mask is not None:
            if self.keyframes_abs_pos_embedding is None:
                raise ValueError(
                    "keyframes_mask requires an LTX transformer with learned keyframe positions"
                )
            if keyframes_mask.ndim == 2:
                keyframes_mask = keyframes_mask[..., None]
            if keyframes_mask.ndim != 3 or keyframes_mask.shape[:2] != hidden.shape[:2]:
                raise ValueError(
                    "keyframes_mask must have shape (batch, tokens) or (batch, tokens, 1)"
                )
            if keyframes_mask.shape[2] != 1:
                raise ValueError("keyframes_mask must have a singleton feature dimension")
            hidden = (
                hidden
                + (keyframes_mask > 0).astype(hidden.dtype)
                * (self.keyframes_abs_pos_embedding().astype(hidden.dtype)[None, ...])
            )
        batch = hidden.shape[0]
        timestep, embedded_timestep = self._timestep(
            modality.timesteps,
            self.adaln,
            batch,
        )
        sigma = self._sigma(modality.sigma, batch)
        prompt_timestep = None
        if self.prompt_adaln is not None:
            prompt_timestep, _ = self._timestep(sigma, self.prompt_adaln, batch)
        context = modality.context.reshape(batch, -1, self.inner_dim)
        positional = modality.positional_embeddings or self._rope(
            modality.positions,
            dim=self.inner_dim,
            max_pos=self.max_pos,
            heads=self.heads,
        )
        args = TransformerArgs(
            x=hidden,
            context=context,
            context_mask=self._context_mask(modality.context_mask),
            self_attention_mask=self._self_attention_mask(modality.attention_mask),
            timesteps=timestep,
            embedded_timestep=embedded_timestep,
            positional_embeddings=positional,
            prompt_timestep=prompt_timestep,
            enabled=modality.enabled,
        )
        if cross_modality is None:
            return args

        cross_positional = modality.cross_positional_embeddings or self._rope(
            modality.positions[:, 0:1, ...],
            dim=self.cross_dim,
            max_pos=(self.cross_max_pos,),
            heads=self.heads,
        )
        scale_shift, _ = self._timestep(
            modality.timesteps,
            self.cross_scale_shift_adaln,
            batch,
        )
        cross_sigma = self._sigma(cross_modality.sigma, batch)
        gate, _ = self.cross_gate_adaln(
            (cross_sigma * self.av_gate_scale).reshape(-1),
            self.compute_dtype,
        )
        gate = gate.reshape(batch, -1, 1, self.inner_dim)
        return replace(
            args,
            cross_positional_embeddings=cross_positional,
            cross_scale_shift_timestep=scale_shift,
            cross_gate_timestep=gate,
        )


__all__ = ["ModalityPreprocessor"]
