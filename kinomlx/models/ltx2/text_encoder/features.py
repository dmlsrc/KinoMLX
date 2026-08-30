"""Shared LTX-2.x feature aggregation over Gemma hidden states."""

from __future__ import annotations

import math

import mlx.core as mx

import kinomlx._mlx_nn as nn

from ._layers import linear_shell


def norm_and_concat_per_token_rms(
    encoded_text: mx.array,
    attention_mask: mx.array,
) -> mx.array:
    """Normalize each token/layer over hidden width, flatten, and zero padding."""
    if encoded_text.ndim != 4:
        raise ValueError("encoded_text must have shape (batch, tokens, hidden, layers)")
    batch, tokens, hidden, layers = encoded_text.shape
    if tuple(attention_mask.shape) != (batch, tokens):
        raise ValueError("attention_mask must match encoded text batch and token axes")
    variance = mx.mean(encoded_text**2, axis=2, keepdims=True)
    normalized = encoded_text * mx.rsqrt(variance + 1e-6)
    normalized = normalized.reshape(batch, tokens, hidden * layers)
    return mx.where(
        attention_mask.astype(mx.bool_)[..., None],
        normalized,
        mx.zeros_like(normalized),
    )


class GemmaFeaturesExtractorV2(nn.Module):
    """Project 49 Gemma states to separate video and audio context widths."""

    def __init__(
        self,
        *,
        hidden_dim: int = 3840,
        num_layers: int = 49,
        video_inner_dim: int = 4096,
        audio_inner_dim: int = 2048,
    ) -> None:
        super().__init__()
        if min(hidden_dim, num_layers, video_inner_dim, audio_inner_dim) <= 0:
            raise ValueError("feature-extractor dimensions must be positive")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.video_inner_dim = video_inner_dim
        self.audio_inner_dim = audio_inner_dim
        self.video_aggregate_embed = linear_shell(bias=True)
        self.audio_aggregate_embed = linear_shell(bias=True)

    def __call__(
        self,
        hidden_states: tuple[mx.array, ...] | list[mx.array],
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if len(hidden_states) != self.num_layers:
            raise ValueError(
                f"expected {self.num_layers} Gemma hidden states, got {len(hidden_states)}"
            )
        encoded = mx.stack(list(hidden_states), axis=-1)
        if encoded.shape[2] != self.hidden_dim:
            raise ValueError(
                f"expected Gemma hidden width {self.hidden_dim}, got {encoded.shape[2]}"
            )
        normalized = norm_and_concat_per_token_rms(encoded, attention_mask)
        normalized = normalized.astype(encoded.dtype)
        video = self.video_aggregate_embed(
            normalized * math.sqrt(self.video_aggregate_embed.weight.shape[0] / self.hidden_dim)
        )
        audio = self.audio_aggregate_embed(
            normalized * math.sqrt(self.audio_aggregate_embed.weight.shape[0] / self.hidden_dim)
        )
        if video.shape[-1] != self.video_inner_dim:
            raise ValueError(
                f"video projection produced width {video.shape[-1]}, "
                f"expected {self.video_inner_dim}"
            )
        if audio.shape[-1] != self.audio_inner_dim:
            raise ValueError(
                f"audio projection produced width {audio.shape[-1]}, "
                f"expected {self.audio_inner_dim}"
            )
        return video, audio


__all__ = ["GemmaFeaturesExtractorV2", "norm_and_concat_per_token_rms"]
