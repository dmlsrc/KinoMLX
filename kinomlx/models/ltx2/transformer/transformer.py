"""Joint audio/video DiT blocks for the LTX-2 velocity model."""

from __future__ import annotations

from dataclasses import dataclass, replace

import mlx.core as mx

import kinomlx._mlx_nn as nn

from .attention import Attention, rms_norm
from .feed_forward import FeedForward
from .rope import LTXRopeType


def _adaln(
    value: mx.array,
    scale: mx.array,
    shift: mx.array,
    eps: float,
) -> mx.array:
    normalized = rms_norm(value, eps=eps)
    return (normalized * (1.0 + scale) + shift).astype(value.dtype)


def _gated_residual(
    value: mx.array,
    residual: mx.array,
    gate: mx.array,
) -> mx.array:
    return (value + residual * gate).astype(value.dtype)


_compiled_adaln = mx.compile(_adaln)
_compiled_gated_residual = mx.compile(_gated_residual)


def _table_shell() -> mx.array:
    return mx.zeros((0, 0), dtype=mx.float32)


@dataclass(frozen=True)
class TransformerConfig:
    """Shape and feature selection for one transformer stream."""

    dim: int
    heads: int
    head_dim: int
    context_dim: int
    ff_bias: bool = True
    apply_gated_attention: bool = True
    use_steel_attention: bool = True
    compile_attention: bool = True
    steel_attention_d64: bool = True
    steel_attention_probe: bool = False


@dataclass(frozen=True)
class TransformerArgs:
    """Prepared inputs carried through every joint transformer block."""

    x: mx.array
    context: mx.array
    timesteps: mx.array
    positional_embeddings: tuple[mx.array, mx.array]
    context_mask: mx.array | None = None
    self_attention_mask: mx.array | None = None
    embedded_timestep: mx.array | None = None
    prompt_timestep: mx.array | None = None
    cross_positional_embeddings: tuple[mx.array, mx.array] | None = None
    cross_scale_shift_timestep: mx.array | None = None
    cross_gate_timestep: mx.array | None = None
    enabled: bool = True


class BasicAVTransformerBlock(nn.Module):
    """One LTX-2 block spanning video, audio, text, and A/V attention."""

    def __init__(
        self,
        idx: int,
        *,
        video_config: TransformerConfig,
        audio_config: TransformerConfig,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.idx = idx
        self.norm_eps = norm_eps
        video_attention_backend = {
            "use_steel_attention": video_config.use_steel_attention,
            "compile_attention": video_config.compile_attention,
            "steel_attention_d64": video_config.steel_attention_d64,
            "steel_attention_probe": video_config.steel_attention_probe,
        }
        audio_attention_backend = {
            "use_steel_attention": audio_config.use_steel_attention,
            "compile_attention": audio_config.compile_attention,
            "steel_attention_d64": audio_config.steel_attention_d64,
            "steel_attention_probe": audio_config.steel_attention_probe,
        }

        self.attn1 = Attention(
            video_config.dim,
            heads=video_config.heads,
            dim_head=video_config.head_dim,
            norm_eps=norm_eps,
            rope_type=rope_type,
            apply_gated_attention=video_config.apply_gated_attention,
            **video_attention_backend,
        )
        self.attn2 = Attention(
            video_config.dim,
            context_dim=video_config.context_dim,
            heads=video_config.heads,
            dim_head=video_config.head_dim,
            norm_eps=norm_eps,
            rope_type=rope_type,
            apply_gated_attention=video_config.apply_gated_attention,
            **video_attention_backend,
        )
        self.ff = FeedForward(video_config.dim, bias=video_config.ff_bias)
        self.scale_shift_table = _table_shell()
        self.prompt_scale_shift_table = _table_shell()

        self.audio_attn1 = Attention(
            audio_config.dim,
            heads=audio_config.heads,
            dim_head=audio_config.head_dim,
            norm_eps=norm_eps,
            rope_type=rope_type,
            apply_gated_attention=audio_config.apply_gated_attention,
            **audio_attention_backend,
        )
        self.audio_attn2 = Attention(
            audio_config.dim,
            context_dim=audio_config.context_dim,
            heads=audio_config.heads,
            dim_head=audio_config.head_dim,
            norm_eps=norm_eps,
            rope_type=rope_type,
            apply_gated_attention=audio_config.apply_gated_attention,
            **audio_attention_backend,
        )
        self.audio_ff = FeedForward(audio_config.dim, bias=audio_config.ff_bias)
        self.audio_scale_shift_table = _table_shell()
        self.audio_prompt_scale_shift_table = _table_shell()

        self.audio_to_video_attn = Attention(
            video_config.dim,
            context_dim=audio_config.dim,
            heads=audio_config.heads,
            dim_head=audio_config.head_dim,
            norm_eps=norm_eps,
            rope_type=rope_type,
            apply_gated_attention=video_config.apply_gated_attention,
            **video_attention_backend,
        )
        self.video_to_audio_attn = Attention(
            audio_config.dim,
            context_dim=video_config.dim,
            heads=audio_config.heads,
            dim_head=audio_config.head_dim,
            norm_eps=norm_eps,
            rope_type=rope_type,
            apply_gated_attention=audio_config.apply_gated_attention,
            **audio_attention_backend,
        )
        self.scale_shift_table_a2v_ca_audio = _table_shell()
        self.scale_shift_table_a2v_ca_video = _table_shell()

    @staticmethod
    def _ada_values(
        table: mx.array,
        timestep: mx.array,
        start: int,
        stop: int,
    ) -> tuple[mx.array, ...]:
        values = table[start:stop][None, None, :, :] + timestep[:, :, start:stop, :]
        return tuple(values[:, :, index, :] for index in range(stop - start))

    @staticmethod
    def _av_ada_values(
        table: mx.array,
        scale_shift_timestep: mx.array,
        gate_timestep: mx.array,
    ) -> tuple[mx.array, ...]:
        scale_shift = table[:4][None, None, :, :] + scale_shift_timestep
        gate = table[4:5][None, None, :, :] + gate_timestep
        return (
            *(scale_shift[:, :, index, :] for index in range(4)),
            gate[:, :, 0, :],
        )

    def _text_cross_attention(
        self,
        value: mx.array,
        args: TransformerArgs,
        attention: Attention,
        scale_shift_table: mx.array,
        prompt_scale_shift_table: mx.array,
    ) -> mx.array:
        shift_query, scale_query, gate = self._ada_values(
            scale_shift_table,
            args.timesteps,
            6,
            9,
        )
        prompt = prompt_scale_shift_table[None, None, :, :]
        if args.prompt_timestep is not None:
            prompt = prompt + args.prompt_timestep
        shift_context = prompt[:, :, 0, :]
        scale_context = prompt[:, :, 1, :]
        query = (rms_norm(value, eps=self.norm_eps) * (1.0 + scale_query) + shift_query).astype(
            value.dtype
        )
        context = (args.context * (1.0 + scale_context) + shift_context).astype(args.context.dtype)
        output = attention(query, context=context, mask=args.context_mask)
        return (output * gate).astype(output.dtype)

    def __call__(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        video_value = video.x if video is not None else None
        audio_value = audio.x if audio is not None else None
        run_video = (
            video is not None and video_value is not None and video.enabled and video_value.size > 0
        )
        run_audio = (
            audio is not None and audio_value is not None and audio.enabled and audio_value.size > 0
        )
        # ``enabled`` gates updates to the query stream. A present disabled
        # stream remains valid K/V context for the other modality's update.
        run_audio_to_video = (
            run_video and audio is not None and audio_value is not None and audio_value.size > 0
        )
        run_video_to_audio = (
            run_audio and video is not None and video_value is not None and video_value.size > 0
        )

        if run_video:
            assert video is not None
            assert video_value is not None
            shift, scale, gate = self._ada_values(
                self.scale_shift_table,
                video.timesteps,
                0,
                3,
            )
            normalized = _compiled_adaln(
                video_value,
                scale,
                shift,
                self.norm_eps,
            )
            attended = self.attn1(
                normalized,
                pe=video.positional_embeddings,
                mask=video.self_attention_mask,
            )
            video_value = _compiled_gated_residual(video_value, attended, gate)
            video_value = (
                video_value
                + self._text_cross_attention(
                    video_value,
                    video,
                    self.attn2,
                    self.scale_shift_table,
                    self.prompt_scale_shift_table,
                )
            ).astype(video_value.dtype)

        if run_audio:
            assert audio is not None
            assert audio_value is not None
            shift, scale, gate = self._ada_values(
                self.audio_scale_shift_table,
                audio.timesteps,
                0,
                3,
            )
            normalized = _compiled_adaln(
                audio_value,
                scale,
                shift,
                self.norm_eps,
            )
            attended = self.audio_attn1(
                normalized,
                pe=audio.positional_embeddings,
                mask=audio.self_attention_mask,
            )
            audio_value = _compiled_gated_residual(audio_value, attended, gate)
            audio_value = (
                audio_value
                + self._text_cross_attention(
                    audio_value,
                    audio,
                    self.audio_attn2,
                    self.audio_scale_shift_table,
                    self.audio_prompt_scale_shift_table,
                )
            ).astype(audio_value.dtype)

        if run_audio_to_video or run_video_to_audio:
            assert video is not None
            assert audio is not None
            assert video_value is not None
            assert audio_value is not None
            if (
                video.cross_scale_shift_timestep is None
                or video.cross_gate_timestep is None
                or video.cross_positional_embeddings is None
                or audio.cross_scale_shift_timestep is None
                or audio.cross_gate_timestep is None
                or audio.cross_positional_embeddings is None
            ):
                raise ValueError("A/V attention requires cross-modal preprocessing")
            video_base = video_value
            audio_base = audio_value
            video_norm = rms_norm(video_base, eps=self.norm_eps)
            audio_norm = rms_norm(audio_base, eps=self.norm_eps)
            (
                audio_scale_a2v,
                audio_shift_a2v,
                audio_scale_v2a,
                audio_shift_v2a,
                gate_v2a,
            ) = self._av_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
            )
            (
                video_scale_a2v,
                video_shift_a2v,
                video_scale_v2a,
                video_shift_v2a,
                gate_a2v,
            ) = self._av_ada_values(
                self.scale_shift_table_a2v_ca_video,
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
            )

            if run_audio_to_video:
                a2v_query = (video_norm * (1.0 + video_scale_a2v) + video_shift_a2v).astype(
                    video_norm.dtype
                )
                a2v_context = (audio_norm * (1.0 + audio_scale_a2v) + audio_shift_a2v).astype(
                    audio_norm.dtype
                )
                a2v = self.audio_to_video_attn(
                    a2v_query,
                    context=a2v_context,
                    pe=video.cross_positional_embeddings,
                    k_pe=audio.cross_positional_embeddings,
                )
                video_value = _compiled_gated_residual(video_base, a2v, gate_a2v)

            if run_video_to_audio:
                v2a_query = (audio_norm * (1.0 + audio_scale_v2a) + audio_shift_v2a).astype(
                    audio_norm.dtype
                )
                v2a_context = (video_norm * (1.0 + video_scale_v2a) + video_shift_v2a).astype(
                    video_norm.dtype
                )
                v2a = self.video_to_audio_attn(
                    v2a_query,
                    context=v2a_context,
                    pe=audio.cross_positional_embeddings,
                    k_pe=video.cross_positional_embeddings,
                )
                audio_value = _compiled_gated_residual(audio_base, v2a, gate_v2a)

        if run_video:
            assert video is not None
            assert video_value is not None
            shift, scale, gate = self._ada_values(
                self.scale_shift_table,
                video.timesteps,
                3,
                6,
            )
            normalized = _compiled_adaln(
                video_value,
                scale,
                shift,
                self.norm_eps,
            )
            video_value = _compiled_gated_residual(
                video_value,
                self.ff(normalized),
                gate,
            )

        if run_audio:
            assert audio is not None
            assert audio_value is not None
            shift, scale, gate = self._ada_values(
                self.audio_scale_shift_table,
                audio.timesteps,
                3,
                6,
            )
            normalized = _compiled_adaln(
                audio_value,
                scale,
                shift,
                self.norm_eps,
            )
            audio_value = _compiled_gated_residual(
                audio_value,
                self.audio_ff(normalized),
                gate,
            )

        if video is not None:
            assert video_value is not None
            video = replace(video, x=video_value)
        if audio is not None:
            assert audio_value is not None
            audio = replace(audio, x=audio_value)
        return video, audio


__all__ = ["BasicAVTransformerBlock", "TransformerArgs", "TransformerConfig"]
