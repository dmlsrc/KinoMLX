"""Top-level LTX-2 joint audio/video velocity model."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, TypedDict, TypeGuard, Unpack, cast

import mlx.core as mx

import kinomlx._mlx_nn as nn

from .attention import _linear_shell
from .graph import TransformerGraphSpec, transformer_parameter_shapes
from .preprocessing import ModalityPreprocessor
from .timestep import AdaLayerNormSingle, resolve_transformer_dtype
from .transformer import BasicAVTransformerBlock, TransformerArgs, TransformerConfig
from .wrappers import Modality

if TYPE_CHECKING:
    from ..metadata import TransformerConstructorConfig

_log = logging.getLogger(__name__)

type _PackedTransformerArgs = tuple[
    mx.array,
    mx.array,
    mx.array,
    tuple[mx.array, mx.array],
    mx.array | None,
    mx.array | None,
    mx.array | None,
    mx.array | None,
    tuple[mx.array, mx.array] | None,
    mx.array | None,
    mx.array | None,
]
type _CompiledBlockGroup = Callable[
    [_PackedTransformerArgs | None, _PackedTransformerArgs | None],
    tuple[_PackedTransformerArgs | None, _PackedTransformerArgs | None],
]


class _LTXAVModelOptions(TypedDict, total=False):
    compute_dtype: str | mx.Dtype
    double_precision_rope: bool
    apply_gated_attention: bool
    use_steel_attention: bool
    compile_attention: bool
    steel_attention_d64: bool
    steel_attention_probe: bool
    fast_mode: bool
    compile_block_groups: int | None
    transformer_compile_group_size: int | None


def _pack_transformer_args(args: TransformerArgs | None) -> _PackedTransformerArgs | None:
    """Flatten prepared arguments across an ``mx.compile`` boundary."""
    if args is None:
        return None
    return (
        args.x,
        args.context,
        args.timesteps,
        args.positional_embeddings,
        args.context_mask,
        args.self_attention_mask,
        args.embedded_timestep,
        args.prompt_timestep,
        args.cross_positional_embeddings,
        args.cross_scale_shift_timestep,
        args.cross_gate_timestep,
    )


def _unpack_transformer_args(
    packed: _PackedTransformerArgs | None,
) -> TransformerArgs | None:
    """Restore prepared arguments after an ``mx.compile`` boundary."""
    if packed is None:
        return None
    (
        x,
        context,
        timesteps,
        positional_embeddings,
        context_mask,
        self_attention_mask,
        embedded_timestep,
        prompt_timestep,
        cross_positional_embeddings,
        cross_scale_shift_timestep,
        cross_gate_timestep,
    ) = packed
    return TransformerArgs(
        x=x,
        context=context,
        timesteps=timesteps,
        positional_embeddings=positional_embeddings,
        context_mask=context_mask,
        self_attention_mask=self_attention_mask,
        embedded_timestep=embedded_timestep,
        prompt_timestep=prompt_timestep,
        cross_positional_embeddings=cross_positional_embeddings,
        cross_scale_shift_timestep=cross_scale_shift_timestep,
        cross_gate_timestep=cross_gate_timestep,
        enabled=True,
    )


def _compile_transformer_block_group(
    blocks: Sequence[BasicAVTransformerBlock],
) -> _CompiledBlockGroup:
    """Compile a block group while keeping its parameters dynamic inputs."""
    captured = list(blocks)

    def call(
        video_packed: _PackedTransformerArgs | None,
        audio_packed: _PackedTransformerArgs | None,
    ) -> tuple[_PackedTransformerArgs | None, _PackedTransformerArgs | None]:
        video = _unpack_transformer_args(video_packed)
        audio = _unpack_transformer_args(audio_packed)
        for block in captured:
            video, audio = block(video, audio)
        return _pack_transformer_args(video), _pack_transformer_args(audio)

    return cast(_CompiledBlockGroup, mx.compile(call, inputs=captured))


def _matrix_parameter_shell() -> mx.array:
    return mx.zeros((0, 0), dtype=mx.float32)


class LTXAVModel(nn.Module):
    """A metadata-selected LTX-2 audio/video diffusion transformer."""

    def __init__(
        self,
        *,
        num_layers: int = 48,
        video_heads: int = 32,
        video_head_dim: int = 128,
        audio_heads: int = 32,
        audio_head_dim: int = 64,
        video_in_channels: int = 128,
        video_out_channels: int = 128,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        video_context_dim: int = 4096,
        audio_context_dim: int = 2048,
        video_max_pos: tuple[int, int, int] = (20, 2048, 2048),
        audio_max_pos: tuple[int] = (20,),
        positional_embedding_theta: float = 10000.0,
        timestep_scale_multiplier: float = 1000.0,
        av_ca_timestep_scale_multiplier: float = 1000.0,
        norm_eps: float = 1e-6,
        ff_bias: bool = True,
        audio_ff_bias: bool = True,
        use_keyframes_abs_pos_embedding: bool = False,
        use_prompt_adaln_single: bool = True,
        model_generation: str = "2.3",
        compute_dtype: str | mx.Dtype = mx.bfloat16,
        double_precision_rope: bool = True,
        apply_gated_attention: bool = True,
        use_steel_attention: bool = True,
        compile_attention: bool = True,
        steel_attention_d64: bool = True,
        steel_attention_probe: bool = False,
        fast_mode: bool = True,
        compile_block_groups: int | None = None,
        transformer_compile_group_size: int | None = None,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("transformer must contain at least one block")
        if video_heads != audio_heads:
            raise ValueError("LTX-2 A/V cross-attention requires equal head counts")
        if compile_block_groups is not None and compile_block_groups <= 0:
            raise ValueError("compile_block_groups must be positive")
        if transformer_compile_group_size is not None and transformer_compile_group_size <= 0:
            raise ValueError("transformer_compile_group_size must be positive")
        self.compute_dtype = resolve_transformer_dtype(compute_dtype)
        self.norm_eps = norm_eps
        self.video_inner_dim = video_heads * video_head_dim
        self.audio_inner_dim = audio_heads * audio_head_dim
        self.inner_dim = self.video_inner_dim
        if video_context_dim != self.video_inner_dim:
            raise ValueError("LTX-2 video context must match the video hidden dimension")
        if audio_context_dim != self.audio_inner_dim:
            raise ValueError("LTX-2 audio context must match the audio hidden dimension")
        if model_generation not in {"2.3", "2.5"}:
            raise ValueError(f"unsupported LTX model generation {model_generation!r}")
        graph_spec = TransformerGraphSpec(
            num_layers=num_layers,
            video_in_channels=video_in_channels,
            audio_in_channels=audio_in_channels,
            video_out_channels=video_out_channels,
            video_heads=video_heads,
            video_head_dim=video_head_dim,
            audio_heads=audio_heads,
            audio_head_dim=audio_head_dim,
            audio_out_channels=audio_out_channels,
            video_context_dim=video_context_dim,
            audio_context_dim=audio_context_dim,
            ff_bias=ff_bias,
            audio_ff_bias=audio_ff_bias,
            use_keyframes_abs_pos_embedding=use_keyframes_abs_pos_embedding,
            use_prompt_adaln_single=use_prompt_adaln_single,
        )
        object.__setattr__(self, "_graph_spec", graph_spec)
        object.__setattr__(self, "model_generation", model_generation)

        self.patchify_proj = _linear_shell(bias=True)
        self.adaln_single = AdaLayerNormSingle(self.video_inner_dim, 9)
        self.prompt_adaln_single = (
            AdaLayerNormSingle(self.video_inner_dim, 2) if use_prompt_adaln_single else None
        )
        self.scale_shift_table = _matrix_parameter_shell()
        self.norm_out = nn.LayerNorm(self.video_inner_dim, affine=False, eps=norm_eps)
        self.proj_out = _linear_shell(bias=True)
        if use_keyframes_abs_pos_embedding:
            self.keyframes_abs_pos_embedding = _matrix_parameter_shell()

        self.audio_patchify_proj = _linear_shell(bias=True)
        self.audio_adaln_single = AdaLayerNormSingle(self.audio_inner_dim, 9)
        self.audio_prompt_adaln_single = (
            AdaLayerNormSingle(self.audio_inner_dim, 2) if use_prompt_adaln_single else None
        )
        self.audio_scale_shift_table = _matrix_parameter_shell()
        self.audio_norm_out = nn.LayerNorm(
            self.audio_inner_dim,
            affine=False,
            eps=norm_eps,
        )
        self.audio_proj_out = _linear_shell(bias=True)

        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(self.video_inner_dim, 4)
        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(self.video_inner_dim, 1)
        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(self.audio_inner_dim, 4)
        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(self.audio_inner_dim, 1)

        video_config = TransformerConfig(
            dim=self.video_inner_dim,
            heads=video_heads,
            head_dim=video_head_dim,
            context_dim=video_context_dim,
            ff_bias=ff_bias,
            apply_gated_attention=apply_gated_attention,
            use_steel_attention=use_steel_attention,
            compile_attention=compile_attention,
            steel_attention_d64=steel_attention_d64,
            steel_attention_probe=steel_attention_probe,
        )
        audio_config = TransformerConfig(
            dim=self.audio_inner_dim,
            heads=audio_heads,
            head_dim=audio_head_dim,
            context_dim=audio_context_dim,
            ff_bias=audio_ff_bias,
            apply_gated_attention=apply_gated_attention,
            use_steel_attention=use_steel_attention,
            compile_attention=compile_attention,
            steel_attention_d64=steel_attention_d64,
            steel_attention_probe=steel_attention_probe,
        )
        self.transformer_blocks = [
            BasicAVTransformerBlock(
                index,
                video_config=video_config,
                audio_config=audio_config,
                norm_eps=norm_eps,
            )
            for index in range(num_layers)
        ]
        self.transformer_block_streamer = None
        self._eval_frequency = 0 if fast_mode else 8
        self.compile_block_groups = compile_block_groups
        self.transformer_compile_group_size = transformer_compile_group_size
        object.__setattr__(self, "_compiled_transformer_block_groups", {})
        object.__setattr__(self, "_transformer_block_compile_disabled", False)

        cross_max_pos = max(video_max_pos[0], audio_max_pos[0])
        self._video_preprocessor = ModalityPreprocessor(
            patchify_proj=self.patchify_proj,
            adaln=self.adaln_single,
            prompt_adaln=self.prompt_adaln_single,
            cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
            cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
            inner_dim=self.video_inner_dim,
            max_pos=video_max_pos,
            heads=video_heads,
            keyframes_abs_pos_embedding=(
                (lambda: self.keyframes_abs_pos_embedding)
                if use_keyframes_abs_pos_embedding
                else None
            ),
            cross_dim=self.audio_inner_dim,
            cross_max_pos=cross_max_pos,
            timestep_scale=timestep_scale_multiplier,
            av_gate_scale=av_ca_timestep_scale_multiplier,
            theta=positional_embedding_theta,
            compute_dtype=self.compute_dtype,
            double_precision_rope=double_precision_rope,
        )
        self._audio_preprocessor = ModalityPreprocessor(
            patchify_proj=self.audio_patchify_proj,
            adaln=self.audio_adaln_single,
            prompt_adaln=self.audio_prompt_adaln_single,
            cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
            cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
            inner_dim=self.audio_inner_dim,
            max_pos=audio_max_pos,
            heads=audio_heads,
            cross_dim=self.audio_inner_dim,
            cross_max_pos=cross_max_pos,
            timestep_scale=timestep_scale_multiplier,
            av_gate_scale=av_ca_timestep_scale_multiplier,
            theta=positional_embedding_theta,
            compute_dtype=self.compute_dtype,
            double_precision_rope=double_precision_rope,
        )

    @classmethod
    def from_config(
        cls,
        config: TransformerConstructorConfig,
        **options: Unpack[_LTXAVModelOptions],
    ) -> LTXAVModel:
        """Construct exactly the graph selected by inspected checkpoint facts."""
        return cls(
            num_layers=config.num_layers,
            video_heads=config.video_heads,
            video_head_dim=config.video_head_dim,
            audio_heads=config.audio_heads,
            audio_head_dim=config.audio_head_dim,
            video_in_channels=config.video_in_channels,
            video_out_channels=config.video_out_channels,
            audio_in_channels=config.video_in_channels,
            audio_out_channels=config.audio_out_channels,
            video_context_dim=config.video_context_dim,
            audio_context_dim=config.audio_context_dim,
            video_max_pos=config.video_max_pos,
            audio_max_pos=config.audio_max_pos,
            positional_embedding_theta=config.positional_embedding_theta,
            timestep_scale_multiplier=config.timestep_scale_multiplier,
            av_ca_timestep_scale_multiplier=config.av_ca_timestep_scale_multiplier,
            norm_eps=config.norm_eps,
            ff_bias=config.ff_bias,
            audio_ff_bias=config.audio_ff_bias,
            use_keyframes_abs_pos_embedding=config.use_keyframes_abs_pos_embedding,
            use_prompt_adaln_single=config.use_prompt_adaln_single,
            model_generation=config.model_generation,
            **options,
        )

    def expected_parameter_shapes(
        self,
        *,
        include_audio: bool = True,
    ) -> dict[str, tuple[int, ...]]:
        """Return the selected logical graph in checkpoint weight layout."""
        spec = object.__getattribute__(self, "_graph_spec")
        return transformer_parameter_shapes(spec, include_audio=include_audio)

    @property
    def num_blocks(self) -> int:
        streamer = self.transformer_block_streamer
        return streamer.block_count if streamer is not None else len(self.transformer_blocks)

    def _cast_modality(self, modality: Modality) -> Modality:
        positional = modality.positional_embeddings
        if positional is not None:
            positional = (
                positional[0].astype(mx.float32),
                positional[1].astype(mx.float32),
            )
        cross_positional = modality.cross_positional_embeddings
        if cross_positional is not None:
            cross_positional = (
                cross_positional[0].astype(mx.float32),
                cross_positional[1].astype(mx.float32),
            )
        return replace(
            modality,
            latent=modality.latent.astype(self.compute_dtype),
            context=modality.context.astype(self.compute_dtype),
            positional_embeddings=positional,
            cross_positional_embeddings=cross_positional,
        )

    @staticmethod
    def _present(modality: Modality | None) -> TypeGuard[Modality]:
        return modality is not None and modality.latent.size > 0

    @staticmethod
    def _eval_args(
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
    ) -> None:
        arrays = []
        if video is not None:
            arrays.append(video.x)
        if audio is not None:
            arrays.append(audio.x)
        if arrays:
            mx.eval(*arrays)

    def _process_blocks(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        # The packed compile boundary deliberately omits the Python ``enabled``
        # flag. Guided denoisers can keep a disabled stream present as K/V
        # context for the other modality, so those calls must retain the eager
        # branch semantics instead of being unpacked as enabled.
        compile_modalities_enabled = all(args is None or args.enabled for args in (video, audio))
        streamer = self.transformer_block_streamer
        if streamer is None:
            group_size = self.compile_block_groups
            compile_disabled = object.__getattribute__(
                self,
                "_transformer_block_compile_disabled",
            )
            if group_size is not None and not compile_disabled and compile_modalities_enabled:
                initial_video, initial_audio = video, audio
                compiled = object.__getattribute__(
                    self,
                    "_compiled_transformer_block_groups",
                )
                try:
                    for start in range(0, len(self.transformer_blocks), group_size):
                        blocks = self.transformer_blocks[start : start + group_size]
                        key = ("eager", start, len(blocks))
                        call = compiled.get(key)
                        if call is None:
                            call = _compile_transformer_block_group(blocks)
                            compiled[key] = call
                        video_packed, audio_packed = call(
                            _pack_transformer_args(video),
                            _pack_transformer_args(audio),
                        )
                        video = _unpack_transformer_args(video_packed)
                        audio = _unpack_transformer_args(audio_packed)
                        if self._eval_frequency:
                            self._eval_args(video, audio)
                    return video, audio
                except (RuntimeError, TypeError, ValueError) as exc:
                    _log.warning(
                        "Transformer block-group compile failed; using eager blocks: %s",
                        exc,
                    )
                    object.__setattr__(
                        self,
                        "_transformer_block_compile_disabled",
                        True,
                    )
                    object.__setattr__(self, "_compiled_transformer_block_groups", {})
                    video, audio = initial_video, initial_audio
            for index, block in enumerate(self.transformer_blocks):
                video, audio = block(video, audio)
                if self._eval_frequency and (index + 1) % self._eval_frequency == 0:
                    self._eval_args(video, audio)
            return video, audio

        resident = len(self.transformer_blocks)
        if resident == 0:
            raise ValueError("streaming transformer has no resident block slots")
        previous: list[int | None] = [None] * resident
        compile_group_size = self.transformer_compile_group_size
        compile_disabled = object.__getattribute__(
            self,
            "_transformer_block_compile_disabled",
        )
        if compile_group_size is not None and not compile_disabled and compile_modalities_enabled:
            compiled = object.__getattribute__(
                self,
                "_compiled_transformer_block_groups",
            )
            use_compiled = True
            for window_start in range(0, streamer.block_count, resident):
                window_size = min(
                    resident,
                    streamer.block_count - window_start,
                )
                for offset in range(window_size):
                    block_index = window_start + offset
                    streamer.bind(
                        self.transformer_blocks[offset],
                        block_index,
                        evict_block_idx=previous[offset],
                    )
                    previous[offset] = block_index
                for group_start in range(0, window_size, compile_group_size):
                    blocks = self.transformer_blocks[
                        group_start : min(
                            group_start + compile_group_size,
                            window_size,
                        )
                    ]
                    if use_compiled:
                        key = ("streaming", group_start, len(blocks))
                        call = compiled.get(key)
                        if call is None:
                            call = _compile_transformer_block_group(blocks)
                            compiled[key] = call
                        try:
                            video_packed, audio_packed = call(
                                _pack_transformer_args(video),
                                _pack_transformer_args(audio),
                            )
                            video = _unpack_transformer_args(video_packed)
                            audio = _unpack_transformer_args(audio_packed)
                        except (RuntimeError, TypeError, ValueError) as exc:
                            _log.warning(
                                "Streaming block-group compile failed; using eager blocks: %s",
                                exc,
                            )
                            use_compiled = False
                            object.__setattr__(
                                self,
                                "_transformer_block_compile_disabled",
                                True,
                            )
                            object.__setattr__(
                                self,
                                "_compiled_transformer_block_groups",
                                {},
                            )
                    if not use_compiled:
                        for block in blocks:
                            video, audio = block(video, audio)
                    self._eval_args(video, audio)
            return video, audio

        for index in range(streamer.block_count):
            slot = index % resident
            block = streamer.bind(
                self.transformer_blocks[slot],
                index,
                evict_block_idx=previous[slot],
            )
            previous[slot] = index
            video, audio = block(video, audio)
            if (index + 1) % resident == 0 or index + 1 == streamer.block_count:
                self._eval_args(video, audio)
        return video, audio

    def _output(
        self,
        value: mx.array,
        embedded_timestep: mx.array | None,
        table: mx.array,
        norm: Callable[[mx.array], mx.array],
        projection: Callable[[mx.array], mx.array],
    ) -> mx.array:
        if embedded_timestep is None:
            raise ValueError("transformer output requires an embedded timestep")
        modulation = (table[None, None, :, :] + embedded_timestep[:, :, None, :]).astype(
            value.dtype
        )
        shift = modulation[:, :, 0, :]
        scale = modulation[:, :, 1, :]
        value = (norm(value) * (1.0 + scale) + shift).astype(value.dtype)
        return projection(value)

    def __call__(
        self,
        video: Modality | None,
        audio: Modality | None = None,
    ) -> tuple[mx.array | None, mx.array | None]:
        if video is None and audio is None:
            raise ValueError("at least one modality is required")
        video = self._cast_modality(video) if video is not None else None
        audio = self._cast_modality(audio) if audio is not None else None
        video_cross = audio if self._present(audio) else None
        audio_cross = video if self._present(video) else None
        video_args = (
            self._video_preprocessor.prepare(video, video_cross) if self._present(video) else None
        )
        audio_args = (
            self._audio_preprocessor.prepare(audio, audio_cross) if self._present(audio) else None
        )
        video_args, audio_args = self._process_blocks(video_args, audio_args)
        video_output = (
            self._output(
                video_args.x,
                video_args.embedded_timestep,
                self.scale_shift_table,
                self.norm_out,
                self.proj_out,
            )
            if video_args is not None
            else None
        )
        audio_output = (
            self._output(
                audio_args.x,
                audio_args.embedded_timestep,
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
            )
            if audio_args is not None
            else None
        )
        return video_output, audio_output

    def close_streamer(self) -> None:
        """Release a cache-backed block streamer without owning other models."""
        if self.transformer_block_streamer is not None:
            self.transformer_block_streamer.close()
            self.transformer_block_streamer = None
        object.__setattr__(self, "_compiled_transformer_block_groups", {})


LTXModel = LTXAVModel

__all__ = [
    "LTXAVModel",
    "LTXModel",
    "resolve_transformer_dtype",
]
