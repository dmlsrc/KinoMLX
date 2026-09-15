"""Gemma hidden-state aggregation and LTX audio/video context encoding."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.io.safetensors import read_metadata
from kinomlx.reporting import NullReporter, Reporter

from .connector import Embeddings1DConnector, RopeType
from .features import GemmaFeaturesExtractorV2


def _config_int(config: Mapping[str, object], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"connector {key} must be an integer")
    return value


def _config_float(config: Mapping[str, object], key: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"connector {key} must be numeric")
    return float(value)


def _config_bool(config: Mapping[str, object], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"connector {key} must be a boolean")
    return value


@dataclass(frozen=True)
class AudioVideoGemmaEncoderOutput:
    """Separate video/audio contexts and their all-valid connector mask."""

    video_encoding: mx.array
    audio_encoding: mx.array
    attention_mask: mx.array


@dataclass(frozen=True)
class AVTextEncoderConfig:
    """Shared LTX-2.x connector architecture derived from transformer metadata."""

    hidden_dim: int = 3840
    num_gemma_states: int = 49
    video_inner_dim: int = 4096
    audio_inner_dim: int = 2048
    video_heads: int = 32
    video_head_dim: int = 128
    audio_heads: int = 32
    audio_head_dim: int = 64
    num_layers: int = 8
    audio_num_layers: int | None = None
    num_registers: int = 128
    positional_theta: float = 10000.0
    positional_max: tuple[int, ...] = (4096,)
    norm_eps: float = 1e-6
    gated_attention: bool = True
    ff_bias: bool = True
    double_precision_rope: bool = True
    rope_type: RopeType = RopeType.SPLIT

    def __post_init__(self) -> None:
        dimensions = (
            self.hidden_dim,
            self.num_gemma_states,
            self.video_inner_dim,
            self.audio_inner_dim,
            self.video_heads,
            self.video_head_dim,
            self.audio_heads,
            self.audio_head_dim,
            self.num_layers,
            self.num_registers,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("AV text-encoder dimensions must be positive")
        if self.video_heads * self.video_head_dim != self.video_inner_dim:
            raise ValueError("video connector heads do not match video_inner_dim")
        if self.audio_heads * self.audio_head_dim != self.audio_inner_dim:
            raise ValueError("audio connector heads do not match audio_inner_dim")
        if self.audio_num_layers is not None and self.audio_num_layers <= 0:
            raise ValueError("audio connector layer count must be positive")
        if not self.positional_max or any(value <= 0 for value in self.positional_max):
            raise ValueError("connector positional maxima must be positive")
        if self.rope_type is not RopeType.SPLIT:
            raise ValueError("only split connector RoPE is supported")

    @classmethod
    def from_checkpoint(cls, path: Path | str) -> AVTextEncoderConfig:
        """Parse the required connector fields from LTX checkpoint metadata."""
        metadata = read_metadata(path)
        try:
            root: object = json.loads(metadata["config"])
        except KeyError as exc:
            raise ValueError(f"{path}: checkpoint metadata has no config") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: checkpoint config is invalid JSON") from exc
        transformer_value = root.get("transformer") if isinstance(root, dict) else None
        if not isinstance(transformer_value, dict):
            raise ValueError(f"{path}: checkpoint has no transformer config")
        transformer = cast(Mapping[str, object], transformer_value)
        rope_name = str(transformer.get("rope_type", "split")).lower()
        if rope_name != "split":
            raise ValueError(f"unsupported connector rope_type={rope_name!r}")
        positional_max = transformer.get("connector_positional_embedding_max_pos", [4096])
        if (
            not isinstance(positional_max, list)
            or not positional_max
            or any(
                isinstance(value, bool) or not isinstance(value, int) for value in positional_max
            )
        ):
            raise ValueError("connector_positional_embedding_max_pos must be an integer list")
        positional_max_values = cast(list[int], positional_max)
        video_heads = _config_int(transformer, "connector_num_attention_heads", 32)
        video_head_dim = _config_int(transformer, "connector_attention_head_dim", 128)
        audio_head_dim_default = (
            video_head_dim if "connector_attention_head_dim" in transformer else 64
        )
        connector_layers = _config_int(transformer, "connector_num_layers", 8)
        return cls(
            hidden_dim=_config_int(transformer, "caption_channels", 3840),
            num_gemma_states=49,
            video_inner_dim=_config_int(transformer, "cross_attention_dim", 4096),
            audio_inner_dim=_config_int(transformer, "audio_cross_attention_dim", 2048),
            video_heads=video_heads,
            video_head_dim=video_head_dim,
            audio_heads=_config_int(
                transformer,
                "audio_connector_num_attention_heads",
                video_heads,
            ),
            audio_head_dim=_config_int(
                transformer,
                "audio_connector_attention_head_dim",
                audio_head_dim_default,
            ),
            num_layers=connector_layers,
            audio_num_layers=_config_int(
                transformer,
                "audio_connector_num_layers",
                connector_layers,
            ),
            num_registers=_config_int(transformer, "connector_num_learnable_registers", 128),
            positional_theta=_config_float(transformer, "positional_embedding_theta", 10000.0),
            positional_max=tuple(positional_max_values),
            norm_eps=_config_float(transformer, "norm_eps", 1e-6),
            gated_attention=_config_bool(transformer, "connector_apply_gated_attention", True),
            ff_bias=_config_bool(transformer, "connector_ff_bias", True),
            double_precision_rope=(
                str(transformer.get("frequencies_precision", "float64")).casefold() == "float64"
            ),
            rope_type=RopeType.SPLIT,
        )


class AudioVideoGemmaTextEncoderModel(nn.Module):
    """Feature extractor followed by independent video and audio connectors."""

    def __init__(
        self,
        config: AVTextEncoderConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or AVTextEncoderConfig()
        self.feature_extractor = GemmaFeaturesExtractorV2(
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_gemma_states,
            video_inner_dim=self.config.video_inner_dim,
            audio_inner_dim=self.config.audio_inner_dim,
        )
        self.embeddings_connector = Embeddings1DConnector(
            attention_head_dim=self.config.video_head_dim,
            num_attention_heads=self.config.video_heads,
            num_layers=self.config.num_layers,
            positional_embedding_max_pos=self.config.positional_max,
            num_learnable_registers=self.config.num_registers,
            positional_embedding_theta=self.config.positional_theta,
            norm_eps=self.config.norm_eps,
            gated_attention=self.config.gated_attention,
            ff_bias=self.config.ff_bias,
            double_precision_rope=self.config.double_precision_rope,
        )
        self.audio_embeddings_connector = Embeddings1DConnector(
            attention_head_dim=self.config.audio_head_dim,
            num_attention_heads=self.config.audio_heads,
            num_layers=(
                self.config.audio_num_layers
                if self.config.audio_num_layers is not None
                else self.config.num_layers
            ),
            positional_embedding_max_pos=self.config.positional_max,
            num_learnable_registers=self.config.num_registers,
            positional_embedding_theta=self.config.positional_theta,
            norm_eps=self.config.norm_eps,
            gated_attention=self.config.gated_attention,
            ff_bias=self.config.ff_bias,
            double_precision_rope=self.config.double_precision_rope,
        )

    @staticmethod
    def _additive_mask(attention_mask: mx.array, dtype: mx.Dtype) -> mx.array:
        maximum = mx.finfo(dtype).max
        return ((attention_mask.astype(dtype) - 1) * maximum).reshape(
            attention_mask.shape[0],
            1,
            1,
            attention_mask.shape[1],
        )

    def __call__(
        self,
        hidden_states: tuple[mx.array, ...] | list[mx.array],
        attention_mask: mx.array,
        *,
        reporter: Reporter | None = None,
    ) -> AudioVideoGemmaEncoderOutput:
        if (
            attention_mask.ndim != 2
            or not hidden_states
            or attention_mask.shape[0] != hidden_states[0].shape[0]
            or attention_mask.shape[1] != hidden_states[0].shape[1]
        ):
            raise ValueError("attention_mask must match the hidden-state batch and token axes")
        if not cast(bool, mx.all(mx.equal(attention_mask, 1)).item()):
            raise ValueError(
                "padded attention masks are not supported by the public connector; "
                "trim padding before projection"
            )
        sink = reporter if reporter is not None else NullReporter()
        phase = "project audio/video text context"
        # Indeterminate phase: one lazy graph, evaluated once at the outputs.
        sink.phase_start(phase)
        try:
            video_input, audio_input = self.feature_extractor(
                hidden_states,
                attention_mask,
            )
            additive_mask = self._additive_mask(attention_mask, video_input.dtype)
            video, output_mask = self.embeddings_connector(video_input, additive_mask)
            binary_mask = (output_mask[:, 0, 0, :] >= -0.5).astype(mx.int32)
            video = video * binary_mask[..., None]
            audio, _ = self.audio_embeddings_connector(audio_input, additive_mask)
            mx.eval(video, audio, binary_mask)
            return AudioVideoGemmaEncoderOutput(video, audio, binary_mask)
        finally:
            sink.phase_end(phase)


def create_av_text_encoder_v2(
    config: AVTextEncoderConfig | None = None,
) -> AudioVideoGemmaTextEncoderModel:
    """Construct the shared LTX-2.x AV connector stack."""
    return AudioVideoGemmaTextEncoderModel(config)


def create_av_text_encoder_v2_from_checkpoint(
    path: Path | str,
) -> AudioVideoGemmaTextEncoderModel:
    """Construct the AV connector stack from checkpoint metadata."""
    return create_av_text_encoder_v2(AVTextEncoderConfig.from_checkpoint(path))


__all__ = [
    "AVTextEncoderConfig",
    "AudioVideoGemmaEncoderOutput",
    "AudioVideoGemmaTextEncoderModel",
    "create_av_text_encoder_v2",
    "create_av_text_encoder_v2_from_checkpoint",
]
