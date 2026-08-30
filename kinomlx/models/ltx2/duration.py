"""Natural-duration prediction from materialized LTX text conditioning."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.io.safetensors import load_weights, read_metadata
from kinomlx.kernels import gelu_approx
from kinomlx.reporting import NullReporter, Reporter

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DurationHeadArchitecture:
    """Small execution graph serialized by the LTX-2.5 duration artifact."""

    video_context_dim: int = 4096
    audio_context_dim: int = 2048
    hidden_dim: int = 256
    num_queries: int = 1
    num_heads: int = 4
    mlp_hidden_dim: int = 256

    def __post_init__(self) -> None:
        for name, value in (
            ("video_context_dim", self.video_context_dim),
            ("audio_context_dim", self.audio_context_dim),
            ("hidden_dim", self.hidden_dim),
            ("num_queries", self.num_queries),
            ("num_heads", self.num_heads),
            ("mlp_hidden_dim", self.mlp_hidden_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.num_heads:
            raise ValueError("duration hidden_dim must be divisible by num_heads")

    @classmethod
    def from_checkpoint(cls, path: Path | str) -> DurationHeadArchitecture:
        metadata = read_metadata(path)
        try:
            raw = json.loads(metadata["config"])
        except KeyError as exc:
            raise ValueError(f"{path}: duration head metadata has no config") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: duration head config is invalid JSON") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: duration head config must be an object")
        transformer = raw.get("transformer")
        duration = raw.get("duration_head")
        if not isinstance(transformer, dict) or not isinstance(duration, dict):
            raise ValueError(f"{path}: duration head config is incomplete")
        return cls(
            video_context_dim=int(transformer.get("cross_attention_dim", 4096)),
            audio_context_dim=int(transformer.get("audio_cross_attention_dim", 2048)),
            hidden_dim=int(duration.get("pooler_hidden_dim", 256)),
            num_queries=int(duration.get("num_queries", 1)),
            num_heads=int(duration.get("num_pooler_heads", 4)),
            mlp_hidden_dim=int(duration.get("mlp_hidden_dim", 256)),
        )


def snap_duration_to_frame_count(
    seconds: float,
    *,
    frame_rate: float,
    temporal_compression_ratio: int,
    min_seconds: float = 1.0,
    max_seconds: float = 20.0,
) -> int:
    """Clamp seconds, then snap the rounded count to the causal VAE grid."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"predicted duration must be finite and positive, got {seconds!r}")
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError("frame_rate must be finite and positive")
    if (
        isinstance(temporal_compression_ratio, bool)
        or not isinstance(temporal_compression_ratio, int)
        or temporal_compression_ratio <= 0
    ):
        raise ValueError("temporal_compression_ratio must be a positive integer")
    if (
        not math.isfinite(min_seconds)
        or not math.isfinite(max_seconds)
        or min_seconds <= 0
        or max_seconds < min_seconds
    ):
        raise ValueError("duration bounds must be finite, positive, and ordered")

    min_frames = max(1, round(min_seconds * frame_rate))
    max_frames = round(max_seconds * frame_rate)
    clamped = max(min_frames, min(round(seconds * frame_rate), max_frames))
    snapped = ((clamped - 1) // temporal_compression_ratio) * temporal_compression_ratio + 1
    if snapped < min_frames:
        snapped_up = snapped + temporal_compression_ratio
        if snapped_up <= max_frames:
            snapped = snapped_up
        else:
            if abs(snapped_up - clamped) < abs(snapped - clamped):
                snapped = snapped_up
            _log.warning(
                "Duration bounds [%.2fs, %.2fs] at %.2f fps contain no k*%d+1 frame count; "
                "using nearest %d-frame count",
                min_seconds,
                max_seconds,
                frame_rate,
                temporal_compression_ratio,
                snapped,
            )
    return snapped


class DurationHead(nn.Module):
    """Predict a positive duration from either or both connector streams."""

    def __init__(
        self,
        architecture: DurationHeadArchitecture | None = None,
        *,
        compute_dtype: mx.Dtype = mx.bfloat16,
    ) -> None:
        super().__init__()
        self.architecture = architecture or DurationHeadArchitecture()
        self.compute_dtype = compute_dtype
        config = self.architecture
        self.video_input_proj = nn.Linear(config.video_context_dim, config.hidden_dim)
        self.audio_input_proj = nn.Linear(config.audio_context_dim, config.hidden_dim)
        self.video_modality_emb = mx.zeros((config.hidden_dim,), dtype=compute_dtype)
        self.audio_modality_emb = mx.zeros((config.hidden_dim,), dtype=compute_dtype)
        self.query_tokens = mx.zeros(
            (config.num_queries, config.hidden_dim),
            dtype=compute_dtype,
        )
        self.in_proj_weight = mx.zeros(
            (config.hidden_dim * 3, config.hidden_dim),
            dtype=compute_dtype,
        )
        self.in_proj_bias = mx.zeros((config.hidden_dim * 3,), dtype=compute_dtype)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.mlp_hidden = nn.Linear(
            config.hidden_dim * config.num_queries,
            config.mlp_hidden_dim,
        )
        self.mlp_out = nn.Linear(config.mlp_hidden_dim, 1)
        self.set_dtype(compute_dtype)

    def _pool(self, tokens: mx.array) -> mx.array:
        config = self.architecture
        batch = tokens.shape[0]
        queries = mx.broadcast_to(
            self.query_tokens[None, ...],
            (batch, config.num_queries, config.hidden_dim),
        )
        q_weight, k_weight, v_weight = mx.split(self.in_proj_weight, 3, axis=0)
        q_bias, k_bias, v_bias = mx.split(self.in_proj_bias, 3, axis=0)
        query = mx.matmul(queries, q_weight.T) + q_bias
        key = mx.matmul(tokens, k_weight.T) + k_bias
        value = mx.matmul(tokens, v_weight.T) + v_bias
        head_dim = config.hidden_dim // config.num_heads
        query = query.reshape(batch, config.num_queries, config.num_heads, head_dim).transpose(
            0, 2, 1, 3
        )
        key = key.reshape(batch, -1, config.num_heads, head_dim).transpose(0, 2, 1, 3)
        value = value.reshape(batch, -1, config.num_heads, head_dim).transpose(0, 2, 1, 3)
        pooled = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=head_dim**-0.5,
        )
        pooled = pooled.transpose(0, 2, 1, 3).reshape(
            batch,
            config.num_queries,
            config.hidden_dim,
        )
        return self.out_proj(pooled)

    def __call__(
        self,
        video_tokens: mx.array | None = None,
        audio_tokens: mx.array | None = None,
    ) -> mx.array:
        if video_tokens is None and audio_tokens is None:
            raise ValueError("duration head requires at least one connector token stream")
        groups = []
        batch = None
        if video_tokens is not None:
            if (
                video_tokens.ndim != 3
                or video_tokens.shape[-1] != self.architecture.video_context_dim
            ):
                raise ValueError(
                    "video duration tokens must have shape (batch, tokens, "
                    f"{self.architecture.video_context_dim})"
                )
            batch = video_tokens.shape[0]
            groups.append(
                self.video_input_proj(video_tokens.astype(self.compute_dtype))
                + self.video_modality_emb
            )
        if audio_tokens is not None:
            if (
                audio_tokens.ndim != 3
                or audio_tokens.shape[-1] != self.architecture.audio_context_dim
            ):
                raise ValueError(
                    "audio duration tokens must have shape (batch, tokens, "
                    f"{self.architecture.audio_context_dim})"
                )
            if batch is not None and audio_tokens.shape[0] != batch:
                raise ValueError("duration token streams must have the same batch size")
            groups.append(
                self.audio_input_proj(audio_tokens.astype(self.compute_dtype))
                + self.audio_modality_emb
            )
        pooled = self._pool(mx.concatenate(groups, axis=1)).reshape(groups[0].shape[0], -1)
        hidden = gelu_approx(self.mlp_hidden(pooled))
        return mx.exp(self.mlp_out(hidden)[..., 0])

    def predict_num_frames(
        self,
        video_tokens: mx.array | None = None,
        audio_tokens: mx.array | None = None,
        *,
        frame_rate: float,
        temporal_compression_ratio: int,
        min_seconds: float = 1.0,
        max_seconds: float = 20.0,
    ) -> int:
        prediction = self(video_tokens, audio_tokens)
        if prediction.size != 1:
            raise ValueError(
                "duration frame prediction requires exactly one prompt, got "
                f"{tuple(prediction.shape)}"
            )
        mx.eval(prediction)
        seconds = float(cast(int | float, prediction.item()))
        frames = snap_duration_to_frame_count(
            seconds,
            frame_rate=frame_rate,
            temporal_compression_ratio=temporal_compression_ratio,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
        )
        if seconds < min_seconds or seconds > max_seconds:
            _log.warning(
                "Duration prediction %.2fs was clamped to %.2fs (%d frames at %.2f fps)",
                seconds,
                frames / frame_rate,
                frames,
                frame_rate,
            )
        else:
            _log.info(
                "Predicted duration %.2fs (%d frames at %.2f fps)",
                seconds,
                frames,
                frame_rate,
            )
        return frames


_DURATION_TARGETS = (
    "attention_pooler.cross_attn.in_proj_bias",
    "attention_pooler.cross_attn.in_proj_weight",
    "attention_pooler.cross_attn.out_proj.bias",
    "attention_pooler.cross_attn.out_proj.weight",
    "attention_pooler.query_tokens",
    "audio_input_proj.bias",
    "audio_input_proj.weight",
    "audio_modality_emb",
    "mlp_hidden.bias",
    "mlp_hidden.weight",
    "mlp_out.bias",
    "mlp_out.weight",
    "video_input_proj.bias",
    "video_input_proj.weight",
    "video_modality_emb",
)


def _duration_parameter(model: DurationHead, key: str) -> tuple[object, str, mx.array]:
    aliases: dict[str, tuple[object, str]] = {
        "attention_pooler.cross_attn.in_proj_bias": (model, "in_proj_bias"),
        "attention_pooler.cross_attn.in_proj_weight": (model, "in_proj_weight"),
        "attention_pooler.cross_attn.out_proj.bias": (model.out_proj, "bias"),
        "attention_pooler.cross_attn.out_proj.weight": (model.out_proj, "weight"),
        "attention_pooler.query_tokens": (model, "query_tokens"),
        "audio_input_proj.bias": (model.audio_input_proj, "bias"),
        "audio_input_proj.weight": (model.audio_input_proj, "weight"),
        "audio_modality_emb": (model, "audio_modality_emb"),
        "mlp_hidden.bias": (model.mlp_hidden, "bias"),
        "mlp_hidden.weight": (model.mlp_hidden, "weight"),
        "mlp_out.bias": (model.mlp_out, "bias"),
        "mlp_out.weight": (model.mlp_out, "weight"),
        "video_input_proj.bias": (model.video_input_proj, "bias"),
        "video_input_proj.weight": (model.video_input_proj, "weight"),
        "video_modality_emb": (model, "video_modality_emb"),
    }
    owner, name = aliases[key]
    return owner, name, getattr(owner, name)


def _source_key(weights: dict[str, mx.array], logical: str) -> str | None:
    direct = f"duration_head.{logical}"
    if direct in weights:
        return direct
    matches = sorted(name for name in weights if name.endswith(f".{direct}"))
    return None if not matches else matches[0]


def load_duration_head_weights(
    model: DurationHead,
    path: Path | str,
    *,
    reporter: Reporter | None = None,
) -> int:
    """Preflight and bind all 15 execution-changing duration tensors."""
    sink = reporter if reporter is not None else NullReporter()
    phase = "load LTX duration head"
    sink.phase_start(phase, total=len(_DURATION_TARGETS), unit="tensor")
    weights: dict[str, mx.array] = {}
    try:
        weights = load_weights(path)
        bindings = {
            logical: source
            for logical in _DURATION_TARGETS
            if (source := _source_key(weights, logical)) is not None
        }
        missing = sorted(set(_DURATION_TARGETS) - bindings.keys())
        if missing:
            raise ValueError(
                "unsupported duration head checkpoint: "
                f"missing {len(missing)} consumed tensors (first: {missing[0]})"
            )
        prepared = []
        for logical in _DURATION_TARGETS:
            owner, name, target = _duration_parameter(model, logical)
            value = weights[bindings[logical]]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"duration head tensor {logical!r} has shape {tuple(value.shape)}, "
                    f"expected {tuple(target.shape)}"
                )
            prepared.append((owner, name, value))
        for owner, name, value in prepared:
            setattr(owner, name, value)
            sink.phase_advance(phase)
        mx.eval(model.parameters())
        return len(prepared)
    finally:
        weights.clear()
        sink.phase_end(phase)


def load_duration_head(
    path: Path | str,
    *,
    compute_dtype: mx.Dtype = mx.bfloat16,
    reporter: Reporter | None = None,
) -> DurationHead:
    model = DurationHead(
        DurationHeadArchitecture.from_checkpoint(path),
        compute_dtype=compute_dtype,
    )
    load_duration_head_weights(model, path, reporter=reporter)
    return model


__all__ = [
    "DurationHead",
    "DurationHeadArchitecture",
    "load_duration_head",
    "load_duration_head_weights",
    "snap_duration_to_frame_count",
]
