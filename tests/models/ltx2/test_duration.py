from __future__ import annotations

import json

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.duration import (
    DurationHead,
    DurationHeadArchitecture,
    load_duration_head_weights,
    snap_duration_to_frame_count,
)


def _small_head() -> DurationHead:
    return DurationHead(
        DurationHeadArchitecture(
            video_context_dim=4,
            audio_context_dim=3,
            hidden_dim=4,
            num_queries=1,
            num_heads=2,
            mlp_hidden_dim=4,
        ),
        compute_dtype=mx.float32,
    )


def _checkpoint_weights(model: DurationHead) -> dict[str, mx.array]:
    return {
        "duration_head.attention_pooler.cross_attn.in_proj_bias": model.in_proj_bias,
        "duration_head.attention_pooler.cross_attn.in_proj_weight": model.in_proj_weight,
        "duration_head.attention_pooler.cross_attn.out_proj.bias": model.out_proj.bias,
        "duration_head.attention_pooler.cross_attn.out_proj.weight": model.out_proj.weight,
        "duration_head.attention_pooler.query_tokens": model.query_tokens,
        "duration_head.audio_input_proj.bias": model.audio_input_proj.bias,
        "duration_head.audio_input_proj.weight": model.audio_input_proj.weight,
        "duration_head.audio_modality_emb": model.audio_modality_emb,
        "duration_head.mlp_hidden.bias": model.mlp_hidden.bias,
        "duration_head.mlp_hidden.weight": model.mlp_hidden.weight,
        "duration_head.mlp_out.bias": model.mlp_out.bias,
        "duration_head.mlp_out.weight": model.mlp_out.weight,
        "duration_head.video_input_proj.bias": model.video_input_proj.bias,
        "duration_head.video_input_proj.weight": model.video_input_proj.weight,
        "duration_head.video_modality_emb": model.video_modality_emb,
    }


@pytest.mark.parametrize(
    ("seconds", "minimum", "maximum", "expected"),
    [
        (0.25, 1.0, 20.0, 25),
        (2.0, 1.0, 20.0, 41),
        (30.0, 1.0, 20.0, 473),
        (1.0, 1.0, 1.02, 25),
    ],
)
def test_duration_snap_clamps_then_uses_causal_grid(
    seconds: float,
    minimum: float,
    maximum: float,
    expected: int,
) -> None:
    assert (
        snap_duration_to_frame_count(
            seconds,
            frame_rate=24.0,
            temporal_compression_ratio=8,
            min_seconds=minimum,
            max_seconds=maximum,
        )
        == expected
    )


def test_duration_head_accepts_either_or_both_connector_streams() -> None:
    model = _small_head()
    video = mx.zeros((1, 3, 4), dtype=mx.float32)
    audio = mx.zeros((1, 2, 3), dtype=mx.float32)

    video_seconds = model(video_tokens=video)
    audio_seconds = model(audio_tokens=audio)
    both_seconds = model(video_tokens=video, audio_tokens=audio)
    mx.eval(video_seconds, audio_seconds, both_seconds)

    assert tuple(video_seconds.shape) == (1,)
    assert tuple(audio_seconds.shape) == (1,)
    assert tuple(both_seconds.shape) == (1,)
    assert mx.all(mx.isfinite(both_seconds)).item()
    with pytest.raises(ValueError, match="at least one"):
        model()


def test_duration_loader_consumes_all_15_targets_and_preflights_shapes(tmp_path) -> None:
    source = _small_head()
    weights = _checkpoint_weights(source)
    weights["community.wrapper.unused"] = mx.zeros((1,))
    path = tmp_path / "duration.safetensors"
    mx.save_safetensors(
        str(path),
        weights,
        metadata={
            "config": json.dumps(
                {
                    "transformer": {
                        "cross_attention_dim": 4,
                        "audio_cross_attention_dim": 3,
                    },
                    "duration_head": {},
                }
            )
        },
    )

    target = _small_head()
    assert load_duration_head_weights(target, path) == 15
    broken = dict(weights)
    broken["duration_head.mlp_out.weight"] = mx.zeros((2, 2))
    broken_path = tmp_path / "broken.safetensors"
    mx.save_safetensors(str(broken_path), broken)
    original = target.video_input_proj.weight
    with pytest.raises(ValueError, match="mlp_out.weight.*shape"):
        load_duration_head_weights(target, broken_path)
    assert target.video_input_proj.weight is original
