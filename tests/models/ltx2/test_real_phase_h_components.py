"""Explicit opt-in real Phase H duration and temporal component oracles."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from kinomlx.cli.args import build_parser
from kinomlx.cli.config import assemble
from kinomlx.models.ltx2.duration import load_duration_head
from kinomlx.models.ltx2.resources import prepare_resources
from kinomlx.models.ltx2.upscaler.temporal import load_temporal_upscaler

_FIXTURE = Path(__file__).parent / "fixtures/ltx25_phase_h_component_oracles.json"


def _sha256(path: Path) -> str:
    resolved = path.resolve()
    if len(resolved.name) == 64 and all(char in "0123456789abcdef" for char in resolved.name):
        return resolved.name
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _release_mlx() -> None:
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


@pytest.mark.requires_weights
@pytest.mark.requires_metal
def test_real_phase_h_component_oracles() -> None:
    if os.environ.get("KINO_RUN_REAL_LTX25_PHASE_H") != "1":
        pytest.skip("set KINO_RUN_REAL_LTX25_PHASE_H=1 for the real Phase H component gate")

    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    options = build_parser().parse_args(["--ltx-generation", "2.5", "--print-config"])
    invocation = assemble(options)
    resources = prepare_resources(
        invocation.model_settings,
        infrastructure=invocation.settings,
    )

    duration_fixture = fixture["duration"]
    duration_path = resources.duration_head_path
    assert _sha256(duration_path) == duration_fixture["artifact_sha256"]
    video = np.sin(
        np.arange(math.prod(duration_fixture["video_shape"]), dtype=np.float32) * np.float32(0.007)
    ).reshape(duration_fixture["video_shape"])
    audio = np.cos(
        np.arange(math.prod(duration_fixture["audio_shape"]), dtype=np.float32) * np.float32(0.011)
    ).reshape(duration_fixture["audio_shape"])
    duration = load_duration_head(duration_path)
    video_tokens = mx.array(video).astype(mx.bfloat16)
    audio_tokens = mx.array(audio).astype(mx.bfloat16)
    seconds = duration(video_tokens, audio_tokens)
    mx.eval(seconds)
    assert float(seconds.item()) == pytest.approx(
        duration_fixture["torch_bfloat16_seconds"],
        abs=1e-6,
    )
    assert (
        duration.predict_num_frames(
            video_tokens,
            audio_tokens,
            frame_rate=24.0,
            temporal_compression_ratio=8,
        )
        == duration_fixture["frames_at_24_fps"]
    )
    del duration, seconds, video_tokens, audio_tokens
    _release_mlx()

    temporal_fixture = fixture["temporal_upscaler"]
    temporal_path = resources.temporal_upscaler_path
    assert _sha256(temporal_path) == temporal_fixture["artifact_sha256"]
    latent = np.sin(
        np.arange(math.prod(temporal_fixture["input_shape"]), dtype=np.float32) * np.float32(0.017)
    ).reshape(temporal_fixture["input_shape"])
    temporal = load_temporal_upscaler(temporal_path)
    result = temporal(mx.array(latent).astype(mx.bfloat16)).astype(mx.float32)
    mx.eval(result)
    assert list(result.shape) == temporal_fixture["output_shape"]
    assert mx.all(mx.isfinite(result)).item()
    flattened = np.asarray(result).reshape(-1)
    actual_anchors = flattened[temporal_fixture["anchor_indices"]]
    np.testing.assert_allclose(
        actual_anchors,
        temporal_fixture["torch_bfloat16_anchors"],
        rtol=0.0,
        atol=temporal_fixture["anchor_atol"],
    )
    assert float(np.mean(flattened)) == pytest.approx(
        temporal_fixture["torch_bfloat16_mean"],
        abs=0.01,
    )
    assert float(np.max(np.abs(flattened))) == pytest.approx(
        temporal_fixture["torch_bfloat16_peak"],
        abs=0.1,
    )
    assert float(np.sqrt(np.mean(np.square(flattened)))) == pytest.approx(
        temporal_fixture["torch_bfloat16_rms"],
        abs=0.02,
    )
    del temporal, result
    _release_mlx()
