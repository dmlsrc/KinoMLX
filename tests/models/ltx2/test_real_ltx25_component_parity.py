"""Real LTX-2.5 component cache coverage and numerical reference parity."""

import gc
import json
import math
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from kinomlx.io.safetensors import read_header
from kinomlx.models.ltx2.audio_vae import (
    AudioDecoder,
    AudioEncoder,
    AudioVAEConfig,
    BWEVocoderConfig,
    VocoderWithBWE,
    load_audio_vae_weights,
    load_vocoder_weights,
)
from kinomlx.models.ltx2.preparation import family_source_preparation
from kinomlx.models.ltx2.upscaler import (
    SpatialUpscaler,
    SpatialUpscalerConfig,
    load_spatial_upscaler_weights,
)
from kinomlx.models.ltx2.video_vae.config import VideoVAEConfig
from kinomlx.models.ltx2.video_vae.loading import load_video_vae
from kinomlx.models.ltx2.weights.keys import weight_family_for_key
from tests._snapshots import pinned_snapshot
from tests.models.ltx2._source_weights import family_weights

_LTX25_REVISION = "6c7e5e573ac1667efc83407806fe9b0b93730e60"
_DIFFUSERS_COMMIT = "2f7e0154a9db246e95c9ede43edba7db5b130805"
_COVERAGE = {
    "video_vae": 170,
    "audio_vae": 102,
    "vocoder": 1227,
    "spatial_upscaler": 72,
}


def _snapshot() -> Path:
    return pinned_snapshot("Lightricks/LTX-2.5", _LTX25_REVISION)


def _paths() -> tuple[Path, Path, Path]:
    root = _snapshot()
    video = root / "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
    audio = root / "vae/ltx-2.5-audio-vae-bf16.safetensors"
    upscaler = (
        root / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    )
    missing = [path.name for path in (video, audio, upscaler) if not path.is_file()]
    if missing:
        pytest.skip(
            "pinned LTX-2.5 component parity artifacts are not cached: " + ", ".join(missing)
        )
    return video, audio, upscaler


def _required_directory(variable: str) -> Path:
    raw = os.environ.get(variable)
    if raw is None:
        pytest.skip(f"{variable} is not configured")
    path = Path(raw).expanduser()
    if not path.is_dir():
        pytest.skip(f"{variable} does not name a directory")
    return path


def _tensor_keys(path: Path) -> set[str]:
    return {key for key in read_header(path) if key != "__metadata__"}


def _component_source_keys(path: Path, source_component: str, family: str) -> set[str]:
    return {
        key
        for key in _tensor_keys(path)
        if weight_family_for_key(key, source_component=source_component) == family
    }


def _component_parity_weights():
    video, audio, _upscaler = _paths()
    return (
        family_weights(video, "video_vae"),
        family_weights(audio, "audio_vae"),
        family_weights(audio, "vocoder"),
    )


def _assert_reference_boundary(
    name: str,
    candidate: mx.array,
    reference: np.ndarray,
) -> dict[str, float]:
    actual = np.asarray(candidate.astype(mx.float32))
    expected = np.asarray(reference, dtype=np.float32)
    assert actual.shape == expected.shape, name
    assert np.isfinite(actual).all(), name
    assert np.isfinite(expected).all(), name
    delta = actual - expected
    max_abs = float(np.max(np.abs(delta)))
    reference_rms = float(np.sqrt(np.mean(np.square(expected, dtype=np.float64))))
    reference_max = float(np.max(np.abs(expected)))
    scale = max(reference_max, reference_rms, 1e-12)
    rms_floor = max(reference_rms, scale * (2.0**-12))
    normalized_rms = float(np.sqrt(np.mean(np.square(delta, dtype=np.float64))) / rms_floor)
    actual_flat = actual.reshape(-1).astype(np.float64)
    expected_flat = expected.reshape(-1).astype(np.float64)
    denominator = float(np.linalg.norm(actual_flat) * np.linalg.norm(expected_flat))
    cosine = (
        1.0
        if denominator == 0.0 and max_abs == 0.0
        else float(np.dot(actual_flat, expected_flat) / denominator)
    )
    epsilon_limit = 4.0 * (2.0**-7)
    assert max_abs <= scale * epsilon_limit, (
        f"{name}: max_abs={max_abs}, limit={scale * epsilon_limit}"
    )
    assert normalized_rms <= epsilon_limit, (
        f"{name}: normalized_rms={normalized_rms}, limit={epsilon_limit}"
    )
    assert cosine >= 1.0 - 0.5 * epsilon_limit**2, (
        f"{name}: cosine={cosine}, minimum={1.0 - 0.5 * epsilon_limit**2}"
    )
    return {
        "max_abs": max_abs,
        "normalized_rms": normalized_rms,
        "cosine": cosine,
    }


def _release() -> None:
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


def _consumed_source_keys(path: Path, family: str, source_component: str) -> set[str]:
    """The source tensors one family's preparation reads from a checkpoint file."""
    preparation = family_source_preparation(path, family=family, source_component=source_component)
    return {item.source_key for item in preparation.sources}


def test_real_ltx25_component_local_family_weights_coverage() -> None:
    video, audio, _upscaler = _paths()

    # Every source tensor the header attributes to a family is one the family's
    # preparation consumes, and the loaded canonical tensors are the counts the
    # component contract pins.
    assert _consumed_source_keys(video, "video_vae", "video_vae") == _component_source_keys(
        video, "video_vae", "video_vae"
    )
    assert _consumed_source_keys(audio, "audio_vae", "audio_vae_vocoder") == (
        _component_source_keys(audio, "audio_vae_vocoder", "audio_vae")
    )
    assert _consumed_source_keys(audio, "vocoder", "audio_vae_vocoder") == (
        _component_source_keys(audio, "audio_vae_vocoder", "vocoder")
    )
    video_weights, audio_weights, vocoder_weights = _component_parity_weights()
    assert len(video_weights) == _COVERAGE["video_vae"]
    assert len(audio_weights) == _COVERAGE["audio_vae"]
    assert len(vocoder_weights) == _COVERAGE["vocoder"]


@pytest.mark.heavy
def test_real_ltx25_component_parity_reference_boundaries() -> None:
    fixture_root = _required_directory("KINO_LTX25_COMPONENT_PARITY_FIXTURE_DIR")
    manifest = json.loads((fixture_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["ltx25_revision"] == _LTX25_REVISION
    assert manifest["diffusers_commit"] == _DIFFUSERS_COMMIT
    assert manifest["coverage"] == _COVERAGE
    fixture = np.load(fixture_root / manifest["fixture"])
    video_path, audio_path, upscaler_path = _paths()
    video_weights, audio_cache, vocoder_cache = _component_parity_weights()

    bundle = None
    video_encoded = video_decoded = None
    try:
        bundle = load_video_vae(
            video_weights,
            config=VideoVAEConfig.from_checkpoint(video_path),
        )
        video_encoded = bundle.encoder(mx.array(fixture["video_input"]).astype(mx.bfloat16))
        video_decoded = bundle.decoder(mx.array(fixture["video_latent_input"]).astype(mx.bfloat16))
        mx.eval(video_encoded, video_decoded)
        _assert_reference_boundary("video_encoded", video_encoded, fixture["video_encoded"])
        _assert_reference_boundary("video_decoded", video_decoded, fixture["video_decoded"])
    finally:
        del bundle, video_encoded, video_decoded
        _release()

    encoder = decoder = audio_decoded = None
    try:
        audio_config = AudioVAEConfig.from_checkpoint(audio_path)
        encoder = AudioEncoder(audio_config)
        decoder = AudioDecoder(audio_config)
        assert load_audio_vae_weights(encoder, decoder, audio_cache) == _COVERAGE["audio_vae"]
        audio_decoded = decoder(mx.array(fixture["audio_latent_input"]).astype(mx.bfloat16))
        mx.eval(audio_decoded)
        _assert_reference_boundary("audio_decoded", audio_decoded, fixture["audio_decoded"])
    finally:
        del encoder, decoder, audio_decoded
        _release()

    vocoder = waveform = None
    try:
        vocoder = VocoderWithBWE(BWEVocoderConfig.from_checkpoint(audio_path))
        assert load_vocoder_weights(vocoder, vocoder_cache) == _COVERAGE["vocoder"]
        waveform = vocoder(mx.array(fixture["vocoder_mel_input"]).astype(mx.bfloat16))
        mx.eval(waveform)
        assert tuple(waveform.shape) == (1, 2, 480)
        assert vocoder.output_sample_rate == 48_000
        assert math.isfinite(float(mx.max(mx.abs(waveform)).item()))
        assert mx.any(waveform != 0).item()
        _assert_reference_boundary("vocoder_waveform", waveform, fixture["vocoder_waveform"])
    finally:
        del vocoder, waveform
        _release()

    upscaler = upscaled = None
    try:
        upscaler = SpatialUpscaler(SpatialUpscalerConfig.from_checkpoint(upscaler_path))
        assert (
            load_spatial_upscaler_weights(upscaler, upscaler_path) == _COVERAGE["spatial_upscaler"]
        )
        upscaled = upscaler(mx.array(fixture["upscaler_input"]).astype(mx.bfloat16))
        mx.eval(upscaled)
        assert mx.any(upscaled != 0).item()
        _assert_reference_boundary("upscaler_output", upscaled, fixture["upscaler_output"])
    finally:
        del upscaler, upscaled
        _release()
