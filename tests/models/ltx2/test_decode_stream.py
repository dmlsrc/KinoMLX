"""Lazy LTX-2.3 VAE frame-boundary tests."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

import kinomlx.models.ltx2.decode as decode
from kinomlx.components import ComponentLease
from kinomlx.media.signals import UnsupportedSignalError
from kinomlx.models.ltx2.signals import (
    ACESCCT_WORKING_SIGNAL,
    SCENE_LINEAR_HDR_SIGNAL,
    ltx23_sdr_signal,
)
from kinomlx.reporting import RecordingReporter
from kinomlx.samplers.noise import NoiseStreamState


def _decoded_chunk() -> mx.array:
    return mx.array(
        [
            [
                [[[-2.0, -1.0], [0.0, 2.0]], [[-0.5, 0.0], [0.5, 1.0]]],
                [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.5], [0.0, -0.5]]],
                [[[2.0, 1.0], [-1.0, -2.0]], [[0.25, -0.25], [0.75, -0.75]]],
            ]
        ],
        dtype=mx.float32,
    )


@pytest.mark.parametrize("signal", [ACESCCT_WORKING_SIGNAL, SCENE_LINEAR_HDR_SIGNAL])
def test_ltx23_sdr_decode_rejects_non_sdr_codes_before_opening_decoder(signal) -> None:
    loads = []

    with pytest.raises(UnsupportedSignalError, match="LTX-2.3 SDR decode cannot produce"):
        decode.decode_ltx23_sdr_frames(
            mx.zeros((1, 128, 1, 1, 1)),
            lambda: loads.append("decoder") or ComponentLease(SimpleNamespace()),
            spec=signal,
            frame_count=1,
        )

    assert loads == []


def test_ltx23_stream_opens_decoder_on_first_pull_and_converts_one_frame(
    monkeypatch,
) -> None:
    events = []
    reporter = RecordingReporter()
    chunk = _decoded_chunk()

    def provider():
        events.append("load")
        return ComponentLease(
            SimpleNamespace(),
            close_component=lambda _decoder: events.append("close"),
        )

    def streaming(_latent, _decoder, **_kwargs):
        events.append("decode")
        yield chunk

    monkeypatch.setattr(decode, "decode_streaming", streaming)
    stream = decode.decode_ltx23_sdr_frames(
        mx.zeros((1, 128, 2, 1, 1)),
        provider,
        spec=ltx23_sdr_signal(width=2, height=2, fps=24.0),
        frame_count=2,
        tiling_config=decode.TilingConfig(),
        reporter=reporter,
    )

    assert events == []
    first = next(stream)
    assert events == ["load", "decode"]
    assert first.shape == (2, 2, 3)
    assert first.dtype == mx.float16
    assert mx.all(first >= 0).item()
    assert mx.all(first <= 1).item()

    second = next(stream)
    with pytest.raises(StopIteration):
        next(stream)
    assert second.dtype == mx.float16
    assert events == ["load", "decode", "close"]
    # The frame stream does not add a second progress phase over the decoder's
    # tile phase. Frame consumption belongs to the terminal sink.
    assert reporter.events == []


def test_early_close_releases_decoder_and_streaming_generator(monkeypatch) -> None:
    events = []

    def provider():
        events.append("load")
        return ComponentLease(
            SimpleNamespace(),
            close_component=lambda _decoder: events.append("decoder-close"),
        )

    def streaming(_latent, _decoder, **_kwargs):
        try:
            yield _decoded_chunk()
        finally:
            events.append("generator-close")

    monkeypatch.setattr(decode, "decode_streaming", streaming)
    stream = decode.decode_ltx23_sdr_frames(
        mx.zeros((1, 128, 2, 1, 1)),
        provider,
        spec=ltx23_sdr_signal(width=2, height=2, fps=24.0),
        frame_count=2,
    )

    next(stream)
    stream.close()

    assert events == ["load", "generator-close", "decoder-close"]


def test_receipt_uses_available_at_entry_memory_for_auto_plan(monkeypatch) -> None:
    receipt = decode.VAEEntryMemoryReceipt(
        active_bytes=3_000_000_000,
        cache_bytes=1_000_000_000,
        peak_bytes=4_000_000_000,
        total_bytes=32_000_000_000,
        recommended_working_set_bytes=28_000_000_000,
        available_bytes=24_000_000_000,
        planner_budget_bytes=12_000_000_000,
    )
    seen = []
    monkeypatch.setattr(
        decode,
        "capture_vae_entry_memory_receipt",
        lambda **_kwargs: receipt,
    )
    monkeypatch.setattr(
        decode.TilingConfig,
        "auto_native_conv3d",
        lambda height, width, frames, *, memory_budget_gb, compute_dtype: (
            seen.append((height, width, frames, memory_budget_gb, compute_dtype))
            or SimpleNamespace()
        ),
    )
    monkeypatch.setattr(
        decode,
        "decode_streaming",
        lambda *_args, **_kwargs: iter((mx.zeros((1, 3, 1, 2, 2)),)),
    )
    stream = decode.decode_ltx23_sdr_frames(
        mx.zeros((1, 128, 1, 1, 1)),
        lambda: ComponentLease(SimpleNamespace()),
        spec=ltx23_sdr_signal(width=2, height=2, fps=24.0),
        frame_count=1,
        auto_tiling=True,
    )

    assert len(list(stream)) == 1
    assert seen == [(2, 2, 1, 12.0, mx.bfloat16)]
    assert stream.receipts["vae_entry"] is receipt
    assert stream.receipts["vae_tiling"] is not None
    assert receipt.live_assets == frozenset({"final_video_latent", "video_decoder"})


def test_auto_full_fit_bypasses_the_legacy_default_budget_replan(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(
        decode.TilingConfig,
        "auto_native_conv3d",
        lambda *_args, **_kwargs: None,
    )

    def streaming(_latent, _decoder, *, tiling_config, **_kwargs):
        seen.append(tiling_config)
        yield mx.zeros((1, 3, 1, 2, 2))

    monkeypatch.setattr(decode, "decode_streaming", streaming)
    stream = decode.decode_ltx23_sdr_frames(
        mx.zeros((1, 128, 1, 1, 1)),
        lambda: ComponentLease(SimpleNamespace()),
        spec=ltx23_sdr_signal(width=2, height=2, fps=24.0),
        frame_count=1,
        auto_tiling=True,
    )

    assert len(list(stream)) == 1
    assert len(seen) == 1
    assert isinstance(seen[0], decode.TilingConfig)
    assert seen[0].temporal_config is None
    assert seen[0].spatial_config is None
    assert stream.receipts["vae_tiling"] is seen[0]


def test_capture_receipt_subtracts_active_and_cache_from_recommended_limit(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        decode.mx,
        "device_info",
        lambda: {
            "memory_size": 32_000_000_000,
            "max_recommended_working_set_size": 28_000_000_000,
        },
    )
    monkeypatch.setattr(decode.mx, "get_active_memory", lambda: 3_000_000_000)
    monkeypatch.setattr(decode.mx, "get_cache_memory", lambda: 1_000_000_000)
    monkeypatch.setattr(decode.mx, "get_peak_memory", lambda: 5_000_000_000)

    latent = mx.zeros((1, 3), dtype=mx.float16)
    decoder = SimpleNamespace(parameters=lambda: {"weight": mx.zeros((4,), dtype=mx.float16)})
    receipt = decode.capture_vae_entry_memory_receipt(
        latent=latent,
        decoder=decoder,
    )

    assert receipt.available_bytes == 24_000_000_000
    assert receipt.planner_budget_bytes == 12_000_000_000
    assert receipt.active_bytes == 3_000_000_000
    assert receipt.cache_bytes == 1_000_000_000
    assert receipt.latent_bytes == 6
    assert receipt.decoder_parameter_bytes == 8
    assert receipt.accounted_asset_bytes == 14
    assert receipt.unaccounted_active_bytes == 3_000_000_000 - 14


def test_video_decode_diagnostics_serializes_memory_and_replayable_plan() -> None:
    entry = decode.VAEEntryMemoryReceipt(
        active_bytes=2_000,
        cache_bytes=300,
        peak_bytes=4_000,
        total_bytes=10_000,
        recommended_working_set_bytes=8_000,
        available_bytes=5_700,
        planner_budget_bytes=2_850,
        latent_bytes=20,
        decoder_parameter_bytes=1_500,
    )
    config = decode.TilingConfig.temporal_only(tile_size=32, overlap=8)
    plan = decode.TilingPlanReceipt(
        latent_shape=(16, 14, 24),
        decoded_shape=(121, 448, 768),
        temporal_tiles=5,
        spatial_height_tiles=1,
        spatial_width_tiles=1,
        total_tiles=5,
        resolved_config=config,
    )
    load_receipt = SimpleNamespace(
        to_dict=lambda: {
            "decoder_kind": "diffusion-na",
            "loaded_tensors": 311,
        }
    )
    stream = decode.VideoFrameStream(
        lambda: iter(()),
        spec=ltx23_sdr_signal(width=768, height=448, fps=24.0),
        frame_count=121,
        receipts={
            "vae_entry": entry,
            "vae_tiling_mode": "auto",
            "vae_tiling": config,
            "vae_plan": plan,
            "vae_decoder_seed": 17,
            "vae_noise_backend": "torch-mps",
            "vae_noise_compatibility_profile": "pytorch-2.13.0-mps",
            "vae_noise_state": NoiseStreamState(
                backend="torch-mps",
                compatibility_profile="pytorch-2.13.0-mps",
                seed=17,
                draws=5,
                elements=1_048_576,
                philox_blocks=262_144,
            ),
            "vae_load": load_receipt,
        },
    )

    assert decode.video_decode_diagnostics(stream) == {
        "vae_decode": {
            "entry_memory": {
                "active_bytes": 2_000,
                "cache_bytes": 300,
                "peak_bytes": 4_000,
                "total_bytes": 10_000,
                "recommended_working_set_bytes": 8_000,
                "available_bytes": 5_700,
                "planner_budget_bytes": 2_850,
                "latent_bytes": 20,
                "decoder_parameter_bytes": 1_500,
                "accounted_asset_bytes": 1_520,
                "unaccounted_active_bytes": 480,
                "live_assets": ["final_video_latent", "video_decoder"],
            },
            "tiling": {
                "requested_mode": "auto",
                "decoder_seed": 17,
                "latent_shape": [16, 14, 24],
                "decoded_shape": [121, 448, 768],
                "temporal_tiles": 5,
                "spatial_height_tiles": 1,
                "spatial_width_tiles": 1,
                "total_tiles": 5,
                "resolved_config": {
                    "temporal_tile_frames": 32,
                    "temporal_overlap_frames": 8,
                    "spatial_tile_pixels": None,
                    "spatial_overlap_pixels": None,
                },
            },
            "noise": {
                "backend": "torch-mps",
                "compatibility_profile": "pytorch-2.13.0-mps",
                "seed": 17,
                "draws": 5,
                "elements": 1_048_576,
                "philox_blocks": 262_144,
                "philox_block_width": 4,
            },
            "decoder_load": {
                "decoder_kind": "diffusion-na",
                "loaded_tensors": 311,
            },
        }
    }
