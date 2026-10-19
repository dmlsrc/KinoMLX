"""Decoded-media boundary contracts."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

import kinomlx.models.ltx2.decode as decode
from kinomlx.components import ComponentLease
from kinomlx.models.ltx2.signals import ltx23_sdr_signal
from kinomlx.models.ltx2.video_vae.kinds import VideoVAEKind


def _entry_receipt() -> decode.VAEEntryMemoryReceipt:
    return decode.VAEEntryMemoryReceipt(
        active_bytes=1_000,
        cache_bytes=0,
        peak_bytes=1_000,
        total_bytes=40_000_000_000,
        recommended_working_set_bytes=40_000_000_000,
        available_bytes=31_430_000_000,
        planner_budget_bytes=31_430_000_000,
    )


def test_decode_video_passes_tiling_config_by_keyword(monkeypatch) -> None:
    tiling = SimpleNamespace(name="tiling")
    reporter = SimpleNamespace(name="reporter")
    expected = mx.zeros((1, 3, 1, 2, 2))
    calls = []

    def fake_decode_streaming(
        latent,
        decoder,
        *,
        tiling_config,
        reporter,
    ):
        calls.append((latent, decoder, tiling_config, reporter))
        yield expected

    monkeypatch.setattr(decode, "decode_streaming", fake_decode_streaming)
    latent = mx.zeros((1, 128, 1, 1, 1))
    decoder = object()

    actual = decode.decode_video(
        latent,
        decoder,
        tiling_config=tiling,
        reporter=reporter,
    )

    assert actual is expected
    assert calls == [(latent, decoder, tiling, reporter)]


def test_video_frames_preserves_normalized_sdr_values_as_float16() -> None:
    ramp = mx.array([-1.0, 0.0, 1.0]).reshape(1, 1, 1, 1, 3)
    video = mx.broadcast_to(ramp, (1, 3, 1, 1, 3))

    frame = next(decode.video_frames(video))

    assert frame.dtype == mx.float16
    assert mx.array_equal(
        frame[0, :, 0],
        mx.array([0.0, 0.5, 1.0], dtype=mx.float16),
    ).item()


def test_video_frames_widens_bfloat16_before_normalized_sdr_conversion() -> None:
    values = mx.array([-0.99609375, 0.5, 0.99609375], dtype=mx.bfloat16)
    video = mx.broadcast_to(values.reshape(1, 1, 1, 1, 3), (1, 3, 1, 1, 3))

    frame = next(decode.video_frames(video))
    expected = (
        mx.clip(
            (video[0].astype(mx.float32) + 1.0) * 0.5,
            0.0,
            1.0,
        )
        .transpose(1, 2, 3, 0)
        .astype(mx.float16)[0]
    )

    assert mx.array_equal(frame, expected).item()


def test_auto_tiler_receives_loaded_decoder_compute_dtype(monkeypatch) -> None:
    received = []
    decoder = SimpleNamespace(compute_dtype=mx.float32)

    def auto_conv3d(height, width, frames, *, memory_budget_gb, compute_dtype, optimized_fp16):
        received.append((height, width, frames, memory_budget_gb, compute_dtype, optimized_fp16))
        return decode.TilingConfig()

    def fake_decode_streaming(_latent, _decoder, **_kwargs):
        yield mx.zeros((1, 3, 1, 64, 64), dtype=mx.float32)

    monkeypatch.setattr(
        decode.TilingConfig,
        "auto_conv3d",
        staticmethod(auto_conv3d),
    )
    monkeypatch.setattr(decode, "decode_streaming", fake_decode_streaming)
    monkeypatch.setattr(
        decode,
        "capture_vae_entry_memory_receipt",
        lambda **_kwargs: _entry_receipt(),
    )
    stream = decode.decode_ltx23_sdr_frames(
        mx.zeros((1, 128, 1, 2, 2), dtype=mx.bfloat16),
        lambda: ComponentLease(decoder),
        spec=ltx23_sdr_signal(width=64, height=64, fps=24.0),
        frame_count=1,
        auto_tiling=True,
    )

    assert len(list(stream)) == 1
    assert len(received) == 3
    assert received[0] == (
        64,
        64,
        1,
        pytest.approx((31_430_000_000 - 5 * 1024**3) / 1e9),
        mx.float32,
        False,
    )
    assert stream.receipts["vae_cache"]["selected_limit_gb"] == 4.0


def test_diffusion_decode_receives_selected_noise_backend(monkeypatch) -> None:
    received = []
    decoder = SimpleNamespace(decoder_kind=VideoVAEKind.DIFFUSION)

    def fake_decode_streaming(_latent, _decoder, **kwargs):
        received.append(kwargs["noise_backend"])
        yield mx.zeros((1, 3, 1, 64, 64), dtype=mx.float32)

    monkeypatch.setattr(decode, "decode_streaming", fake_decode_streaming)
    monkeypatch.setattr(
        decode,
        "capture_vae_entry_memory_receipt",
        lambda **_kwargs: _entry_receipt(),
    )
    stream = decode.decode_ltx23_sdr_frames(
        mx.zeros((1, 128, 1, 2, 2), dtype=mx.bfloat16),
        lambda: ComponentLease(decoder),
        spec=ltx23_sdr_signal(width=64, height=64, fps=24.0),
        frame_count=1,
        noise_backend="torch-mps",
    )

    assert len(list(stream)) == 1
    assert received == ["torch-mps"]
    assert stream.receipts["vae_noise_backend"] == "torch-mps"
    assert stream.receipts["vae_noise_compatibility_profile"] == "pytorch-2.13.0-mps"
