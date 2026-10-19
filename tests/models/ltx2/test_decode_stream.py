"""Lazy LTX-2.3 VAE frame-boundary tests."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

import kinomlx.modeling.memory as memory
import kinomlx.models.ltx2.decode as decode
from kinomlx.components import ComponentLease
from kinomlx.media.signals import UnsupportedSignalError
from kinomlx.models.ltx2.signals import (
    ACESCCT_WORKING_SIGNAL,
    SCENE_LINEAR_HDR_SIGNAL,
    ltx23_sdr_signal,
)
from kinomlx.models.ltx2.video_vae.kinds import VideoVAEKind
from kinomlx.models.ltx2.video_vae.sharpening import ContextSharpening
from kinomlx.models.ltx2.video_vae.tiling import TilingConfig
from kinomlx.samplers.noise import NoiseStreamState
from tests._reporting import RecordingReporter


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
    clears = []
    monkeypatch.setattr(decode.mx, "clear_cache", lambda: clears.append(True))
    receipt = decode.VAEEntryMemoryReceipt(
        active_bytes=3_000_000_000,
        cache_bytes=1_000_000_000,
        peak_bytes=4_000_000_000,
        total_bytes=32_000_000_000,
        recommended_working_set_bytes=28_000_000_000,
        available_bytes=24_000_000_000,
        planner_budget_bytes=24_000_000_000,
    )
    seen = []
    monkeypatch.setattr(
        decode,
        "capture_vae_entry_memory_receipt",
        lambda **_kwargs: receipt,
    )
    monkeypatch.setattr(
        decode.TilingConfig,
        "auto_conv3d",
        lambda height, width, frames, *, memory_budget_gb, compute_dtype, optimized_fp16: (
            seen.append((height, width, frames, memory_budget_gb, compute_dtype, optimized_fp16))
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
        lambda: ComponentLease(SimpleNamespace(compute_dtype=mx.float16, causal=False)),
        spec=ltx23_sdr_signal(width=2, height=2, fps=24.0),
        frame_count=1,
        auto_tiling=True,
    )

    assert len(list(stream)) == 1
    assert len(seen) == 6
    assert all(item[:3] == (2, 2, 1) and item[4:] == (mx.float16, True) for item in seen)
    assert seen[0][3] == pytest.approx((24_000_000_000 - 17 * 1024**3) / 1e9)
    assert stream.receipts["vae_entry"].planner_budget_gb == seen[0][3]
    assert stream.receipts["vae_cache"]["selected_limit_gb"] == 16.0
    assert stream.receipts["vae_tiling"] is not None
    assert receipt.live_assets == frozenset({"final_video_latent", "video_decoder"})
    assert clears == []


def test_auto_cache_preserves_the_faster_temporal_tile(monkeypatch) -> None:
    def plan(_height, _width, _frames, *, memory_budget_gb, **_kwargs):
        size = 128 if memory_budget_gb >= 16 else 64
        return decode.TilingConfig.temporal_only(size, overlap=8)

    monkeypatch.setattr(decode.TilingConfig, "auto_conv3d", plan)
    common = {
        "available_bytes": 30_000_000_000,
        "auto_tiling": True,
        "tiling_config": None,
        "decoder_kind": VideoVAEKind.CONV,
        "compute_dtype": mx.float16,
        "optimized_fp16": True,
        "height": 448,
        "width": 768,
        "frames": 721,
    }

    cache_gb, _budget, tiling = decode._plan_vae_cache_and_tiling(**common, requested_cache_gb=None)
    assert cache_gb == 12.0
    assert tiling.temporal_config.chunk_size_in_frames == 128

    cache_gb, _budget, tiling = decode._plan_vae_cache_and_tiling(**common, requested_cache_gb=16.0)
    assert cache_gb == 16.0
    assert tiling.temporal_config.chunk_size_in_frames == 64


def test_custom_temporal_tile_reserves_its_estimated_working_set() -> None:
    cache_gb, _budget, tiling = decode._plan_vae_cache_and_tiling(
        available_bytes=30_000_000_000,
        requested_cache_gb=None,
        auto_tiling=False,
        tiling_config=decode.TilingConfig.temporal_only(256, overlap=8),
        decoder_kind=VideoVAEKind.CONV,
        compute_dtype=mx.float16,
        optimized_fp16=True,
        height=448,
        width=768,
        frames=721,
    )
    assert cache_gb == 1.0
    assert tiling.temporal_config.chunk_size_in_frames == 256


def test_vae_cache_override_restores_prior_limit_on_early_close(monkeypatch) -> None:
    receipt = decode.VAEEntryMemoryReceipt(
        active_bytes=1_000,
        cache_bytes=0,
        peak_bytes=1_000,
        total_bytes=50_000_000_000,
        recommended_working_set_bytes=50_000_000_000,
        available_bytes=40_000_000_000,
        planner_budget_bytes=40_000_000_000,
    )
    monkeypatch.setattr(decode, "capture_vae_entry_memory_receipt", lambda **_kwargs: receipt)
    calls = []

    def set_cache_limit(value):
        calls.append(value)
        return 1234

    monkeypatch.setattr(decode.mx, "set_cache_limit", set_cache_limit)
    monkeypatch.setattr(
        decode,
        "decode_streaming",
        lambda *_args, **_kwargs: iter((mx.zeros((1, 3, 2, 2, 2)),)),
    )
    stream = decode.decode_ltx23_sdr_frames(
        mx.zeros((1, 128, 1, 1, 1)),
        lambda: ComponentLease(SimpleNamespace()),
        spec=ltx23_sdr_signal(width=2, height=2, fps=24.0),
        frame_count=2,
        tiling_config=decode.TilingConfig(),
        vae_cache_limit_gb=4.0,
    )

    next(stream)
    stream.close()
    assert calls == [4 * 1024**3, 1234]
    assert stream.receipts["vae_cache"]["selected_limit_gb"] == 4.0


def test_auto_full_fit_bypasses_the_legacy_default_budget_replan(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(
        decode.TilingConfig,
        "auto_conv3d",
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


def test_context_sharpening_reaches_the_decoder_and_the_run_record(monkeypatch) -> None:
    seen = []

    def streaming(_latent, _decoder, *, context_sharpening, **_kwargs):
        seen.append(context_sharpening)
        yield mx.zeros((1, 3, 1, 2, 2))

    monkeypatch.setattr(decode, "decode_streaming", streaming)
    sharpening = ContextSharpening(amount=2.0, radius=3.0)
    stream = decode.decode_ltx23_sdr_frames(
        mx.zeros((1, 128, 1, 1, 1)),
        lambda: ComponentLease(SimpleNamespace(decoder_kind=VideoVAEKind.DIFFUSION)),
        spec=ltx23_sdr_signal(width=2, height=2, fps=24.0),
        frame_count=1,
        tiling_config=TilingConfig(),
        context_sharpening=sharpening,
    )

    assert len(list(stream)) == 1
    assert seen == [sharpening]
    diagnostics = decode.video_decode_diagnostics(stream)
    assert diagnostics["vae_decode"]["context_sharpening"] == {
        "amount": 2.0,
        "radius_pixels": 3.0,
    }


def test_capture_receipt_uses_full_headroom_without_charging_reclaimable_weights(
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
    monkeypatch.setattr(memory, "system_reclaimable_memory", lambda: 30_000_000_000)

    latent = mx.zeros((1, 3), dtype=mx.float16)
    decoder = SimpleNamespace(parameters=lambda: {"weight": mx.zeros((4,), dtype=mx.float16)})
    receipt = decode.capture_vae_entry_memory_receipt(
        latent=latent,
        decoder=decoder,
    )

    assert receipt.available_bytes == 25_000_000_000
    assert receipt.planner_budget_bytes == 25_000_000_000
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
        planner_budget_bytes=5_700,
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
            "decoder_kind": VideoVAEKind.DIFFUSION,
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
                "planner_budget_bytes": 5_700,
                "system_reclaimable_bytes": None,
                "memory_limit_bytes": None,
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
                "decoder_kind": VideoVAEKind.DIFFUSION,
                "loaded_tensors": 311,
            },
        }
    }


def _dtype_decoder(dtype: mx.Dtype, events: list[str]):
    def provider():
        events.append(f"load {dtype}")
        return ComponentLease(
            SimpleNamespace(compute_dtype=dtype),
            close_component=lambda _decoder: events.append(f"close {dtype}"),
        )

    return provider


def _overflowing_streaming(*, overflow_chunk: int):
    """Stream two chunks; an FP16 decoder's chunk ``overflow_chunk`` holds an infinity."""

    def streaming(_latent, decoder, **_kwargs):
        for index in range(2):
            chunk = _decoded_chunk()
            if decoder.compute_dtype == mx.float16 and index == overflow_chunk:
                chunk = chunk.at[0, 0, 0, 0, 0].add(mx.inf)
            yield chunk

    return streaming


def test_a_nonfinite_first_chunk_decodes_again_with_the_overflow_fallback(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(decode, "decode_streaming", _overflowing_streaming(overflow_chunk=0))
    stream = decode.decode_ltx23_sdr_frames(
        mx.zeros((1, 128, 2, 1, 1)),
        _dtype_decoder(mx.float16, events),
        spec=ltx23_sdr_signal(width=2, height=2, fps=24.0),
        frame_count=4,
        tiling_config=decode.TilingConfig(),
        overflow_fallback=_dtype_decoder(mx.bfloat16, events),
    )

    frames = list(stream)

    assert len(frames) == 4
    assert all(mx.all(mx.isfinite(frame)).item() for frame in frames)
    assert events == [
        f"load {mx.float16}",
        f"close {mx.float16}",
        f"load {mx.bfloat16}",
        f"close {mx.bfloat16}",
    ]
    diagnostics = decode.video_decode_diagnostics(stream)
    assert diagnostics["vae_decode"]["overflow_fallback"] == {
        "from_dtype": "float16",
        "to_dtype": "bfloat16",
    }


def test_a_nonfinite_later_chunk_stops_the_stream_after_its_frames(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(decode, "decode_streaming", _overflowing_streaming(overflow_chunk=1))
    stream = decode.decode_ltx23_sdr_frames(
        mx.zeros((1, 128, 2, 1, 1)),
        _dtype_decoder(mx.float16, events),
        spec=ltx23_sdr_signal(width=2, height=2, fps=24.0),
        frame_count=4,
        tiling_config=decode.TilingConfig(),
        overflow_fallback=_dtype_decoder(mx.bfloat16, events),
    )

    next(stream)
    next(stream)
    with pytest.raises(decode.LTX2DecodeError, match="after 2 frames were written") as failure:
        next(stream)

    assert "--vae-decode-dtype bfloat16" in str(failure.value)
    assert events == [f"load {mx.float16}", f"close {mx.float16}"]
    assert "overflow_fallback" not in decode.video_decode_diagnostics(stream)["vae_decode"]


def test_a_nonfinite_decode_without_a_fallback_stops_before_any_frame(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(decode, "decode_streaming", _overflowing_streaming(overflow_chunk=0))
    stream = decode.decode_ltx23_sdr_frames(
        mx.zeros((1, 128, 2, 1, 1)),
        _dtype_decoder(mx.float16, events),
        spec=ltx23_sdr_signal(width=2, height=2, fps=24.0),
        frame_count=4,
        tiling_config=decode.TilingConfig(),
    )

    with pytest.raises(decode.LTX2DecodeError, match=r"nonfinite frames in float16\. The"):
        next(stream)
    assert events == [f"load {mx.float16}", f"close {mx.float16}"]


@pytest.mark.parametrize(
    ("available_gb", "dtype", "expected_cache_gb"),
    [
        # A 768x448x121 decode runs in one pass beside its block set (five 1.33 GB features and
        # 4 GiB, 10 whole GiB) with room to spare, and beside the largest whole GiB the footprint
        # bound leaves otherwise: 5 GiB in 16 GB, 3 GiB in 14 GB. BF16 keeps the masked path and
        # its 4 GiB ladder.
        (40.0, mx.float16, 10.0),
        (16.0, mx.float16, 5.0),
        (14.0, mx.float16, 3.0),
        (40.0, mx.bfloat16, 4.0),
    ],
)
def test_fp16_diffusion_shrinks_its_cache_before_it_tiles(
    available_gb: float, dtype: mx.Dtype, expected_cache_gb: float
) -> None:
    cache_gb, _budget, tiling = decode._plan_vae_cache_and_tiling(
        available_bytes=int(available_gb * 1e9),
        requested_cache_gb=None,
        auto_tiling=True,
        tiling_config=None,
        decoder_kind=VideoVAEKind.DIFFUSION,
        compute_dtype=dtype,
        optimized_fp16=False,
        height=448,
        width=768,
        frames=121,
    )
    assert cache_gb == expected_cache_gb
    assert tiling is None


def _fp16_diffusion_plan(available_gb: float, height: int, width: int, frames: int):
    return decode._plan_vae_cache_and_tiling(
        available_bytes=int(available_gb * 1e9),
        requested_cache_gb=None,
        auto_tiling=True,
        tiling_config=None,
        decoder_kind=VideoVAEKind.DIFFUSION,
        compute_dtype=mx.float16,
        optimized_fp16=False,
        height=height,
        width=width,
        frames=frames,
        max_buffer_bytes=1 << 40,
    )


def test_fp16_diffusion_tiles_at_the_cache_its_time_estimate_favors() -> None:
    # 1024x576x721 with the 55 GB a 64 GB machine had: the fewest tiles that fit, three of 264
    # frames, beside the largest whole GiB the footprint bound leaves, 22 GiB, which holds the
    # churn. A smaller cache with longer tiles faults in more pages than the overlap it saves
    # (192-frame tiles at 8 GiB took 157.9 s against 154.6 for 128 at 16 GiB); beside the same
    # cache fewer tiles save overlap (192-frame tiles at 16 GiB took 145.4 s against 152.1 for
    # 128, 224-frame ones 141.9, three 264-frame ones 138.3, and 137.1 with the block set cached).
    cache_gb, _budget, tiling = _fp16_diffusion_plan(55.17, 576, 1024, 721)
    assert cache_gb == 22.0
    assert tiling is not None
    assert tiling.spatial_config is None
    assert tiling.temporal_config is not None
    assert tiling.temporal_config.chunk_size_in_frames == 264
    assert tiling.temporal_config.chunk_overlap_in_frames == 24
    # With more room the cache grows toward the tile's block set (five 5.0 GB features and
    # 4 GiB, 27 GiB) and no further; with 110 GB the clip decodes in one pass.
    cache_gb, _budget, tiling = _fp16_diffusion_plan(60.0, 576, 1024, 721)
    assert tiling is not None
    assert tiling.temporal_config is not None
    assert (cache_gb, tiling.temporal_config.chunk_size_in_frames) == (26.0, 264)
    cache_gb, _budget, tiling = _fp16_diffusion_plan(110.0, 576, 1024, 721)
    assert tiling is None
    assert cache_gb == 56.0
    # A requested cache is kept as such.
    cache_gb, _budget, tiling = decode._plan_vae_cache_and_tiling(
        available_bytes=int(55.17e9),
        requested_cache_gb=16.0,
        auto_tiling=True,
        tiling_config=None,
        decoder_kind=VideoVAEKind.DIFFUSION,
        compute_dtype=mx.float16,
        optimized_fp16=False,
        height=576,
        width=1024,
        frames=721,
    )
    assert tiling is not None
    assert tiling.temporal_config is not None
    assert (cache_gb, tiling.temporal_config.chunk_size_in_frames) == (16.0, 264)
    # With less headroom the tiles shorten and the cache shrinks in 4 GiB steps.
    cache_gb, _budget, tiling = _fp16_diffusion_plan(40.0, 576, 1024, 721)
    assert tiling is not None
    assert tiling.temporal_config is not None
    assert (cache_gb, tiling.temporal_config.chunk_size_in_frames) == (12.0, 208)


def test_fp16_diffusion_prefers_temporal_tiles_of_64_frames_or_more() -> None:
    # 1080p, 257 frames: shorter tiles fit beside a larger cache, but each would recompute most
    # of what it advances.
    _cache_gb, _budget, tiling = _fp16_diffusion_plan(55.17, 1088, 1920, 257)
    assert tiling is not None
    assert tiling.spatial_config is None
    assert tiling.temporal_config is not None
    assert tiling.temporal_config.chunk_size_in_frames >= 64


def test_fp16_diffusion_never_plans_a_tile_the_kernels_cannot_address() -> None:
    # 3,700 frames of 1024x576 hold more stage-5 tokens than the fused Q/K pass launches
    # (134,217,696): with memory for one pass, the planner still tiles.
    _cache_gb, _budget, tiling = _fp16_diffusion_plan(2000.0, 576, 1024, 3700)
    assert tiling is not None
    assert tiling.spatial_config is None
    assert tiling.temporal_config is not None
    # A requested single pass of that clip is refused before the decode starts.
    with pytest.raises(ValueError, match="cannot address"):
        decode._plan_vae_cache_and_tiling(
            available_bytes=int(2000e9),
            requested_cache_gb=None,
            auto_tiling=False,
            tiling_config=TilingConfig(),
            decoder_kind=VideoVAEKind.DIFFUSION,
            compute_dtype=mx.float16,
            optimized_fp16=False,
            height=576,
            width=1024,
            frames=3700,
            max_buffer_bytes=1 << 40,
        )
