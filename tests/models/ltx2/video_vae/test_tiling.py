"""Streaming VAE tiling, blending, and memory-plan tests."""

from __future__ import annotations

from collections.abc import Iterator

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.video_vae.tiling import (
    DEFAULT_SPATIAL_SCALE,
    DEFAULT_TEMPORAL_SCALE,
    SpatialTilingConfig,
    TemporalChunkConfig,
    TilingConfig,
    _split_axis,
    _split_temporal_axis,
    compute_trapezoidal_mask_1d,
    decode_single_pass,
    decode_streaming,
    default_vae_decode_budget_gb,
    detect_system_memory_gb,
)
from kinomlx.reporting import RecordingReporter

TEMPORAL_SCALE = DEFAULT_TEMPORAL_SCALE
SPATIAL_SCALE = DEFAULT_SPATIAL_SCALE


def _causal_upscale(latent: mx.array) -> mx.array:
    """Pointwise fake decoder with the native VAE's causal output geometry."""
    _batch, _channels, frames, _height, _width = latent.shape
    base = mx.concatenate(
        [latent[:, :1], latent[:, :1] + 1.0, latent[:, :1] * 0.5],
        axis=1,
    )
    output_frames = 1 + (frames - 1) * TEMPORAL_SCALE
    indices = [0] + [1 + (frame - 1) // TEMPORAL_SCALE for frame in range(1, output_frames)]
    decoded = mx.take(base, mx.array(indices), axis=2)
    decoded = mx.repeat(decoded, SPATIAL_SCALE, axis=3)
    decoded = mx.repeat(decoded, SPATIAL_SCALE, axis=4)
    return decoded.astype(mx.float32)


def _decoder_fn(
    tile: mx.array,
    timestep: float | None = None,
) -> mx.array:
    del timestep
    return _causal_upscale(tile)


def _run(
    latent: mx.array,
    config: TilingConfig | None,
    *,
    reporter: RecordingReporter | None = None,
) -> mx.array:
    chunks = list(
        decode_streaming(
            latent,
            _decoder_fn,
            config,
            timestep=None,
            reporter=reporter,
        )
    )
    return mx.concatenate(chunks, axis=2)


def _latent() -> mx.array:
    count = 1 * 4 * 6 * 2 * 2
    return (mx.arange(count).astype(mx.float32) / 7.0).reshape(1, 4, 6, 2, 2)


def _temporal_only() -> TemporalChunkConfig:
    return TemporalChunkConfig(chunk_size_in_frames=16, chunk_overlap_in_frames=8)


def test_temporal_only_reconstructs_full_decode() -> None:
    latent = _latent()
    output = _run(
        latent,
        TilingConfig(temporal_config=_temporal_only()),
    )
    expected = _causal_upscale(latent)
    assert tuple(output.shape) == tuple(expected.shape)
    assert mx.allclose(output, expected, atol=1e-4, rtol=1e-4).item()


def test_reduced_temporal_weights_match_full_spatial_weights() -> None:
    latent = _latent()
    temporal = _temporal_only()
    reduced = _run(latent, TilingConfig(temporal_config=temporal))
    full = _run(
        latent,
        TilingConfig(
            spatial_config=SpatialTilingConfig(tile_size_in_pixels=64),
            temporal_config=temporal,
        ),
    )
    assert tuple(reduced.shape) == tuple(full.shape)
    assert mx.allclose(reduced, full, atol=1e-5, rtol=1e-5).item()


def test_tiled_decode_reports_every_job_and_closes_phase() -> None:
    reporter = RecordingReporter()
    plans = []
    config = TilingConfig(temporal_config=_temporal_only())
    list(
        decode_streaming(
            _latent(),
            _decoder_fn,
            config,
            reporter=reporter,
            plan_callback=plans.append,
        )
    )
    assert reporter.events[0] == (
        "start",
        "VAE decode tiles",
        {"total": 5, "unit": "tile"},
    )
    advances = [event for event in reporter.events if event[0] == "advance"]
    assert len(advances) == 5
    assert reporter.events[-1] == ("end", "VAE decode tiles", {})
    assert len(plans) == 1
    assert plans[0].to_dict() == {
        "latent_shape": [6, 2, 2],
        "decoded_shape": [41, 64, 64],
        "temporal_tiles": 5,
        "spatial_height_tiles": 1,
        "spatial_width_tiles": 1,
        "total_tiles": 5,
        "resolved_config": {
            "temporal_tile_frames": 16,
            "temporal_overlap_frames": 8,
            "spatial_tile_pixels": None,
            "spatial_overlap_pixels": None,
        },
    }


def test_single_job_stream_yields_raw_decoder_output_without_fp32_accumulator() -> None:
    reporter = RecordingReporter()
    expected = mx.zeros((1, 3, 1, 32, 32), dtype=mx.bfloat16)

    def decoder(
        _latent: mx.array,
        *,
        timestep: float | None = None,
    ) -> mx.array:
        del timestep
        return expected

    stream = decode_streaming(
        mx.zeros((1, 4, 1, 1, 1)),
        decoder,
        TilingConfig(),
        reporter=reporter,
    )
    chunk = next(stream)
    assert chunk is expected
    assert chunk.dtype == mx.bfloat16
    # A single-tile decode is complete before its materialized chunk is handed
    # to a slower downstream consumer such as VSR + HEVC.
    assert reporter.events == [
        ("start", "VAE decode tiles", {"total": 1, "unit": "tile"}),
        ("advance", "VAE decode tiles", {"advance": 1.0}),
        ("end", "VAE decode tiles", {}),
    ]
    with pytest.raises(StopIteration):
        next(stream)


def test_closing_partly_consumed_stream_releases_reporter() -> None:
    reporter = RecordingReporter()
    stream: Iterator[mx.array] = decode_streaming(
        _latent(),
        _decoder_fn,
        TilingConfig(temporal_config=_temporal_only()),
        reporter=reporter,
    )
    next(stream)
    stream.close()
    assert reporter.events[-1] == ("end", "VAE decode tiles", {})


def test_decoder_failure_releases_reporter() -> None:
    reporter = RecordingReporter()

    def fail(_latent: mx.array, *, timestep: float | None = None) -> mx.array:
        del timestep
        raise RuntimeError("decode failed")

    with pytest.raises(RuntimeError, match="decode failed"):
        list(
            decode_streaming(
                _latent(),
                fail,
                TilingConfig(temporal_config=_temporal_only()),
                reporter=reporter,
            )
        )
    assert reporter.events[-1] == ("end", "VAE decode tiles", {})


def test_single_pass_accepts_unbatched_latent_and_forwards_reporter() -> None:
    reporter = RecordingReporter()
    seen: dict[str, object] = {}

    def decoder(
        latent: mx.array,
        *,
        timestep: float | None,
        reporter: RecordingReporter | None,
    ) -> mx.array:
        seen["shape"] = tuple(latent.shape)
        seen["timestep"] = timestep
        seen["reporter"] = reporter
        return mx.zeros((1, 3, 1, 32, 32))

    output = decode_single_pass(
        mx.zeros((4, 1, 1, 1)),
        decoder,
        timestep=None,
        reporter=reporter,
    )
    assert tuple(output.shape) == (1, 3, 1, 32, 32)
    assert seen == {
        "shape": (1, 4, 1, 1, 1),
        "timestep": None,
        "reporter": reporter,
    }


def test_auto_plan_prefers_temporal_only_before_spatial_tiling() -> None:
    assert TilingConfig.auto(64, 64, 9, memory_budget_gb=6.0) is None
    plan = TilingConfig.auto(1024, 576, 300, memory_budget_gb=31.8)
    assert plan is not None
    assert plan.spatial_config is None
    assert plan.temporal_config == TemporalChunkConfig(64, 8)


def test_auto_plan_scales_conv3d_estimate_for_fp32_compute() -> None:
    budget_gb = 31.43
    assert (
        TilingConfig.auto(
            768,
            448,
            121,
            memory_budget_gb=budget_gb,
            compute_dtype=mx.bfloat16,
        )
        is None
    )
    with pytest.raises(ValueError, match="unsupported native Conv3d compute dtype"):
        TilingConfig.auto(
            768,
            448,
            121,
            memory_budget_gb=budget_gb,
            compute_dtype=mx.float16,
        )

    fp32 = TilingConfig.auto(
        768,
        448,
        121,
        memory_budget_gb=budget_gb,
        compute_dtype=mx.float32,
    )
    assert fp32 is not None
    assert fp32.spatial_config is None
    assert fp32.temporal_config == TemporalChunkConfig(40, 8)


def test_auto_plan_uses_spatial_tiling_when_incremental_peak_exceeds_budget() -> None:
    plan = TilingConfig.auto(1024, 576, 300, memory_budget_gb=12.0)

    assert plan is not None
    assert plan.spatial_config == SpatialTilingConfig(512, 64)
    assert plan.temporal_config == TemporalChunkConfig(32, 8)


def test_auto_plan_accounts_for_full_frame_spatial_assembly_buffers() -> None:
    plan = TilingConfig.auto(2048, 1152, 300, memory_budget_gb=12.0)

    assert plan is not None
    assert plan.spatial_config == SpatialTilingConfig(512, 64)
    assert plan.temporal_config == TemporalChunkConfig(32, 8)


def test_public_streaming_decode_never_exceeds_spatial_tile_maximum() -> None:
    seen: list[tuple[int, int]] = []

    def recording_decoder(
        tile: mx.array,
        *,
        timestep: float | None = None,
    ) -> mx.array:
        del timestep
        seen.append((int(tile.shape[3]), int(tile.shape[4])))
        return mx.zeros(
            (
                tile.shape[0],
                3,
                1 + (tile.shape[2] - 1) * TEMPORAL_SCALE,
                tile.shape[3] * SPATIAL_SCALE,
                tile.shape[4] * SPATIAL_SCALE,
            ),
            dtype=mx.float32,
        )

    list(
        decode_streaming(
            mx.zeros((1, 128, 1, 18, 18)),
            recording_decoder,
            TilingConfig(
                spatial_config=SpatialTilingConfig(512, 64),
            ),
        )
    )

    assert seen
    assert all(height <= 16 and width <= 16 for height, width in seen)


@pytest.mark.parametrize(
    ("length", "tile_size", "overlap"),
    [
        (32, 16, 2),
        (31, 16, 2),
        (17, 16, 2),
        (61, 32, 8),
    ],
)
def test_axis_tiles_match_declared_overlap_and_maximum(
    length: int,
    tile_size: int,
    overlap: int,
) -> None:
    tiles = _split_axis(length, tile_size, overlap)

    assert all(
        end - start <= tile_size for start, end in zip(tiles.starts, tiles.ends, strict=True)
    )
    for index in range(1, len(tiles.starts)):
        actual = tiles.ends[index - 1] - tiles.starts[index]
        assert actual == tiles.right_ramps[index - 1]
        assert actual == tiles.left_ramps[index]


def test_long_temporal_plan_keeps_causal_context_separate_from_blend_overlap() -> None:
    tiles = _split_temporal_axis(61, 8, 1)  # 481 decoded frames.

    assert all(end - start <= 9 for start, end in zip(tiles.starts, tiles.ends, strict=True))
    for index in range(1, len(tiles.starts)):
        actual = tiles.ends[index - 1] - tiles.starts[index]
        assert actual == tiles.left_ramps[index]
        assert actual == tiles.right_ramps[index - 1] + 1


def test_auto_plan_has_no_stale_conv3d_address_boundary() -> None:
    assert (
        TilingConfig.auto(
            1024,
            576,
            457,
            memory_budget_gb=200.0,
        )
        is None
    )


def test_memory_detection_and_budget_use_mlx_device_info(monkeypatch) -> None:
    monkeypatch.setattr(
        mx,
        "device_info",
        lambda: {"memory_size": 24_000_000_000},
    )
    assert detect_system_memory_gb() == 24.0
    assert default_vae_decode_budget_gb() == 12.0


def test_trapezoidal_mask_has_expected_ramps() -> None:
    mask = compute_trapezoidal_mask_1d(5, 2, 2)
    assert mx.allclose(
        mask,
        mx.array([1 / 3, 2 / 3, 1.0, 2 / 3, 1 / 3]),
    ).item()


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: SpatialTilingConfig(32), "64 or greater"),
        (lambda: SpatialTilingConfig(64, -32), "must be non-negative"),
        (lambda: SpatialTilingConfig(64.0), "must be an integer"),
        (lambda: TemporalChunkConfig(8), "at least 16"),
        (lambda: TemporalChunkConfig(16, -8), "must be non-negative"),
        (lambda: TemporalChunkConfig(True), "must be an integer"),
    ],
)
def test_tiling_configs_reject_unsafe_geometry(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()
