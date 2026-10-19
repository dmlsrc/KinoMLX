"""Streaming VAE tiling, blending, and memory-plan tests."""

from collections.abc import Iterator

import mlx.core as mx
import pytest

from kinomlx.modeling.memory import DecodeMemoryBudget
from kinomlx.models.ltx2.video_vae import tiling
from kinomlx.models.ltx2.video_vae.kinds import VideoVAEKind
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
)
from tests._reporting import RecordingReporter

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


def test_streaming_tiles_leave_allocator_cache_to_mlx(monkeypatch) -> None:
    clears = []
    monkeypatch.setattr(mx, "clear_cache", lambda: clears.append(True))

    _run(_latent(), TilingConfig(temporal_config=_temporal_only()))

    assert clears == []


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
    assert (
        TilingConfig.auto(
            768,
            448,
            121,
            memory_budget_gb=budget_gb,
            compute_dtype=mx.float16,
        )
        is None
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


def test_optimized_fp16_plan_caps_long_decode_at_measured_temporal_size() -> None:
    kwargs = {"compute_dtype": mx.float16, "optimized_fp16": True}
    assert TilingConfig.auto(768, 448, 121, memory_budget_gb=50.0, **kwargs) is None
    assert TilingConfig.auto(768, 448, 721, memory_budget_gb=200.0, **kwargs) == TilingConfig(
        temporal_config=TemporalChunkConfig(128, 8)
    )
    assert TilingConfig.auto(768, 448, 721, memory_budget_gb=12.0, **kwargs) == TilingConfig(
        temporal_config=TemporalChunkConfig(64, 8)
    )


def test_auto_plan_tries_smallest_temporal_chunk_before_spatial_tiling() -> None:
    plan = TilingConfig.auto(1024, 576, 300, memory_budget_gb=12.0)

    assert plan is not None
    assert plan.spatial_config is None
    assert plan.temporal_config == TemporalChunkConfig(16, 8)


def test_auto_plan_accounts_for_full_frame_spatial_assembly_buffers() -> None:
    plan = TilingConfig.auto(2048, 1152, 300, memory_budget_gb=12.0)

    assert plan is not None
    assert plan.spatial_config == SpatialTilingConfig(512, 64)
    assert plan.temporal_config == TemporalChunkConfig(16, 8)


def test_conv3d_does_not_return_an_over_budget_fallback() -> None:
    with pytest.raises(MemoryError, match="No Conv3d VAE tile fits"):
        TilingConfig.auto_conv3d(1024, 576, 121, memory_budget_gb=0.5)


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


def test_low_level_auto_plan_uses_shared_headroom_without_a_16gb_cap(monkeypatch) -> None:
    monkeypatch.setattr(
        tiling,
        "capture_decode_memory_budget",
        lambda: DecodeMemoryBudget(
            limit_bytes=60 * 1024**3,
            active_bytes=2 * 1024**3,
            cache_bytes=0,
            system_reclaimable_bytes=55 * 1024**3,
        ),
    )
    assert TilingConfig.auto(1024, 576, 121) is None


@pytest.mark.parametrize("compute_dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_diffusion_auto_plan_uses_memory_instead_of_fixed_dimensions(compute_dtype) -> None:
    assert (
        TilingConfig.auto_diffusion(
            1024, 576, 121, memory_budget_gb=60.0, compute_dtype=compute_dtype
        )
        is None
    )
    assert (
        TilingConfig.auto_diffusion(
            1024, 576, 121, memory_budget_gb=12.0, compute_dtype=compute_dtype
        )
        is not None
    )


def test_diffusion_does_not_return_a_plan_that_exceeds_the_smallest_budget() -> None:
    with pytest.raises(MemoryError, match="No diffusion VAE tile fits"):
        TilingConfig.auto_diffusion(1024, 576, 121, memory_budget_gb=0.5)


def test_diffusion_auto_accounts_for_minimum_padded_working_grid() -> None:
    assert TilingConfig.auto_diffusion(64, 64, 1, memory_budget_gb=1.4) is not None


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


def test_fp16_diffusion_estimate_covers_the_kernel_routes_measured_peaks() -> None:
    # Active peaks above the resident weights of two 768x448 decodes in one pass at a 16 GiB
    # cache on the kernel route: the SwiGLU half at 65 frames, the attention half at 121.
    for frames, measured_gb in ((65, 5.00), (121, 8.44)):
        estimate = tiling._estimate_diffusion_peak_gb(448, 768, frames, compute_dtype=mx.float16)
        assert measured_gb < estimate < 1.25 * measured_gb


def test_diffusion_candidates_list_every_plan_and_options_keep_those_that_fit() -> None:
    candidates = TilingConfig.diffusion_candidates(576, 1024, 721, max_buffer_bytes=1 << 40)
    assert candidates[0] is None
    temporal_only = [
        option.temporal_config.chunk_size_in_frames
        for option in candidates[1:]
        if option is not None and option.spatial_config is None
    ]
    assert temporal_only[:4] == [384, 264, 208, 168]
    # Temporal-only plans come before every spatially tiled one.
    kinds = [option.spatial_config is not None for option in candidates[1:] if option is not None]
    assert kinds == sorted(kinds)
    options = TilingConfig.diffusion_options(
        576, 1024, 721, memory_budget_gb=45.0, compute_dtype=mx.float16, max_buffer_bytes=1 << 40
    )
    assert None not in options
    assert all(option in candidates for option in options)
    assert options == [
        option
        for option in candidates
        if option is not None
        and option.estimated_peak_gb(
            576, 1024, 721, decoder_kind=VideoVAEKind.DIFFUSION, compute_dtype=mx.float16
        )
        <= 45.0
    ]


def test_kernel_route_feature_size_follows_the_tile() -> None:
    # A 264-frame 1024x576 tile: 265 frames of 144x256 tokens, 256 channels, FP16.
    plan = TilingConfig(temporal_config=TemporalChunkConfig(264, 24))
    assert abs(plan.kernel_route_feature_gb(576, 1024, 721) - 265 * 144 * 256 * 512 / 1e9) < 1e-9
    assert (
        abs(TilingConfig().kernel_route_feature_gb(448, 768, 121) - 121 * 112 * 192 * 512 / 1e9)
        < 1e-9
    )


def test_diffusion_candidates_leave_out_what_the_kernels_or_device_cannot_address() -> None:
    # A 4K plane holds 522,240 stage-5 tokens; one pass of 17 frames is addressable since the
    # attention kernels form each key run's address in 64 bits.
    assert TilingConfig.diffusion_candidates(2160, 3840, 17, max_buffer_bytes=1 << 40)[0] is None
    # 3,700 frames of 1024x576 hold 136 million stage-5 tokens, past the fused Q/K pass's
    # 134,217,696: the single pass is left out and the fewest temporal tiles lead.
    vast = TilingConfig.diffusion_candidates(576, 1024, 3700, max_buffer_bytes=1 << 40)
    assert vast[0] is not None
    assert vast[0].spatial_config is None
    assert all(
        option is not None
        and option.addressable(576, 1024, 3700, compute_dtype=mx.float16, max_buffer_bytes=1 << 40)
        for option in vast
    )
    single = TilingConfig()
    assert not single.addressable(
        576, 1024, 3700, compute_dtype=mx.float16, max_buffer_bytes=1 << 40
    )
    # The masked path has no such bound; the device's buffer limit binds every route.
    assert single.addressable(576, 1024, 3700, compute_dtype=mx.bfloat16, max_buffer_bytes=1 << 40)
    assert not single.addressable(
        576, 1024, 3700, compute_dtype=mx.bfloat16, max_buffer_bytes=1 << 35
    )
    # 10 GB leaves out the 13.6 GB single pass of 721 frames and keeps two 384-frame tiles.
    limited = TilingConfig.diffusion_candidates(576, 1024, 721, max_buffer_bytes=10_000_000_000)
    assert limited[0] is not None
    assert limited[0].temporal_config is not None
    assert limited[0].temporal_config.chunk_size_in_frames == 384
    # This device addresses a 768x448x121 pass (a 1.33 GB feature).
    assert TilingConfig.diffusion_candidates(448, 768, 121)[0] is None
    # A clip whose latent context alone exceeds the limit has no plan.
    with pytest.raises(ValueError, match="latent context exceeds"):
        TilingConfig.diffusion_candidates(576, 1024, 721, max_buffer_bytes=1_000_000_000)


def test_temporal_tiles_are_chosen_by_count_with_equal_lengths() -> None:
    # 721 decoded frames are 361 stage-4 frames; with the 24-frame overlap, the shortest tiles
    # that give 2, 3, 4 and 5 tiles. The work is set by the count, so the shortest tile with a
    # count has the smallest peak: 200-frame tiles cover what a fixed 224-frame ladder covered.
    sizes = tiling._equalized_temporal_tiles(361)
    assert sizes[:4] == [384, 264, 208, 168]
    assert sizes[-1] == 16
    assert sizes == sorted(sizes, reverse=True)
    counts = []
    for size in sizes:
        assert size % 8 == 0
        overlap = min(24, size - 8)
        intervals = tiling.tile_intervals(361, size // 2, (size - overlap) // 2, 6)
        counts.append(len(intervals))
        # Tiles start a stride apart from zero and end at the clip; only the ones reaching the
        # clip's end differ from the tile length, the last by less than a stride either way.
        lengths = [end - start for start, end in intervals]
        tile, stride = size // 2, (size - overlap) // 2
        assert tile in lengths
        assert [start for start, _end in intervals] == [
            index * stride for index in range(len(intervals))
        ]
        assert intervals[-1][1] == 361
        assert all(end == 361 for start, end in intervals if end - start != tile)
        assert 6 <= lengths[-1] < tile + stride
    # One size per count it can produce, fewest tiles first; counts rounding skips are absent.
    assert counts[:4] == [2, 3, 4, 5]
    assert counts == sorted(set(counts))
    assert [end - start for start, end in tiling.tile_intervals(361, 104, 92, 6)] == [
        104,
        104,
        104,
        85,
    ]
    # A clip too short to split gives no temporal tiles at all.
    assert tiling._equalized_temporal_tiles(9) == []


def test_tile_intervals_merge_a_short_last_tile() -> None:
    assert tiling.tile_intervals(10, 16, 8, 6) == [(0, 10)]
    assert tiling.tile_intervals(361, 64, 52, 6) == [
        (0, 64),
        (52, 116),
        (104, 168),
        (156, 220),
        (208, 272),
        (260, 324),
        (312, 361),
    ]
    # A last start leaving fewer than six rows folds into the tile before it.
    assert tiling.tile_intervals(61, 40, 28, 6) == [(0, 40), (28, 61)]


def test_fp16_diffusion_time_estimate_ranks_the_measured_plans() -> None:
    # One tile of each 1024x576x721 plan measured on an M1 Max and scaled to the plan's tiles
    # (the first tile reproduces the full decodes' tiles), without the 11.6 s of stages 1 to 3,
    # blends and output that every plan shares.
    def estimate(size: int, cache_gb: float) -> float:
        plan = TilingConfig(temporal_config=TemporalChunkConfig(size, 24))
        return tiling.estimated_diffusion_seconds(plan, 576, 1024, 721, cache_gb=cache_gb)

    for size, cache_gb, measured in (
        (128, 16.0, 140.9),
        (192, 16.0, 133.3),
        (192, 8.0, 138.8),
        (192, 4.0, 147.7),
        (224, 16.0, 131.5),
    ):
        assert abs(estimate(size, cache_gb) - measured) < 0.03 * measured
    # Beside one cache a longer tile saves overlap; a smaller cache faults in more pages.
    assert estimate(224, 16.0) < estimate(192, 16.0) < estimate(128, 16.0)
    assert estimate(192, 16.0) < estimate(192, 8.0) < estimate(192, 4.0)
    # More cache never costs time; one pass costs no tile overhead.
    assert estimate(128, 16.0) <= estimate(128, 4.0)
    single = tiling.estimated_diffusion_seconds(None, 448, 768, 121, cache_gb=16.0)
    tiled = tiling.estimated_diffusion_seconds(
        TilingConfig(temporal_config=TemporalChunkConfig(64, 24)), 448, 768, 121, cache_gb=16.0
    )
    assert single < tiled


def test_diffusion_options_list_every_plan_that_fits_largest_first() -> None:
    options = TilingConfig.diffusion_options(
        448, 768, 121, memory_budget_gb=60.0, compute_dtype=mx.float16
    )
    assert options[0] is None
    assert all(
        option is None or option.spatial_config is not None or option.temporal_config is not None
        for option in options
    )
    # One pass needs about 9.5 GB here (the kernel route's bound plus the stage-3 context).
    tight = TilingConfig.diffusion_options(
        448, 768, 121, memory_budget_gb=9.0, compute_dtype=mx.float16
    )
    assert None not in tight
    assert (
        TilingConfig.auto_diffusion(448, 768, 121, memory_budget_gb=9.0, compute_dtype=mx.float16)
        == tight[0]
    )
