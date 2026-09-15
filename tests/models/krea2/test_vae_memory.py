"""Krea VAE plans consume shared headroom and respect real allocation costs."""

import mlx.core as mx
import pytest

from kinomlx.modeling.memory import DecodeMemoryBudget
from kinomlx.models.krea2.metadata import Krea2VAEConfig
from kinomlx.models.krea2.vae.tiling import Krea2VAETilingConfig, estimate_decode_memory

GIB = 1024**3


def _config() -> Krea2VAEConfig:
    return Krea2VAEConfig(
        base_dim=96,
        dim_mult=(1, 2, 4, 4),
        input_channels=3,
        z_dim=16,
        num_res_blocks=2,
        attn_scales=(),
        latent_mean=(0.0,) * 16,
        latent_std=(1.0,) * 16,
    )


def _budget(gib: int, *, buffer_gib: int = 39) -> DecodeMemoryBudget:
    return DecodeMemoryBudget(
        limit_bytes=gib * GIB,
        active_bytes=0,
        cache_bytes=0,
        max_buffer_bytes=buffer_gib * GIB,
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("size", [192, 1024, 2048])
def test_auto_full_image_is_not_limited_to_256_pixels(dtype, size) -> None:
    assert (
        Krea2VAETilingConfig.auto(_config(), size, size, compute_dtype=dtype, budget=_budget(60))
        is None
    )


def test_auto_uses_largest_fitting_tile_and_preserves_alignment() -> None:
    config = _config()
    plan = Krea2VAETilingConfig.auto(
        config, 2048, 2048, compute_dtype=mx.bfloat16, budget=_budget(12)
    )
    assert plan is not None
    assert 256 < plan.tile_sample_size < 2048
    assert plan.tile_sample_size % 8 == plan.tile_sample_stride % 8 == 0
    assembly = 2048 * 2048 * 3 * 4 * 6
    peak, _buffer = estimate_decode_memory(
        config, plan.tile_sample_size, plan.tile_sample_size, compute_dtype=mx.bfloat16
    )
    next_peak, _buffer = estimate_decode_memory(
        config, plan.tile_sample_size + 8, plan.tile_sample_size + 8, compute_dtype=mx.bfloat16
    )
    assert peak + assembly <= 12 * GIB < next_peak + assembly


def test_auto_respects_maximum_single_buffer_even_with_plenty_of_memory() -> None:
    plan = Krea2VAETilingConfig.auto(
        _config(), 1024, 1024, compute_dtype=mx.bfloat16, budget=_budget(60, buffer_gib=1)
    )
    assert plan is not None
    _peak, buffer = estimate_decode_memory(
        _config(), plan.tile_sample_size, plan.tile_sample_size, compute_dtype=mx.bfloat16
    )
    assert buffer <= GIB


def test_auto_reserves_the_outputs_retained_by_a_multi_image_request() -> None:
    plan = Krea2VAETilingConfig.auto(
        _config(),
        2048,
        2048,
        compute_dtype=mx.bfloat16,
        budget=_budget(60),
        reserved_output_bytes=40 * GIB,
    )
    assert plan is not None


def test_estimate_includes_unfused_fp32_attention_at_large_resolution() -> None:
    peak, buffer = estimate_decode_memory(_config(), 2048, 2048, compute_dtype=mx.bfloat16)
    score = (256 * 256) ** 2 * 4
    assert peak >= 2 * score
    assert buffer >= score


def test_auto_fails_before_decode_if_not_even_a_small_tile_fits() -> None:
    with pytest.raises(MemoryError, match="No Krea VAE tile fits"):
        Krea2VAETilingConfig.auto(
            _config(), 1024, 1024, compute_dtype=mx.bfloat16, budget=_budget(0)
        )
