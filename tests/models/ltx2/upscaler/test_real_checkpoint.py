"""Opt-in canary for the official LTX-2.3 x2 spatial upscaler."""

import gc

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.upscaler import load_spatial_upscaler
from tests.models.ltx2._source_weights import NO_LTX23_CHECKPOINT, ltx23_checkpoint


@pytest.mark.heavy
@pytest.mark.parametrize("version", ["1.0", "1.1"])
def test_real_ltx23_spatial_upscaler(version: str) -> None:
    checkpoint = ltx23_checkpoint()
    if checkpoint is None:
        pytest.skip(NO_LTX23_CHECKPOINT)
    path = checkpoint.parent / f"ltx-2.3-spatial-upscaler-x2-{version}.safetensors"
    if not path.is_file():
        pytest.skip("the official LTX-2.3 x2 spatial upscaler is not cached")

    model = None
    latent = None
    result = None
    try:
        model = load_spatial_upscaler(path)
        latent = mx.random.normal((1, 128, 1, 2, 2)).astype(mx.bfloat16)
        result = model(latent)
        mx.eval(result)
        assert tuple(result.shape) == (1, 128, 1, 4, 4)
        assert result.dtype == mx.bfloat16
        assert mx.all(mx.isfinite(result)).item()
        assert mx.any(result != 0).item()
    finally:
        del model, latent, result
        gc.collect()
        mx.clear_cache()
