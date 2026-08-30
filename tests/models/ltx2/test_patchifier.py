from __future__ import annotations

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.patchifier import VideoLatentPatchifier, get_pixel_coords
from kinomlx.types import SpatioTemporalScaleFactors, VideoLatentShape


def test_video_patchifier_round_trip_and_grid_order() -> None:
    patchifier = VideoLatentPatchifier(patch_size=2)
    shape = VideoLatentShape(1, 2, 2, 4, 6)
    latent = mx.arange(96, dtype=mx.float32).reshape(shape.to_tuple())

    tokens = patchifier.patchify(latent)
    restored = patchifier.unpatchify(tokens, shape)
    bounds = patchifier.get_patch_grid_bounds(shape)
    mx.eval(tokens, restored, bounds)

    assert patchifier.patch_size == (1, 2, 2)
    assert patchifier.get_token_count(shape) == 12
    assert tuple(tokens.shape) == (1, 12, 8)
    assert mx.array_equal(restored, latent).item()
    assert tuple(bounds.shape) == (1, 3, 12, 2)
    assert bounds[0, :, 0, :].tolist() == [[0, 1], [0, 2], [0, 2]]
    assert bounds[0, :, 1, :].tolist() == [[0, 1], [0, 2], [2, 4]]


def test_video_patchifier_rejects_partial_spatial_patches() -> None:
    patchifier = VideoLatentPatchifier(patch_size=2)
    with pytest.raises(ValueError, match="divide evenly"):
        patchifier.patchify(mx.zeros((1, 2, 1, 3, 4)))


def test_pixel_coords_apply_scale_and_causal_first_frame_fix() -> None:
    shape = VideoLatentShape(1, 1, 2, 1, 1)
    latent = VideoLatentPatchifier().get_patch_grid_bounds(shape)
    scale = SpatioTemporalScaleFactors(time=8, height=32, width=32)

    regular = get_pixel_coords(latent, scale)
    causal = get_pixel_coords(latent, scale, causal_fix=True)
    mx.eval(regular, causal)

    assert regular[0, :, 0, :].tolist() == [[0, 8], [0, 32], [0, 32]]
    assert regular[0, 0, 1, :].tolist() == [8, 16]
    assert causal[0, 0, 0, :].tolist() == [0, 1]
    assert causal[0, 0, 1, :].tolist() == [1, 9]
    assert mx.array_equal(causal[:, 1:, ...], regular[:, 1:, ...]).item()
