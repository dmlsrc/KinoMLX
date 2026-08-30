"""HDR IC-LoRA reference preprocessing and appended-token semantics."""

from __future__ import annotations

import mlx.core as mx
import pytest

from kinomlx.models.ltx2 import hdr_reference
from kinomlx.models.ltx2.conditioning.reference import VideoConditionByReferenceLatent
from kinomlx.models.ltx2.hdr_reference import (
    encode_reference_video,
    resize_and_reflect_pad_sdr,
)
from kinomlx.models.ltx2.state import create_video_latent_tools, init_video_latent_state
from kinomlx.types import VideoLatentShape


def test_short_reference_video_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # The IC-LoRA is a video-to-video converter: generation frames without a
    # reference-track frame run it off-distribution, so coverage is required.
    monkeypatch.setattr(
        hdr_reference,
        "read_sdr_video_frames",
        lambda _path, *, max_frames: mx.zeros((3, 4, 4, 3), dtype=mx.float32),
    )
    with pytest.raises(ValueError, match="covers 3 of the 9 requested frames"):
        encode_reference_video(
            "reference.mp4",
            lambda *_args, **_kwargs: pytest.fail("short reference must not reach the VAE"),
            width=4,
            height=4,
            frames=9,
            compute_dtype=mx.float32,
        )


def test_full_coverage_reference_video_encodes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hdr_reference,
        "read_sdr_video_frames",
        lambda _path, *, max_frames: mx.zeros((max_frames, 4, 4, 3), dtype=mx.float32),
    )
    latent = encode_reference_video(
        "reference.mp4",
        lambda video, *, reporter=None: mx.zeros((1, 128, 2, 1, 1), dtype=mx.float32),
        width=4,
        height=4,
        frames=9,
        compute_dtype=mx.float32,
    )
    assert tuple(latent.shape) == (1, 128, 2, 1, 1)


def test_reference_resize_uses_bottom_right_reflect_padding() -> None:
    source = mx.arange(2 * 3, dtype=mx.float32).reshape(1, 2, 3, 1)
    source = mx.broadcast_to(source, (1, 2, 3, 3))
    actual = resize_and_reflect_pad_sdr(source, width=5, height=3)
    expected = mx.take(
        mx.take(source, mx.array([0, 1, 0]), axis=1), mx.array([0, 1, 2, 1, 0]), axis=2
    )
    assert tuple(actual.shape) == (1, 3, 5, 3)
    assert mx.array_equal(actual, expected).item()


def test_reference_condition_appends_clean_tokens_positions_and_strength_mask() -> None:
    target = VideoLatentShape(1, 128, 2, 2, 2)
    tools = create_video_latent_tools(target, fps=24.0)
    state = init_video_latent_state(tools, dtype=mx.float32)
    reference = mx.arange(128 * 1 * 2 * 2, dtype=mx.float32).reshape(1, 128, 1, 2, 2)
    condition = VideoConditionByReferenceLatent(reference, strength=0.75)

    appended = condition.apply_to(state, tools)
    base_tokens = 8
    assert condition.token_count == 4
    assert tuple(appended.latent.shape) == (1, 12, 128)
    assert mx.array_equal(
        appended.latent[:, base_tokens:], appended.clean_latent[:, base_tokens:]
    ).item()
    assert mx.all(appended.denoise_mask[:, base_tokens:] == 0.25).item()
    assert tuple(appended.positions.shape) == (1, 3, 12, 2)
    assert not appended.uniform_mask

    cleared = tools.clear_conditioning(appended)
    assert tuple(cleared.latent.shape) == (1, base_tokens, 128)
    assert cleared.uniform_mask


def test_reference_condition_requires_same_spatial_latent_grid() -> None:
    tools = create_video_latent_tools(VideoLatentShape(1, 128, 2, 2, 2), fps=24.0)
    state = init_video_latent_state(tools, dtype=mx.float32)
    condition = VideoConditionByReferenceLatent(mx.zeros((1, 128, 1, 1, 2)))
    with pytest.raises(ValueError, match="incompatible"):
        condition.apply_to(state, tools)
