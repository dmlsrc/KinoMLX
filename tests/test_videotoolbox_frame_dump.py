"""Decoded-frame PNG dump behavior before the media terminal."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.media.frames import VideoFrameStream
from kinomlx.models.ltx2.signals import ltx23_sdr_signal
from kinomlx.videotoolbox.frame_dump import FrameDumpError, PNGFrameDumpStream


def _source(*frames: mx.array) -> VideoFrameStream:
    return VideoFrameStream(
        lambda: iter(frames),
        spec=ltx23_sdr_signal(width=2, height=2, fps=24.0),
        frame_count=len(frames),
    )


def test_png_frame_dump_maps_frames_and_writes_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kinomlx.videotoolbox import images

    first = mx.array(
        [
            [[0.0, 0.5, 1.0], [0.25, 0.75, 1.5]],
            [[-0.5, 0.0, 1.0], [0.25, 0.5, 0.75]],
        ],
        dtype=mx.float16,
    )
    second = mx.ones((2, 2, 3), dtype=mx.float16)
    source = _source(first, second)
    saved: dict[str, mx.array] = {}

    def save_image(image: mx.array, path: Path) -> Path:
        mx.eval(image)
        saved[path.name] = image
        return path

    monkeypatch.setattr(images, "save_image", save_image)
    directory = tmp_path / "vae_frames"
    stream = PNGFrameDumpStream(source, directory)

    dumped = list(stream)
    assert dumped[0] is first
    assert dumped[1] is second
    stream.commit()
    stream.close()

    assert source.closed
    assert set(saved) == {"frame_000000.png", "frame_000001.png"}
    assert saved["frame_000000.png"].dtype == mx.uint8
    assert saved["frame_000000.png"].tolist() == [
        [[0, 128, 255], [64, 191, 255]],
        [[0, 0, 255], [64, 128, 191]],
    ]
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["frame_count"] == 2
    assert manifest["format"] == "png"
    assert manifest["stored_dtype"] == "uint8"
    assert manifest["source_signal"]["dtype"] == "float16"
    assert manifest["source_signal"]["value_domain"] == "normalized-sdr"


def test_png_frame_dump_failure_removes_only_its_new_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kinomlx.videotoolbox import images

    source = _source(mx.zeros((2, 2, 3), dtype=mx.float16))
    directory = tmp_path / "vae_frames"

    def fail(_image: mx.array, _path: Path) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(images, "save_image", fail)
    stream = PNGFrameDumpStream(source, directory)
    with pytest.raises(FrameDumpError, match="disk full"):
        next(stream)

    assert source.closed
    assert not directory.exists()


def test_png_frame_dump_refuses_a_preexisting_path(tmp_path: Path) -> None:
    source = _source(mx.zeros((2, 2, 3), dtype=mx.float16))
    directory = tmp_path / "vae_frames"
    directory.mkdir()
    marker = directory / "keep.txt"
    marker.write_text("keep")

    with pytest.raises(FrameDumpError, match="cannot create"):
        PNGFrameDumpStream(source, directory)

    assert marker.read_text() == "keep"
    assert not source.closed
    source.close()
