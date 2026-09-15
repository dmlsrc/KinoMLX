"""Source-aspect output sizing: rounding, pinned dimensions, validation, and assembly."""

import logging
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.config import ConfigError
from kinomlx.config_rules import validate_record
from kinomlx.io.geometry import derive_output_size, pinned_axis
from kinomlx.io.image import save_image
from kinomlx.models.krea2.arguments import build_parser
from kinomlx.models.krea2.command import _log_request
from kinomlx.models.krea2.config import assemble
from kinomlx.models.krea2.sizing import (
    pinned_pixels,
    resolve_source_sizing,
)
from kinomlx.models.krea2.types import Krea2GenerateConfig, Krea2Request


def _portrait(tmp_path: Path, width: int = 90, height: int = 160) -> Path:
    path = tmp_path / "portrait.png"
    save_image(path, mx.full((height, width, 3), 0.5))
    return path


@pytest.mark.parametrize(
    ("source", "axis", "pixels", "expected"),
    [
        ((720, 1280), "height", 1024, (576, 1024)),
        ((720, 1280), "width", 1024, (1024, 1824)),
        ((1280, 720), "width", 1024, (1024, 576)),
        ((1000, 1000), "height", 1000, (1008, 1008)),
        ((90, 160), "width", 288, (288, 512)),
        ((4000, 100), "height", 16, (640, 16)),
        ((100, 4000), "width", 16, (16, 640)),
    ],
)
def test_derive_output_size_rounds_pinned_up_and_other_to_nearest(
    source, axis, pixels, expected
) -> None:
    assert derive_output_size(source, axis=axis, pixels=pixels) == expected


def test_derive_output_size_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        derive_output_size((0, 10), axis="width", pixels=64)
    with pytest.raises(ValueError, match="width or height"):
        derive_output_size((10, 10), axis="long", pixels=64)
    with pytest.raises(ValueError, match="positive"):
        derive_output_size((10, 10), axis="width", pixels=0)


def test_pinned_axis_maps_long_and_short_onto_the_source_orientation() -> None:
    assert pinned_axis("long", (720, 1280)) == "height"
    assert pinned_axis("short", (720, 1280)) == "width"
    assert pinned_axis("long", (1280, 720)) == "width"
    assert pinned_axis("short", (1280, 720)) == "height"
    assert pinned_axis("long", (512, 512)) == "height"
    assert pinned_axis("width", (720, 1280)) == "width"
    with pytest.raises(ValueError, match="must be one of"):
        pinned_axis("auto", (1, 1))


def test_pinned_pixels_takes_the_single_command_line_size() -> None:
    request = Krea2Request(
        prompt="t",
        width=768,
        height=512,
        source_images=(Path("s.png"),),
        match_source_aspect="long",
    )
    assert pinned_pixels(request, "long", frozenset()) == 768
    assert pinned_pixels(request, "short", frozenset()) == 512
    assert pinned_pixels(request, "long", frozenset({"height"})) == 512
    assert pinned_pixels(request, "short", frozenset({"width"})) == 768
    assert pinned_pixels(request, "width", frozenset({"width"})) == 768
    assert pinned_pixels(request, "height", frozenset()) == 512
    with pytest.raises(ValueError, match="not both"):
        pinned_pixels(request, "long", frozenset({"width", "height"}))
    with pytest.raises(ValueError, match="keeps --height; drop --width"):
        pinned_pixels(request, "height", frozenset({"width"}))


def test_request_validates_match_source_aspect() -> None:
    with pytest.raises(ValueError, match="requires source images"):
        validate_record(
            Krea2Request(prompt="test", match_source_aspect="long"), location="[generate]"
        )
    with pytest.raises(ValueError, match="must be one of"):
        Krea2Request(prompt="test", source_images=(Path("s.png"),), match_source_aspect="auto")
    request = Krea2Request(
        prompt="test", source_images=(Path("s.png"),), match_source_aspect="short"
    )
    assert request.match_source_aspect == "short"


def test_resolve_source_sizing_reads_the_image_and_is_idempotent(tmp_path: Path) -> None:
    source = _portrait(tmp_path)
    request = Krea2Request(prompt="test", source_images=(source,), match_source_aspect="long")
    sized, sizing = resolve_source_sizing(request)
    assert (sized.width, sized.height, sized.match_source_aspect) == (576, 1024, "long")
    assert sizing is not None
    assert (sizing.source_width, sizing.source_height, sizing.axis, sizing.pixels) == (
        90,
        160,
        "height",
        1024,
    )
    assert sizing.describe() == "576x1024 matched to source 90x160, long edge (height) 1024"
    again, again_sizing = resolve_source_sizing(sized)
    assert (again.width, again.height) == (576, 1024)
    assert again_sizing == sizing
    short, short_sizing = resolve_source_sizing(
        Krea2Request(prompt="test", source_images=(source,), match_source_aspect="short")
    )
    assert (short.width, short.height) == (1024, 1824)
    assert short_sizing is not None
    assert short_sizing.describe() == "1024x1824 matched to source 90x160, short edge (width) 1024"
    untouched = Krea2Request(prompt="test", source_images=(source,))
    assert resolve_source_sizing(untouched) == (untouched, None)


def test_resolve_source_sizing_uses_loaded_dimensions_when_given(tmp_path: Path) -> None:
    request = Krea2Request(
        prompt="test", source_images=(tmp_path / "missing.png",), match_source_aspect="width"
    )
    sized, sizing = resolve_source_sizing(request, source_size=(1280, 720), alignment=32)
    assert (sized.width, sized.height) == (1024, 576)
    assert sizing is not None
    assert sizing.to_metadata()["axis"] == "width"
    assert sizing.describe() == "1024x576 matched to source 1280x720, width 1024"
    with pytest.raises(FileNotFoundError):
        resolve_source_sizing(request)


def _assemble(*args: str):
    return assemble(build_parser().parse_args(list(args)))


def test_assemble_pins_the_named_dimension(tmp_path: Path, caplog) -> None:
    source = _portrait(tmp_path)
    base = ["--source-image", str(source), "--prompt", "test"]
    bare = _assemble(*base, "--match-source-aspect")
    assert bare.request is not None
    assert (bare.request.width, bare.request.height) == (576, 1024)
    assert bare.request.match_source_aspect == "long"
    assert (bare.generate.width, bare.generate.height) == (576, 1024)
    assert bare.resolved_config["generate"]["width"] == 576

    long_sized = _assemble(*base, "--match-source-aspect", "--width", "288")
    assert long_sized.request is not None
    assert (long_sized.request.width, long_sized.request.height) == (160, 288)

    short = _assemble(*base, "--match-source-aspect", "short", "--height", "512")
    assert short.request is not None
    assert (short.request.width, short.request.height) == (512, 912)

    by_width = _assemble(*base, "--match-source-aspect", "width", "--width", "288")
    assert by_width.request is not None
    assert (by_width.request.width, by_width.request.height) == (288, 512)

    with pytest.raises(ConfigError, match="keeps --height; drop --width"):
        _assemble(*base, "--match-source-aspect", "height", "--width", "288")
    with pytest.raises(ConfigError, match="not both"):
        _assemble(*base, "--match-source-aspect", "--width", "288", "--height", "512")
    with pytest.raises(ConfigError, match="Image not found"):
        _assemble(
            "--source-image", str(tmp_path / "gone.png"), "--prompt", "t", "--match-source-aspect"
        )

    with caplog.at_level(logging.INFO, logger="kinomlx.models.krea2.command"):
        _log_request(bare)
    assert "Resolution: 576x1024" in caplog.text
    assert "Output size: 576x1024 matched to source 90x160, long edge (height) 1024" in caplog.text


def test_generate_config_round_trips_match_source_aspect() -> None:
    request = Krea2GenerateConfig(
        prompt="test", source_images=(Path("s.png"),), match_source_aspect="short"
    ).request()
    assert request is not None
    assert request.match_source_aspect == "short"
    assert Krea2GenerateConfig(prompt="test").request().match_source_aspect is None
