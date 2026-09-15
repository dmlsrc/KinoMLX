"""Coordinate regions, token-grid geometry, and human-readable syntax."""

import mlx.core as mx
import pytest

from kinomlx.models.krea2.reference_regions import (
    parse_reference_region,
    reference_region_selection,
)


def selection(*regions: str, grid=(4, 4)) -> mx.array:
    return reference_region_selection(
        tuple(parse_reference_region(r) for r in regions), grid
    ).reshape(grid)


@pytest.mark.parametrize("syntax", ["rect:0,0,50,100", "rectangle: 0%, 0%, 50%, 100%"])
def test_rectangle_percentages(syntax) -> None:
    assert selection(syntax).tolist() == [[True, True, False, False]] * 4


def test_circle_uses_short_side_and_ellipse_uses_axis_radii() -> None:
    circle = selection("circle:50,50,50", grid=(2, 4))
    assert circle.tolist() == [[False, True, True, False]] * 2
    assert mx.array_equal(circle, selection("ellipse:50,50,25,50", grid=(2, 4))).item()
    assert selection("ellipse:50,50,50,50", grid=(2, 4)).tolist() == [[True] * 4] * 2


@pytest.mark.parametrize("vertices", ["0,0;100,0;0,100", "0,100;100,0;0,0", "0,0;100,0;0,100;0,0"])
def test_polygon_includes_boundary_and_accepts_either_winding(vertices) -> None:
    assert selection("polygon:" + vertices).tolist() == [
        [True, True, True, True],
        [True, True, True, False],
        [True, True, False, False],
        [True, False, False, False],
    ]


def test_concave_polygon_and_region_union() -> None:
    polygon = selection("polygon:0,0;100,0;100,50;50,50;50,100;0,100")
    rectangles = selection("rect:0,0,100,50", "rect:0,50,50,100")
    assert mx.array_equal(polygon, rectangles).item()
    assert mx.array_equal(
        selection("rect:0,0,50,100", "rect:50,0,100,100"), mx.ones((4, 4), dtype=mx.bool_)
    ).item()


@pytest.mark.parametrize(
    "region",
    [
        "face:50,50",
        "rect:0,0,50",
        "rect:50,0,0,100",
        "rect:0,0,101,100",
        "circle:50,50,0",
        "ellipse:50,50,-1,10",
        "circle:nan,50,10",
        "polygon:0,0;50,50",
        "polygon:0,0;50,50;100,100",
        "polygon:0,0;0,0;100,50;0,100",
    ],
)
def test_invalid_regions_fail_before_sampling(region) -> None:
    with pytest.raises(ValueError, match="reference region|rect|circle|ellipse|polygon"):
        parse_reference_region(region)


def test_region_cli_and_request_roundtrip() -> None:
    from pathlib import Path

    from kinomlx.models.krea2.arguments import build_parser
    from kinomlx.models.krea2.types import Krea2GenerateConfig

    regions = ("rect:0,0,50,100", "circle:75,50,10")
    args = build_parser().parse_args(
        ["--reference-boost-region", regions[0], "--reference-boost-region", regions[1]]
    )
    request = Krea2GenerateConfig(
        prompt="test",
        source_images=(Path("source.png"),),
        reference_boost_regions=tuple(args.reference_boost_regions),
        reference_strength=2,
    ).request()
    assert request.reference_boost_regions == regions
