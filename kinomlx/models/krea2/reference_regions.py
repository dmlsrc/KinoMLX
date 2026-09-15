"""Human-readable geometric selections on a reference token grid."""

import math
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class ReferenceRegion:
    kind: str
    coordinates: tuple[float, ...]


def parse_reference_region(value: str) -> ReferenceRegion:
    """Parse percentages (optional % suffix), with semicolon-separated polygon vertices."""
    kind, separator, body = value.strip().partition(":")
    kind = {"rectangle": "rect", "poly": "polygon"}.get(kind.lower(), kind.lower())
    if not separator or kind not in {"rect", "circle", "ellipse", "polygon"}:
        raise ValueError("reference region needs rect:, circle:, ellipse:, or polygon:")
    if kind == "polygon":
        vertices = body.split(";")
        if len(vertices) < 3 or any(len(vertex.split(",")) != 2 for vertex in vertices):
            raise ValueError("polygon needs at least three x,y pairs separated by semicolons")
    try:
        coordinates = tuple(
            float(part.strip().removesuffix("%")) / 100
            for part in body.replace(";", ",").split(",")
        )
    except ValueError as exc:
        raise ValueError(
            "reference region coordinates must be percentages, for example rect:35,15,65,55"
        ) from exc
    expected = {"rect": 4, "circle": 3, "ellipse": 4}.get(kind)
    if expected is not None and len(coordinates) != expected:
        raise ValueError(f"{kind} requires {expected} coordinates")
    if any(not math.isfinite(number) or not 0 <= number <= 1 for number in coordinates):
        raise ValueError("reference region percentages must be finite and between 0 and 100")
    if kind == "rect" and not (coordinates[0] < coordinates[2] and coordinates[1] < coordinates[3]):
        raise ValueError("rect requires left < right and top < bottom")
    if kind in {"circle", "ellipse"} and any(radius == 0 for radius in coordinates[2:]):
        raise ValueError("circle and ellipse radii must be positive")
    if kind == "polygon":
        points = list(zip(coordinates[::2], coordinates[1::2], strict=True))
        if points[-1] == points[0]:
            points.pop()
            coordinates = tuple(coordinate for point in points for coordinate in point)
        if len(set(points)) != len(points):
            raise ValueError(
                "polygon vertices must be distinct, except for an optional closing vertex"
            )
        if len(set(points)) < 3 or all(
            (x - points[0][0]) * (points[1][1] - points[0][1])
            == (y - points[0][1]) * (points[1][0] - points[0][0])
            for x, y in points[2:]
        ):
            raise ValueError("polygon requires at least three non-collinear vertices")
    return ReferenceRegion(kind, coordinates)


def reference_region_selection(
    regions: tuple[ReferenceRegion, ...], grid: tuple[int, int]
) -> mx.array:
    """Union shapes at token centers once; circles use the shorter image side."""
    height, width = grid
    if min(grid) <= 0:
        raise ValueError("reference region grid must be positive")
    x = (mx.arange(width, dtype=mx.float32)[None, :] + 0.5) / width
    y = (mx.arange(height, dtype=mx.float32)[:, None] + 0.5) / height
    selected = mx.zeros(grid, dtype=mx.bool_)
    for region in regions:
        c = region.coordinates
        if region.kind == "rect":
            inside = (x >= c[0]) & (x <= c[2]) & (y >= c[1]) & (y <= c[3])
        elif region.kind in {"circle", "ellipse"}:
            rx, ry = (
                (c[2] * min(grid) / width, c[2] * min(grid) / height)
                if region.kind == "circle"
                else c[2:]
            )
            inside = ((x - c[0]) / rx) ** 2 + ((y - c[1]) / ry) ** 2 <= 1
        else:
            points = list(zip(c[::2], c[1::2], strict=True))
            inside = mx.zeros(grid, dtype=mx.bool_)
            boundary = mx.zeros(grid, dtype=mx.bool_)
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1], strict=True):
                cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
                boundary = boundary | (
                    (mx.abs(cross) <= 1e-7)
                    & (x >= min(x1, x2))
                    & (x <= max(x1, x2))
                    & (y >= min(y1, y2))
                    & (y <= max(y1, y2))
                )
                if y1 != y2:
                    inside = inside ^ (
                        ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
                    )
            inside = inside | boundary
        selected = selected | inside
    return selected.reshape(-1)
