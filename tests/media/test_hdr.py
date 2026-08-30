from __future__ import annotations

import mlx.core as mx
import pytest

from kinomlx.media.hdr import (
    acescct_to_scene_linear,
    convert_scene_linear_primaries,
    logc3_to_scene_linear,
    scene_linear_to_acescct,
    scene_linear_to_logc3,
)
from kinomlx.media.signals import ColorPrimaries


@pytest.mark.parametrize(
    ("code", "linear"),
    [
        (0.0729055341958355, 0.0),
        (0.4135884024924423, 0.18),
        (0.5547945205479452, 1.0),
        (0.7594156678493811, 12.0),
        (1.0, 222.8609442038076),
    ],
)
def test_acescct_matches_published_vectors(code: float, linear: float) -> None:
    decoded = float(acescct_to_scene_linear(mx.array(code)).item())
    encoded = float(scene_linear_to_acescct(mx.array(linear)).item())
    assert decoded == pytest.approx(linear, rel=2e-6, abs=2e-7)
    assert encoded == pytest.approx(code, rel=2e-6, abs=2e-7)


@pytest.mark.parametrize(
    ("code", "linear"),
    [
        (0.092809, 0.0),
        (0.39100683203408376, 0.18),
        (0.5706315581204173, 1.0),
        (0.8364731508120006, 12.0),
        (1.0, 55.079576698813185),
    ],
)
def test_logc3_matches_published_vectors(code: float, linear: float) -> None:
    decoded = float(logc3_to_scene_linear(mx.array(code)).item())
    encoded = float(scene_linear_to_logc3(mx.array(linear)).item())
    assert decoded == pytest.approx(linear, rel=6e-6, abs=3e-7)
    assert encoded == pytest.approx(code, rel=6e-6, abs=3e-7)


def test_working_transfers_bound_nonfinite_codes_without_clipping_linear_highlights() -> None:
    codes = mx.array([float("-inf"), float("nan"), 0.75, float("inf")])
    aces = acescct_to_scene_linear(codes)
    logc = logc3_to_scene_linear(codes)
    mx.eval(aces, logc)
    assert bool(mx.all(mx.isfinite(aces)).item())
    assert bool(mx.all(mx.isfinite(logc)).item())
    assert float(aces[2].item()) > 10.0
    assert float(logc[2].item()) > 4.0
    assert float(aces[3].item()) > 200.0
    assert float(logc[3].item()) > 50.0


def test_acescg_rec709_primary_round_trip_and_bt2020_neutral_axis() -> None:
    rgb = mx.array([[[0.2, 1.5, 4.0], [8.0, 0.1, 0.5]]], dtype=mx.float32)
    rec709 = convert_scene_linear_primaries(
        rgb,
        source=ColorPrimaries.ACESCG,
        target=ColorPrimaries.REC709,
    )
    round_trip = convert_scene_linear_primaries(
        rec709,
        source=ColorPrimaries.REC709,
        target=ColorPrimaries.ACESCG,
    )
    neutral = mx.array([[[4.0, 4.0, 4.0]]], dtype=mx.float32)
    bt2020 = convert_scene_linear_primaries(
        neutral,
        source=ColorPrimaries.ACESCG,
        target=ColorPrimaries.BT2020,
    )
    mx.eval(round_trip, bt2020)
    assert mx.allclose(round_trip, rgb, rtol=2e-5, atol=2e-5)
    assert mx.allclose(bt2020, neutral, rtol=2e-5, atol=2e-5)
