"""The typed GMNet request/resources/recipe/runner public surface."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.components import ComponentLease
from kinomlx.models.gmnet.catalog import GMNetVariant, variant_spec
from kinomlx.models.gmnet.pipeline import expand_gmnet
from kinomlx.models.gmnet.resources import GMNetResources
from kinomlx.models.gmnet.runner import GMNetRunner
from kinomlx.models.gmnet.types import GMNetRequest
from kinomlx.reporting import RecordingReporter


class _FakeGMNet:
    def __call__(self, local, _thumbnail):
        batch, height, width, _channels = local.shape
        gain = mx.full((batch, height * 2, width * 2, 1), 0.5, dtype=mx.float32)
        qmax = mx.full((batch, 1, 1, 1), 0.75, dtype=mx.float32)
        return gain, gain, qmax


class _Components:
    def __init__(self) -> None:
        self.cleaned = False

    def generator(self, _resources):
        return ComponentLease(_FakeGMNet(), cleanup=self._cleanup)

    def _cleanup(self) -> None:
        self.cleaned = True


def _resources(tmp_path: Path) -> GMNetResources:
    return GMNetResources(
        weights_path=tmp_path / "unused.safetensors",
        spec=variant_spec(GMNetVariant.REALWORLD),
        mlx_cache_limit_bytes=None,
    )


def test_public_recipe_uses_injected_components_and_bounded_ownership(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(
        "kinomlx.io.image.load_image",
        lambda _path: mx.full((4, 6, 3), 0.5, dtype=mx.float32),
    )
    components = _Components()
    reporter = RecordingReporter()

    result = expand_gmnet(
        GMNetRequest(source),
        _resources(tmp_path),
        components=components,
        reporter=reporter,
    )

    assert result.linear_rgb.shape == (4, 6, 3)
    assert result.gain_map.shape == (4, 6)
    assert result.qmax_normalized == pytest.approx(0.75)
    assert components.cleaned
    assert [(kind, phase) for kind, phase, _details in reporter.events] == [
        ("start", "load SDR image"),
        ("advance", "load SDR image"),
        ("end", "load SDR image"),
        ("start", "GMNet expansion"),
        ("advance", "GMNet expansion"),
        ("end", "GMNet expansion"),
    ]


def test_runner_hosts_the_same_public_recipe(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(
        "kinomlx.io.image.load_image",
        lambda _path: mx.full((4, 4, 3), 0.25, dtype=mx.float32),
    )
    runner = GMNetRunner(resources=_resources(tmp_path), components=_Components())
    result = runner.expand(GMNetRequest(source))
    assert result.linear_rgb.shape == (4, 4, 3)


def test_runner_preserves_missing_input_as_file_not_found(tmp_path) -> None:
    runner = GMNetRunner(resources=_resources(tmp_path), components=_Components())
    with pytest.raises(FileNotFoundError):
        runner.expand(GMNetRequest(tmp_path / "missing.png"))
