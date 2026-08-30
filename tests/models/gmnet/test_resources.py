"""GMNet resource preparation respects infrastructure-owned storage."""

from __future__ import annotations

from kinomlx.models.gmnet.catalog import GMNetVariant, variant_weights_path
from kinomlx.models.gmnet.resources import prepare_resources
from kinomlx.models.gmnet.settings import GMNetSettings
from kinomlx.settings import Settings


def test_installed_package_weights_resolve_under_the_configured_cache(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "kinomlx.models.gmnet.catalog._editable_checkout_weights_dir",
        lambda: None,
    )
    cache_dir = tmp_path / "cache"
    expected = variant_weights_path(GMNetVariant.SYNTHETIC, cache_dir)
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"converted")

    resources = prepare_resources(
        GMNetSettings(variant=GMNetVariant.SYNTHETIC.value),
        infrastructure=Settings(cache_dir=cache_dir),
    )

    assert resources.weights_path == expected.absolute()


def test_editable_checkout_keeps_converted_weights_beside_the_model(
    tmp_path,
    monkeypatch,
):
    checkout_weights = tmp_path / "checkout" / "kinomlx" / "models" / "gmnet" / "weights"
    monkeypatch.setattr(
        "kinomlx.models.gmnet.catalog._editable_checkout_weights_dir",
        lambda: checkout_weights,
    )
    expected = variant_weights_path(GMNetVariant.REALWORLD, tmp_path / "cache")

    assert expected == checkout_weights / "gmnet_realworld.safetensors"


def test_explicit_weights_override_still_wins(tmp_path):
    override = tmp_path / "custom.safetensors"
    override.write_bytes(b"converted")

    resources = prepare_resources(
        GMNetSettings(
            variant=GMNetVariant.REALWORLD.value,
            weights_path=override,
        ),
        infrastructure=Settings(cache_dir=tmp_path / "unused-cache"),
    )

    assert resources.weights_path == override.absolute()
