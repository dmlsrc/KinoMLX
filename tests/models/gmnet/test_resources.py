"""GMNet resource preparation respects infrastructure-owned storage."""

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
    app_cache_dir = tmp_path / "cache"
    expected = variant_weights_path(GMNetVariant.SYNTHETIC, app_cache_dir)
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"converted")

    resources = prepare_resources(
        GMNetSettings(variant=GMNetVariant.SYNTHETIC.value),
        infrastructure=Settings(app_cache_dir=app_cache_dir),
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
        infrastructure=Settings(app_cache_dir=tmp_path / "unused-cache"),
    )

    assert resources.weights_path == override.absolute()


def test_dry_run_report_names_the_resolved_checkpoint_and_its_contract(tmp_path):
    override = tmp_path / "custom.safetensors"
    override.write_bytes(b"converted")

    report = prepare_resources(
        GMNetSettings(variant=GMNetVariant.SYNTHETIC.value, weights_path=override),
        infrastructure=Settings(app_cache_dir=tmp_path / "unused-cache"),
    ).describe()

    # One checkpoint with no per-component records has no bindings, graph
    # identities or tensor accounting to report, so the keys are stated here
    # rather than compared against another generated structure.
    assert report == {
        "model": "gmnet",
        "variant": "synthetic",
        "weights": str(override.absolute()),
        "network_scale": 1,
        "peak_over_sdr_white": 8.0,
        "gain_stops": 3.0,
        "sdr_reference_white_nits": 100.0,
        "memory_policy": {"cache_limit_gb": 1.0, "wired_limit_gb": None},
    }
