"""Opt-in header-only gates for current official Krea 2 Raw and Turbo sources."""

import pytest

from kinomlx.models.krea2.catalog import Krea2Variant, variant_spec
from kinomlx.models.krea2.resources import prepare_resources
from kinomlx.models.krea2.settings import Krea2Settings
from kinomlx.settings import Settings
from tests._snapshots import pinned_snapshot

from ._text_conditioning_fixtures import RAW_REVISION, TURBO_REVISION

_PINNED_REVISIONS = {
    Krea2Variant.RAW: RAW_REVISION,
    Krea2Variant.TURBO: TURBO_REVISION,
}


def _available_settings() -> Settings:
    for variant in Krea2Variant:
        pinned_snapshot(variant_spec(variant).repo_id, _PINNED_REVISIONS[variant])
    return Settings.from_env()


def test_current_raw_and_turbo_headers_share_only_the_expected_components() -> None:
    infrastructure = _available_settings()
    raw = prepare_resources(
        Krea2Settings(variant="raw"),
        infrastructure=infrastructure,
    )
    turbo = prepare_resources(
        Krea2Settings(variant="turbo"),
        infrastructure=infrastructure,
    )

    assert raw.revision == _PINNED_REVISIONS[Krea2Variant.RAW]
    assert turbo.revision == _PINNED_REVISIONS[Krea2Variant.TURBO]
    assert raw.text_encoder.tensor_count == turbo.text_encoder.tensor_count == 713
    assert raw.text_encoder.consumed_tensor_count == turbo.text_encoder.consumed_tensor_count == 386
    assert raw.transformer.tensor_count == turbo.transformer.tensor_count == 430
    assert raw.vae.tensor_count == turbo.vae.tensor_count == 194
    assert raw.vae.consumed_tensor_count == turbo.vae.consumed_tensor_count == 104
    assert raw.tokenizer.source_fingerprint == turbo.tokenizer.source_fingerprint
    assert raw.text_encoder.source_fingerprint == turbo.text_encoder.source_fingerprint
    assert raw.vae.source_fingerprint == turbo.vae.source_fingerprint
    assert raw.transformer.source_fingerprint != turbo.transformer.source_fingerprint
    assert raw.recipe_identity != turbo.recipe_identity


@pytest.mark.parametrize("variant", list(Krea2Variant), ids=lambda item: item.value)
def test_current_native_monolith_resolves_to_the_same_complete_target_graph(
    variant: Krea2Variant,
) -> None:
    infrastructure = _available_settings()
    spec = variant_spec(variant)
    root = pinned_snapshot(spec.repo_id, _PINNED_REVISIONS[variant])
    resources = prepare_resources(
        Krea2Settings(
            variant=variant.value,
            source_path=root,
            transformer_path=root / spec.monolith_filename,
        ),
        infrastructure=infrastructure,
    )

    assert resources.transformer.layout == "native-monolith"
    assert resources.transformer.tensor_count == 430
    assert resources.transformer.consumed_tensor_count == 430
