"""Explicit HDR recipe facts remain metadata-driven and generation-specific."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from kinomlx.io.safetensors import save_weights
from kinomlx.media.signals import ColorTransfer
from kinomlx.models.ltx2.cache import LoRAAdapterReceipt
from kinomlx.models.ltx2.hdr_profile import (
    resolve_hdr_recipe,
    validate_hdr_adapter_placement,
)
from kinomlx.models.ltx2.types import (
    DistilledRequest,
    HDRReferenceConditioningConfig,
    ImageConditioningConfig,
)


def _resources(generation: str, *, native_hdr: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        capabilities=SimpleNamespace(
            model_generation=generation,
            native_hdr=native_hdr,
        )
    )


def _adapter(path: Path, **metadata: str) -> Path:
    save_weights(path, {"placeholder": mx.zeros((1,), dtype=mx.float32)}, metadata)
    return path


def _receipt(path: Path, *, placed: int, complete: int) -> LoRAAdapterReceipt:
    return LoRAAdapterReceipt(
        path=path,
        fingerprint="test",
        base_model_generation="2.3",
        declared_model_generation=None,
        generation_mismatch=None,
        strength=1.0,
        knockouts=(),
        complete_targets=complete,
        placed_targets=placed,
        structural_coverage=placed / complete if complete else 0.0,
        target_categories=(),
        skipped_reasons=(),
        warning=False,
    )


def test_ltx25_native_hdr_resolves_acescct_without_adapter() -> None:
    facts = resolve_hdr_recipe(
        DistilledRequest(
            prompt="test",
            width=64,
            height=64,
            frames=9,
            hdr="ACESCG",
            image=ImageConditioningConfig(Path("condition.exr")),
        ),
        _resources("2.5", native_hdr=True),
    )
    assert facts is not None
    assert facts.producer == "ltx25-native"
    assert facts.working_transfer is ColorTransfer.ACESCCT
    assert facts.semantic_anchor == "exr-condition"
    assert facts.adapter_path is None


def test_ltx25_native_hdr_rejects_unanchored_t2v_and_sdr_condition() -> None:
    resources = _resources("2.5", native_hdr=True)
    with pytest.raises(ValueError, match="image-to-video from an HDR EXR condition"):
        resolve_hdr_recipe(
            DistilledRequest(prompt="test", width=64, height=64, frames=9, hdr="ACESCG"),
            resources,
        )
    with pytest.raises(ValueError, match="SDR-only generation remain SDR"):
        resolve_hdr_recipe(
            DistilledRequest(
                prompt="test",
                width=64,
                height=64,
                frames=9,
                hdr="ACESCG",
                image=ImageConditioningConfig(Path("condition.png")),
            ),
            resources,
        )


def test_ltx23_hdr_uses_declared_facts_not_filename_or_hash(tmp_path: Path) -> None:
    adapter = _adapter(
        tmp_path / "community-name.safetensors",
        hdr_transform="logc3",
        reference_downscale_factor="1",
    )
    facts = resolve_hdr_recipe(
        DistilledRequest(
            prompt="test",
            width=64,
            height=64,
            frames=9,
            hdr="SRGB_LINEAR",
            lora_paths=(adapter,),
            hdr_reference=HDRReferenceConditioningConfig(Path("reference.mp4")),
        ),
        _resources("2.3"),
    )
    assert facts is not None
    assert facts.producer == "ltx23-hdr-ic-lora"
    assert facts.working_transfer is ColorTransfer.LOGC3
    assert facts.reference_downscale_factor == 1
    assert facts.stage_1_strength == 1.0
    assert facts.stage_2_strength == 1.0


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({}, "explicitly declares"),
        ({"hdr_transform": "acescct"}, "unsupported hdr_transform"),
        (
            {"hdr_transform": "logc3", "reference_downscale_factor": "2"},
            "reference_downscale_factor=1",
        ),
    ],
)
def test_ltx23_hdr_rejects_missing_or_unsupported_recipe_facts(
    tmp_path: Path,
    metadata: dict[str, str],
    message: str,
) -> None:
    adapter = _adapter(tmp_path / "adapter.safetensors", **metadata)
    request = DistilledRequest(
        prompt="test",
        width=64,
        height=64,
        frames=9,
        hdr="ACESCCT",
        lora_paths=(adapter,),
    )
    with pytest.raises(ValueError, match=message):
        resolve_hdr_recipe(request, _resources("2.3"))


def test_ltx23_hdr_rejects_text_only_until_real_generation_is_validated(
    tmp_path: Path,
) -> None:
    adapter = _adapter(
        tmp_path / "hdr.safetensors",
        hdr_transform="logc3",
        reference_downscale_factor="1",
    )
    request = DistilledRequest(
        prompt="test",
        width=64,
        height=64,
        frames=9,
        hdr="SRGB_LINEAR",
        lora_paths=(adapter,),
    )
    with pytest.raises(ValueError, match="text-only HDR is not supported"):
        resolve_hdr_recipe(request, _resources("2.3"))


def test_generic_companion_lora_does_not_change_hdr_profile_selection(tmp_path: Path) -> None:
    hdr = _adapter(
        tmp_path / "hdr.safetensors",
        hdr_transform="logc3",
        reference_downscale_factor="1",
    )
    style = _adapter(tmp_path / "style.safetensors")
    facts = resolve_hdr_recipe(
        DistilledRequest(
            prompt="test",
            width=64,
            height=64,
            frames=9,
            hdr="ACESCG",
            lora_paths=(style, hdr),
            hdr_reference=HDRReferenceConditioningConfig(Path("reference.mp4")),
        ),
        _resources("2.3"),
    )
    assert facts is not None
    assert facts.adapter_path == hdr


def test_hdr_adapter_requires_complete_structural_placement(tmp_path: Path) -> None:
    adapter = _adapter(
        tmp_path / "community-hdr.safetensors",
        hdr_transform="logc3",
        reference_downscale_factor="1",
    )
    facts = resolve_hdr_recipe(
        DistilledRequest(
            prompt="test",
            width=64,
            height=64,
            frames=9,
            hdr="ACESCG",
            lora_paths=(adapter,),
            hdr_reference=HDRReferenceConditioningConfig(Path("reference.mp4")),
        ),
        _resources("2.3"),
    )

    validate_hdr_adapter_placement(facts, (_receipt(adapter, placed=480, complete=480),), stage=1)
    with pytest.raises(ValueError, match="placed 479/480"):
        validate_hdr_adapter_placement(
            facts,
            (_receipt(adapter, placed=479, complete=480),),
            stage=2,
        )
