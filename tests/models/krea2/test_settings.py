"""Krea 2 data-only settings and environment contracts."""

from pathlib import Path

import pytest

from kinomlx.models.krea2.settings import OPTIMIZED_DENOISE_MODE, Krea2Settings


def test_krea_settings_read_namespaced_environment(monkeypatch) -> None:
    monkeypatch.setenv("KINO_KREA2_VARIANT", "raw")
    monkeypatch.setenv("KINO_KREA2_SOURCE_PATH", "/models/krea")
    monkeypatch.setenv("KINO_KREA2_TRANSFORMER_PATH", "/parts/transformer.safetensors")
    monkeypatch.setenv("KINO_KREA2_TRANSFORMER_DTYPE", "float16")
    monkeypatch.setenv("KINO_KREA2_VAE_DECODER", "upscale2x")
    monkeypatch.setenv("KINO_KREA2_VAE_DTYPE", "float32")

    settings = Krea2Settings.from_env()

    assert settings.variant == "raw"
    assert settings.source_path == Path("/models/krea")
    assert settings.transformer_path == Path("/parts/transformer.safetensors")
    assert settings.transformer_dtype == "float16"
    assert settings.vae_decoder == "upscale2x"
    assert settings.vae_dtype == "float32"


@pytest.mark.parametrize("variant", ["raw", "turbo"])
def test_krea_settings_accept_supported_variants(variant: str) -> None:
    Krea2Settings(variant=variant).validate()


def test_compact_denoising_is_default_without_changing_default_precision() -> None:
    settings = Krea2Settings()
    assert settings.reference_denoise is False
    assert settings.denoise_mode == OPTIMIZED_DENOISE_MODE
    assert settings.transformer_dtype == "bfloat16"


@pytest.mark.parametrize("variant", ["raw", "turbo"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_compact_denoising_accepts_all_transformer_dtypes(variant, dtype) -> None:
    Krea2Settings(variant=variant, transformer_dtype=dtype).validate()


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_reference_denoising_accepts_wider_exponent_range(dtype) -> None:
    settings = Krea2Settings(reference_denoise=True, transformer_dtype=dtype)
    settings.validate()
    assert settings.denoise_mode == "reference"


def test_reference_denoising_rejects_fp16_with_actionable_overflow_message() -> None:
    with pytest.raises(ValueError, match="padding can cause FP16 overflow") as error:
        Krea2Settings(reference_denoise=True, transformer_dtype="float16").validate()
    assert "Remove --reference-denoise" in str(error.value)
    assert "--transformer-dtype bfloat16" in str(error.value)


def test_reference_denoising_reads_namespaced_environment(monkeypatch) -> None:
    monkeypatch.setenv("KINO_KREA2_REFERENCE_DENOISE", "true")
    assert Krea2Settings.from_env().reference_denoise is True


def test_krea_settings_reject_invalid_variant_and_dtype() -> None:
    with pytest.raises(ValueError, match="variant: must be one of"):
        Krea2Settings(variant="fast")
    with pytest.raises(ValueError, match="transformer_dtype: expected one of"):
        Krea2Settings(transformer_dtype="float8")
    with pytest.raises(ValueError, match="vae_dtype: expected one of"):
        Krea2Settings(vae_dtype="float8")
    with pytest.raises(ValueError, match="vae_decoder: must be one of"):
        Krea2Settings(vae_decoder="sharpen")


def test_each_component_path_is_an_independent_override(tmp_path: Path) -> None:
    settings = Krea2Settings(
        source_path=tmp_path / "base",
        tokenizer_path=tmp_path / "tokenizer",
        text_encoder_path=tmp_path / "text",
        transformer_path=tmp_path / "transformer",
        vae_path=tmp_path / "vae",
    )
    assert settings.component_overrides == {
        "text_encoder": tmp_path / "text",
        "tokenizer": tmp_path / "tokenizer",
        "transformer": tmp_path / "transformer",
        "vae": tmp_path / "vae",
    }
