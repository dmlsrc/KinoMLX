"""Exact Klein token/latent layout, partial binding, and source-relative geometry."""

import mlx.core as mx
import pytest

from kinomlx.io.safetensors import save_weights
from kinomlx.models.flux2.image_processing import (
    output_size,
    pack_tokens,
    prepare_reference_pixels,
    reference_geometry,
    spatial_ids,
    unpack_tokens,
)
from kinomlx.models.flux2.loading import load_vae
from kinomlx.models.flux2.resources import inspect_resources
from kinomlx.models.flux2.settings import Flux2Settings
from kinomlx.models.flux2.text import prepare_text
from kinomlx.models.flux2.vae import Flux2VAE, patchify, unpatchify
from kinomlx.settings import Settings


class _Tokenizer:
    pad_token_id = 99

    def encode(self, text):
        assert text.startswith("<|im_start|>user\n")
        assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        return tuple(range(600 if "long prompt" in text else 20))


def test_text_padding_truncation_and_four_axis_positions():
    batch = prepare_text(_Tokenizer(), ("", "long prompt"))  # type: ignore[arg-type]
    assert batch.input_ids.shape == (2, 512)
    assert batch.input_ids[0, 20:].tolist() == [99] * 492
    assert batch.attention_mask.sum(axis=1).tolist() == [20, 512]
    assert batch.input_ids[1, -1].item() == 511
    assert batch.position_ids[0].tolist() == list(range(512))
    assert batch.text_ids[:, :, :3].sum().item() == 0
    assert batch.text_ids[1, :, 3].tolist() == list(range(512))


def test_packing_preserves_channel_then_patch_row_column_order():
    raw = mx.arange(32 * 4 * 6).reshape(1, 32, 4, 6)
    patches = patchify(raw)
    assert patches.shape == (1, 128, 2, 3)
    assert patches[0, :8, 0, 0].tolist() == [0, 1, 6, 7, 24, 25, 30, 31]
    assert mx.array_equal(unpatchify(patches), raw).item()
    tokens = pack_tokens(patches)
    assert tokens.shape == (1, 6, 128)
    assert mx.array_equal(tokens[0, 0], patches[0, :, 0, 0]).item()
    assert mx.array_equal(unpack_tokens(tokens, 2, 3), patches).item()


@pytest.mark.parametrize(
    ("source", "options", "expected"),
    [
        ((720, 1280), {}, (576, 1024)),
        ((720, 1280), {"width": 720}, (720, 1280)),
        ((720, 1280), {"width": 1024, "height": 768}, (1024, 768)),
        ((1280, 720), {"match_source_aspect": "short", "height": 512}, (912, 512)),
    ],
)
def test_single_reference_default_and_explicit_sizing(source, options, expected):
    assert output_size(source, reference_count=1, **options) == expected


def test_multi_reference_defaults_do_not_implicitly_match_first_source():
    assert output_size((720, 1280), reference_count=2) == (1024, 1024)
    geometry = reference_geometry((2048, 1024), (1024, 1024), reference_count=2)
    assert geometry.policy == "native"
    assert geometry.reference_size == (1440, 720)
    assert geometry.offset_hw == (0, 0)
    assert geometry.token_count == 4050


def test_fit_half_token_offsets_and_crop_are_distinct():
    fitted = reference_geometry((128, 64), (128, 112), reference_count=1)
    assert fitted.reference_size == (128, 64)
    assert fitted.offset_hw == (1.5, 0)
    ids = spatial_ids(4, 8, reference_index=1, offset_hw=fitted.offset_hw)
    assert ids[0, 0].tolist() == [20, 1.5, 0, 0]
    assert ids[0, -1].tolist() == [20, 4.5, 7, 0]
    cropped = reference_geometry((128, 64), (128, 112), reference_count=1, reference_fit="crop")
    assert cropped.reference_size == (128, 112)
    assert cropped.offset_hw == (0, 0)
    assert cropped.crop_box[2] - cropped.crop_box[0] < 128


def test_native_preprocessing_preserves_identity_grid_and_range():
    pixels = mx.arange(64 * 128 * 3, dtype=mx.float32).reshape(1, 64, 128, 3) / (64 * 128 * 3)
    geometry = reference_geometry((128, 64), (128, 64), reference_count=1)
    actual = prepare_reference_pixels(pixels, geometry)
    assert actual.shape == (1, 3, 64, 128)
    assert mx.array_equal(actual, (pixels * 2 - 1).transpose(0, 3, 1, 2)).item()


def test_invalid_reference_and_output_combinations_fail_early():
    with pytest.raises(ValueError, match="pins one"):
        output_size((128, 64), reference_count=1, width=128, height=64, match_source_aspect="long")
    with pytest.raises(ValueError, match="reference image"):
        output_size(None, reference_count=0, match_source_aspect="long")
    with pytest.raises(ValueError, match="sides"):
        reference_geometry((16, 128), (128, 128), reference_count=2)
    with pytest.raises(ValueError, match="matching source"):
        prepare_reference_pixels(
            mx.zeros((1, 64, 64, 3)), reference_geometry((128, 64), (128, 64), reference_count=1)
        )


def test_partial_vae_shells_refuse_unloaded_direction():
    encoder, decoder = Flux2VAE("encode"), Flux2VAE("decode")
    assert encoder.decoder is None
    assert encoder.post_quant_conv is None
    assert decoder.encoder is None
    assert decoder.quant_conv is None
    with pytest.raises(ValueError, match="decoder is not loaded"):
        encoder.decode(mx.zeros((1, 32, 8, 8)))
    with pytest.raises(ValueError, match="encoder is not loaded"):
        decoder.encode(mx.zeros((1, 3, 64, 64)))


def test_partial_loader_rejects_bad_shape_before_binding(tmp_path, monkeypatch):
    import kinomlx.modeling.loading as binding

    save_weights(
        tmp_path / "weights.safetensors",
        {"encoder.conv_in.weight": mx.zeros((1, 3, 3, 3), dtype=mx.bfloat16)},
    )

    def forbidden(_):
        pytest.fail("loader read tensor payload before structural preflight")

    monkeypatch.setattr(binding, "load_weights", forbidden)
    with pytest.raises(ValueError, match="expected"):
        load_vae((tmp_path / "weights.safetensors",), mode="encode")


def test_real_partial_vae_loads_preserve_encode_decode_contract():
    try:
        resources = inspect_resources(
            Flux2Settings(variant="kv", transformer_dtype="bfloat16"),
            infrastructure=Settings.from_env(),
        )
    except FileNotFoundError:
        pytest.skip("Klein sources are not cached")
    from kinomlx.models.flux2.loading import component_weight_files

    files = component_weight_files(resources.component("vae"))
    encoder = load_vae(files, mode="encode")
    pixels = mx.zeros((1, 3, 64, 96), dtype=mx.bfloat16)
    latents = encoder.encode_normalized(pixels)
    mx.eval(latents)
    assert latents.shape == (1, 128, 4, 6)
    assert mx.all(mx.isfinite(latents)).item()
    del encoder
    decoder = load_vae(files, mode="decode")
    output = decoder.decode_normalized(latents)
    mx.eval(output)
    assert output.shape == pixels.shape
    assert mx.all(mx.isfinite(output)).item()
