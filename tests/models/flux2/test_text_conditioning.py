"""Saved Klein conditioning contracts and stage reuse across seeds."""

from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import pytest

from kinomlx.memory_policy import MLXMemoryPolicy
from kinomlx.models.flux2 import components, generation
from kinomlx.models.flux2.catalog import Flux2Variant
from kinomlx.models.flux2.image_processing import reference_geometry, spatial_ids
from kinomlx.models.flux2.sampling import SamplingRecipe
from kinomlx.models.flux2.text_conditioning import (
    Flux2Conditioning,
    load_conditioning,
    save_conditioning,
)


def conditioning(*, base=False, edit=False, dtype=mx.float16):
    positions = mx.arange(512)
    zeros = mx.zeros_like(positions)
    geometry = reference_geometry((64, 64), (64, 64), reference_count=1)
    return Flux2Conditioning(
        mx.ones((1, 512, 12288), dtype=dtype),
        mx.stack([zeros, zeros, zeros, positions], axis=-1)[None],
        mx.zeros((1, 512, 12288), dtype=dtype) if base else None,
        mx.ones((1, 16, 128), dtype=dtype) if edit else None,
        spatial_ids(4, 4, reference_index=0) if edit else None,
        (geometry,) if edit else (),
        "a painted bowl",
        "",
        64,
        64,
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("variant", list(Flux2Variant))
@pytest.mark.parametrize("edit", [False, True])
def test_complete_stage_roundtrip(tmp_path, dtype, variant, edit):
    recipe = SamplingRecipe(variant)
    value = conditioning(base=variant == Flux2Variant.BASE, edit=edit, dtype=dtype)
    path = tmp_path / "conditioning.safetensors"
    save_conditioning(path, value, recipe)
    restored = load_conditioning(path, recipe)
    assert restored.geometries == value.geometries
    assert restored.prompt == value.prompt
    assert restored.text.dtype == dtype
    assert restored.tensors().keys() == value.tensors().keys()
    for key, tensor in value.tensors().items():
        assert mx.array_equal(tensor, restored.tensors()[key]).item()


@pytest.mark.parametrize(
    "change",
    [
        {"negative": None},
        {"reference_ids": None},
        {"geometries": ()},
        {"reference_ids": mx.zeros((1, 16, 4))},
        {"text_ids": mx.zeros((1, 512, 4))},
        {"width": 80},
        {"references": mx.zeros((1, 17, 128), dtype=mx.float16)},
    ],
)
def test_incomplete_or_inconsistent_edit_stage_rejected(change):
    with pytest.raises(
        ValueError, match="conditioning|Reference|reference|text positions|geometry|latents"
    ):
        replace(conditioning(base=True, edit=True), **change).validate(
            SamplingRecipe(Flux2Variant.BASE)
        )


def test_nonfinite_stage_rejected():
    value = conditioning()
    with pytest.raises(ValueError, match="finite"):
        replace(value, text=value.text * float("nan")).validate(SamplingRecipe())


@pytest.mark.parametrize(("mode", "loads"), [("sequential", 3), ("stage", 1), ("resident", 1)])
def test_seed_modes_load_once_per_selected_lifetime(monkeypatch, mode, loads):
    import weakref

    from kinomlx.models.flux2.pipelines import generate_flux2
    from kinomlx.models.flux2.types import Flux2Request

    value = conditioning()
    resources = SimpleNamespace(
        memory_policy=MLXMemoryPolicy(),
        check_adapter_weights=True,
        profile=SimpleNamespace(variant=Flux2Variant.DISTILLED),
        component=lambda kind: SimpleNamespace(dtype_policy="float16"),
    )
    events, references, noises = [], [], []

    class Transformer:
        adapter_receipts = ()

    class VAE:
        def decode_normalized(self, latent):
            assert latent.shape[0] == 1
            events.append("decode")
            return mx.zeros((1, 3, 64, 64))

    def transformer(*args, **kwargs):
        model = Transformer()
        events.append("load-transformer")
        references.append(weakref.ref(model))
        return model

    def vae(*args, **kwargs):
        model = VAE()
        events.append("load-vae")
        references.append(weakref.ref(model))
        return model

    def denoise(model, noise, *args, **kwargs):
        assert noise.shape[0] == 1
        events.append("denoise")
        noises.append(noise)
        return noise

    monkeypatch.setattr(components, "load_transformer_component", transformer)
    monkeypatch.setattr(components, "load_vae", vae)
    monkeypatch.setattr(components, "component_weight_files", lambda component: ())
    monkeypatch.setattr(components, "apply_flux2_adapters", lambda *a, **k: ())
    monkeypatch.setattr(generation, "denoise", denoise)
    output = generate_flux2(
        Flux2Request(prompt=value.prompt, seeds=(9, 2, 9), seed_mode=mode),
        resources,
        conditioning=value,
        on_image=lambda index, image: events.append(f"save-{index}"),
    )
    assert len(output.images) == 3
    assert events.count("load-transformer") == events.count("load-vae") == loads
    assert mx.array_equal(noises[0], noises[2]).item()
    assert not mx.array_equal(noises[0], noises[1]).item()
    assert all(reference() is None for reference in references)
    if mode == "stage":
        assert events[:4] == ["load-transformer", "denoise", "denoise", "denoise"]
    else:
        assert (
            events.index("save-0")
            < [index for index, event in enumerate(events) if event == "denoise"][1]
        )
