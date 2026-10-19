"""Klein restart selection, replay, and decode-stage source admission."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from kinomlx.io.fingerprints import file_sha256
from kinomlx.memory_policy import MLXMemoryPolicy
from kinomlx.model_registry import config_registry
from kinomlx.models.flux2 import components, generation
from kinomlx.models.flux2.catalog import Flux2Variant
from kinomlx.models.flux2.restart import Flux2RestartConfig, RestartManifest
from kinomlx.models.flux2.sampling import SamplingRecipe
from kinomlx.models.flux2.text_conditioning import (
    save_conditioning,
)
from tests.config.saved_records import saved_run_configuration
from tests.models.flux2.test_text_conditioning import conditioning


@pytest.mark.parametrize("dtype", ["float32", "float16", "bogus"])
def test_decode_restart_rejects_invalid_vae_dtype_before_source_io(tmp_path, dtype):
    manifest = RestartManifest(
        Flux2RestartConfig(run=tmp_path / "run.json"),
        {
            "generation": {
                "resources": {
                    "variant": "distilled",
                    "components": [
                        {"kind": "vae", "dtype": dtype, "source": str(tmp_path / "absent")}
                    ],
                }
            }
        },
    )
    with pytest.raises(ValueError, match="component receipt"):
        manifest.resources("distilled", stage="decode")


def test_decode_rejects_bad_stage_before_vae_load(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid final latents reached a model loader")

    monkeypatch.setattr(components, "load_vae", forbidden)
    for value in (mx.zeros((1, 15, 128)), mx.full((1, 16, 128), float("nan"))):
        with pytest.raises(ValueError, match="latents"):
            generation.decode_latents(tmp_path, value, 64, 64)


def test_replay_bypasses_text_and_reference_encoders(monkeypatch, tmp_path):
    value = conditioning(edit=True)
    resources = SimpleNamespace(
        memory_policy=MLXMemoryPolicy(),
        check_adapter_weights=True,
        profile=SimpleNamespace(variant=Flux2Variant.KV),
        component=lambda kind: SimpleNamespace(
            dtype_policy="float16",
            source_path=tmp_path / kind,
        ),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("replay attempted upstream encoding")

    monkeypatch.setattr(generation, "load_text_encoder", forbidden)
    monkeypatch.setattr(generation, "load_tokenizer", forbidden)
    monkeypatch.setattr(generation, "load_reference_pixels", forbidden)

    class Transformer:
        adapter_receipts = ()

    monkeypatch.setattr(components, "load_transformer_component", lambda *a, **k: Transformer())
    monkeypatch.setattr(components, "apply_flux2_adapters", lambda *a, **k: ())
    calls = []

    def denoise(model, noise, text, image_ids, text_ids, **kwargs):
        calls.append(kwargs)
        return noise

    monkeypatch.setattr(generation, "denoise", denoise)
    monkeypatch.setattr(generation, "decode_latents", lambda *a, **k: mx.zeros((64, 64, 3)))
    result = generation.generate_image(
        resources, value.prompt, conditioning=value, capture_stages=True
    )
    assert result.conditioning is not None
    assert result.initial_noise.shape == (1, 16, 128)
    assert result.sigmas.shape == (5,)
    assert calls[0]["references"].shape == value.references.shape
    assert "reference_kv" not in calls[0]


@pytest.mark.parametrize("edit", [False, True])
def test_empty_prompt_stage_replay_and_conflict(tmp_path, monkeypatch, edit):
    from kinomlx.models.flux2.pipelines import text_to_image as pipeline
    from kinomlx.models.flux2.types import Flux2Request

    recipe = SamplingRecipe(Flux2Variant.KV if edit else Flux2Variant.DISTILLED)
    value = replace(conditioning(edit=edit), prompt="")
    path = tmp_path / "empty.safetensors"
    save_conditioning(path, value, recipe)
    resources = SimpleNamespace(
        memory_policy=MLXMemoryPolicy(), profile=SimpleNamespace(variant=recipe.variant)
    )
    seen = []

    def generate(*args, **kwargs):
        seen.append((args, kwargs))
        yield value
        yield None
        return generation.Flux2ImageResult(
            mx.zeros((64, 64, 3)), mx.zeros((1, 16, 128)), 64, 64, (), conditioning=value
        )

    monkeypatch.setattr(pipeline, "image_stages", generate)
    for explicit in (False, True):
        pipeline.generate_flux2(
            Flux2Request(text_conditioning=path, prompt_unconditional=explicit),
            resources,
        )
        assert seen[-1][0][1] == ""
        assert seen[-1][1]["conditioning"].prompt == ""
    save_conditioning(path, conditioning(edit=edit), recipe)
    with pytest.raises(ValueError, match="prompt does not match"):
        pipeline.generate_flux2(
            Flux2Request(text_conditioning=path, prompt_unconditional=True),
            resources,
        )


def test_direct_restart_reopens_only_required_source_and_rejects_changes(tmp_path, monkeypatch):

    from kinomlx.models.flux2 import resources
    from kinomlx.models.flux2.catalog import profile
    from kinomlx.models.flux2.resources import Flux2Resources

    from .test_resources import _component

    transformer = _component(tmp_path)
    vae = _component(tmp_path, "vae")
    monkeypatch.setattr(resources, "validate_config", lambda *a: transformer.graph_identity)
    monkeypatch.setattr(resources, "_validate_weights", lambda *a: transformer.bindings)
    transformer = resources.weight_component(
        "transformer",
        transformer.source,
        dtype="float16",
    )
    vae = resources.weight_component("vae", vae.source, dtype="bfloat16")
    live = Flux2Resources(profile("distilled"), None, "test", "test", (transformer, vae))
    payload = {"generation": {"resources": live.describe()}}
    manifest = RestartManifest(Flux2RestartConfig(run=tmp_path / "run.json"), payload)
    restored = manifest.resources("distilled", stage="denoise")
    assert restored.component("transformer").identity == transformer.identity
    source = transformer.source.source_path / "config.json"
    source.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="source identity mismatch"):
        manifest.resources("distilled", stage="denoise")
    source.unlink()
    assert manifest.resources("distilled", stage="decode").component("vae").identity == vae.identity
    with pytest.raises(FileNotFoundError):
        manifest.resources("distilled", stage="denoise")

    (vae.source.source_path / "config.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="saved vae source identity mismatch"):
        manifest.resources("distilled", stage="decode")


def _multi_image_record(tmp_path, *, count=2, with_prompt=True):
    """A Klein run record shaped the way a multi-seed run writes one."""
    names = [f"run-{index + 1:03d}-seed{40 + index}" for index in range(count)]
    for name in names:
        (tmp_path / f"{name}_final_latents.safetensors").write_bytes(b"latents")
    images = [
        {"width": 64, "height": 64, "seed": 40 + index, "references": [], "adapters": []}
        for index in range(count)
    ]
    if with_prompt:
        for index, entry in enumerate(images):
            entry["prompt"] = f"prompt {index // 2}"
            entry["negative_prompt"] = ""
    return {
        "model": "flux2",
        "status": "completed",
        **saved_run_configuration(
            "flux2", {"generate": {"prompt": ["prompt 0"], "images_per_prompt": count}}
        ),
        "generation": {"seeds": [40 + index for index in range(count)], "images": images},
        "outputs": {
            "images": [str(tmp_path / f"{name}.png") for name in names],
            "final_latents": [
                str(tmp_path / f"{name}_final_latents.safetensors") for name in names
            ],
        },
        "output_fingerprints": {
            f"final_latents.{index}": file_sha256(
                tmp_path / f"{names[index]}_final_latents.safetensors"
            )
            for index in range(count)
        },
    }


@pytest.mark.parametrize("published", [0, 1])
def test_a_canceled_run_is_restartable_only_for_what_it_published(tmp_path, published) -> None:
    payload = _multi_image_record(tmp_path)
    payload["status"] = "aborted"
    payload["planned_image_count"] = 2
    # A canceled run records only the images it finished.
    for name in ("images", "final_latents"):
        payload["outputs"][name] = payload["outputs"][name][:published]
    run = tmp_path / "run.json"
    run.write_text(json.dumps(payload), encoding="utf-8")

    for index in range(published + 1, 3):
        with pytest.raises(
            ValueError, match=f"image {index} of 2 was not published before cancellation"
        ):
            RestartManifest.load(
                Flux2RestartConfig(run=run, index=index), schema=config_registry().model("flux2")
            )
    for index in range(1, published + 1):
        manifest = RestartManifest.load(
            Flux2RestartConfig(run=run, index=index), schema=config_registry().model("flux2")
        )
        assert manifest.artifact("final_latents").is_file()


def test_a_failed_run_is_not_offered_to_a_restart(tmp_path) -> None:
    payload = _multi_image_record(tmp_path)
    payload["status"] = "failed"
    run = tmp_path / "failed_run.json"
    run.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="completed or canceled KinoMLX Klein run record"):
        RestartManifest.load(
            Flux2RestartConfig(run=run, index=1), schema=config_registry().model("flux2")
        )


def test_multi_image_restart_selects_one_recorded_image(tmp_path) -> None:
    payload = _multi_image_record(tmp_path)
    for index, seed in ((1, 40), (2, 41)):
        manifest = RestartManifest(
            Flux2RestartConfig(run=tmp_path / "run.json", index=index), payload
        )
        assert manifest.source_image_count == 2
        assert manifest.artifact("final_latents").name.startswith(f"run-{index:03d}-")
        generate = manifest.base_config(schema=config_registry().model("flux2")).values["generate"]
        assert generate["seed"] == seed
        assert generate["seeds"] is None
        assert generate["images_per_prompt"] == 1


def test_multi_image_restart_without_an_index_names_the_range(tmp_path) -> None:
    manifest = RestartManifest(
        Flux2RestartConfig(run=tmp_path / "run.json"), _multi_image_record(tmp_path, count=3)
    )
    with pytest.raises(ValueError, match="recorded 3 images; select one"):
        manifest.artifact("final_latents")
    with pytest.raises(ValueError, match="recorded 3 images; select one"):
        manifest.base_config(schema=config_registry().model("flux2"))


def test_multi_image_restart_requires_current_per_image_prompt_metadata(tmp_path) -> None:
    payload = _multi_image_record(tmp_path, count=2, with_prompt=False)
    payload["configuration"]["values"]["generate"]["prompt"] = ["only prompt"]
    payload["generation"]["seeds"] = [40, 41]
    manifest = RestartManifest(Flux2RestartConfig(run=tmp_path / "run.json", index=2), payload)
    with pytest.raises(ValueError, match="requires recorded seed and prompt"):
        manifest.base_config(schema=config_registry().model("flux2"))


def test_single_image_restart_still_needs_no_index(tmp_path) -> None:
    payload = _multi_image_record(tmp_path, count=1)
    payload["outputs"]["final_latents"] = payload["outputs"]["final_latents"][0]
    payload["output_fingerprints"] = {
        "final_latents": file_sha256(Path(payload["outputs"]["final_latents"]))
    }
    manifest = RestartManifest(Flux2RestartConfig(run=tmp_path / "run.json"), payload)
    assert manifest.artifact("final_latents").exists()
    assert (
        manifest.base_config(schema=config_registry().model("flux2")).values["generate"]["seeds"]
        is None
    )


@pytest.mark.parametrize(
    ("stage", "suffix"), [("decode", "_final_decode"), ("denoise", "_denoise")]
)
def test_restart_names_its_bundle_after_the_source_run(tmp_path, stage, suffix) -> None:
    """Klein and Krea name a resumed bundle the same way: beside the source run."""
    payload = _multi_image_record(tmp_path, count=1)
    payload["outputs"]["final_latents"] = payload["outputs"]["final_latents"][0]
    payload["configuration"]["values"]["output"] = {
        "directory": str(tmp_path / "elsewhere"),
        "force": True,
    }
    run = tmp_path / "source_run.json"
    manifest = RestartManifest(Flux2RestartConfig(run=run, stage=stage), payload)
    output = manifest.base_config(schema=config_registry().model("flux2")).values["output"]
    assert output["directory"] == tmp_path
    assert output["prefix"] == f"source{suffix}"
    assert output["path"] is None
    assert output["force"] is False


def test_restart_components_are_the_kinds_of_the_stages_a_restart_runs():
    from kinomlx.models.flux2.resources import KINDS, STAGE_COMPONENTS, restart_components

    assert restart_components("denoise") == {"transformer", "vae"}
    assert restart_components("decode") == {"vae"}
    assert {kind for kinds in STAGE_COMPONENTS.values() for kind in kinds} == set(KINDS)
