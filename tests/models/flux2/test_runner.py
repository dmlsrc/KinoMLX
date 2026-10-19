"""Public Klein runner follows the shared host and saved-stage contracts."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from kinomlx.models.flux2 import (
    Flux2Error,
    Flux2GenerationOutput,
    Flux2Request,
    Flux2Restart,
    Flux2Runner,
    Flux2Settings,
    runner,
)
from kinomlx.models.flux2.catalog import profile
from kinomlx.models.flux2.generation import Flux2ImageResult
from kinomlx.models.flux2.pipelines import restart as restart_pipeline
from kinomlx.models.flux2.pipelines import text_to_image as pipeline
from kinomlx.models.flux2.restart import PreparedResources
from kinomlx.reporting import NullReporter

from .test_text_conditioning import conditioning


def resources():
    return PreparedResources(
        profile("distilled"),
        (
            SimpleNamespace(kind="transformer", dtype_policy="float16"),
            SimpleNamespace(kind="vae", dtype_policy="bfloat16"),
        ),
    )


def result():
    return Flux2ImageResult(mx.zeros((64, 64, 3)), mx.zeros((1, 16, 128)), 64, 64, ())


def test_settings_prepare_once_and_resources_never_reprepare(monkeypatch):
    prepared = resources()
    calls = []
    monkeypatch.setattr(
        runner, "prepare_resources", lambda *a, **k: calls.append("inspect") or prepared
    )
    host = Flux2Runner(Flux2Settings())
    assert host.resources is prepared
    assert len(calls) == 1
    assert Flux2Runner(resources=prepared).resources is prepared
    assert len(calls) == 1
    with pytest.raises(ValueError, match="settings or prepared resources"):
        Flux2Runner(Flux2Settings(), resources=prepared)


def test_generate_batches_seed_order_and_retains_no_outputs(monkeypatch):
    calls = []

    def generate(_resources, prompt, **kwargs):
        calls.append((prompt, kwargs["seed"], kwargs["reporter"]))
        yield kwargs.get("conditioning")
        yield None
        return result()

    monkeypatch.setattr(pipeline, "image_stages", generate)
    reporter = NullReporter()
    host = Flux2Runner(resources=resources(), reporter=reporter)
    output = host.generate(Flux2Request(prompt=("first", "second"), images_per_prompt=2, seed=10))
    assert len(output.images) == 4
    assert calls == [
        ("first", 10, reporter),
        ("first", 11, reporter),
        ("second", 10, reporter),
        ("second", 11, reporter),
    ]
    assert set(vars(host)) == {"resources", "reporter"}


def test_custom_recipe_and_operational_errors():
    host = Flux2Runner(resources=resources())
    output = Flux2GenerationOutput((result(),))
    assert host.run(lambda *a, **k: output, Flux2Request()) is output
    with pytest.raises(TypeError, match="must return Flux2GenerationOutput"):
        host.run(lambda *a, **k: object(), Flux2Request())

    def fail(*a, **k):
        raise RuntimeError("denoise failed")

    with pytest.raises(Flux2Error, match="generation failed.*denoise failed") as error:
        host.run(fail, Flux2Request())
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


def test_denoise_restart_loads_stage_and_rejects_context_change(monkeypatch):
    stage = conditioning()
    monkeypatch.setattr(pipeline, "load_conditioning", lambda *a: stage)
    calls = []

    def generate(*args, **kwargs):
        calls.append(kwargs)
        yield kwargs.get("conditioning")
        yield None
        return result()

    monkeypatch.setattr(pipeline, "image_stages", generate)
    host = Flux2Runner(resources=resources())
    restart = Flux2Restart(stage="denoise", text_conditioning=Path("text.safetensors"))
    host.restart(Flux2Request(), restart)
    assert calls[0]["conditioning"] is stage
    assert calls[0]["capture_stages"] is True
    with pytest.raises(Flux2Error, match="prompt does not match"):
        host.restart(Flux2Request(prompt="different"), restart)


def test_decode_restart_uses_only_vae_and_does_not_load_text(monkeypatch):
    prepared = replace(resources(), components=(resources().component("vae"),))
    latent = mx.zeros((1, 16, 128))
    monkeypatch.setattr(restart_pipeline, "load_final_latents", lambda path: (latent, 64, 64))
    monkeypatch.setattr(pipeline, "load_conditioning", lambda *a: pytest.fail("text reopened"))
    monkeypatch.setattr(restart_pipeline, "decode_latents", lambda *a, **k: mx.zeros((64, 64, 3)))
    host = Flux2Runner(resources=prepared)
    output = host.restart(
        Flux2Request(text_conditioning=Path("deleted.safetensors")),
        Flux2Restart(latents=Path("latents.safetensors")),
    )
    assert output.images[0].latents is latent
    with pytest.raises(Flux2Error, match="saved output grid"):
        host.restart(Flux2Request(width=128), Flux2Restart(latents=Path("latents.safetensors")))


@pytest.mark.parametrize("editing", [False, True])
def test_variations_reuse_conditioning_only_within_each_prompt(monkeypatch, editing):
    calls = []
    stages = []

    def generate(_resources, prompt, **kwargs):
        stage = kwargs["conditioning"]
        calls.append((prompt, stage, kwargs["images"], kwargs["reference_fit"]))
        if stage is None:
            stage = replace(conditioning(edit=editing), prompt=prompt)
            stages.append(stage)
        yield stage
        yield None
        return replace(result(), conditioning=stage)

    monkeypatch.setattr(pipeline, "image_stages", generate)
    images = (Path("reference.png"),) if editing else ()
    request = Flux2Request(prompt=("first", "second"), images_per_prompt=2, source_images=images)
    output = pipeline.generate_flux2(request, resources())
    assert len(output.images) == 4
    assert calls == [
        ("first", None, images, request.reference_fit),
        ("first", stages[0], (), None),
        ("second", None, images, request.reference_fit),
        ("second", stages[1], (), None),
    ]
