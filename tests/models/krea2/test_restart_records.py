"""Saved-run composition, input identity, and stage-aware sidecar selection."""

import json
import logging
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.config import ConfigError
from kinomlx.io.fingerprints import file_sha256
from kinomlx.model_registry import config_registry
from kinomlx.models.krea2.arguments import build_parser
from kinomlx.models.krea2.config import assemble, validate_for_execution
from kinomlx.models.krea2.output import plan_krea2_output
from kinomlx.models.krea2.pipelines.text_to_image import _negative_prompts
from kinomlx.models.krea2.restart import RestartConfig, load_restart_manifest
from kinomlx.models.krea2.settings import Krea2Settings
from kinomlx.settings import Settings
from tests.config.saved_records import saved_run_configuration


@pytest.fixture
def source_run(tmp_path: Path) -> Path:
    latent = tmp_path / "source_final_latents.safetensors"
    text = tmp_path / "source_text.safetensors"
    mx.save_safetensors(
        str(latent),
        {"final_latents": mx.zeros((1, 16, 2, 2), dtype=mx.bfloat16)},
        metadata={"schema": "kinomlx.krea2.tensor.v1", "artifact": "final_latents"},
    )
    mx.save_safetensors(
        str(text),
        {"placeholder": mx.zeros((1,), dtype=mx.float32)},
        metadata={
            "schema_version": "1",
            "artifact": "krea2_text_conditioning",
            "positive_prompts": '["fixture prompt"]',
            "negative_prompts": '["saved negative"]',
        },
    )
    run = tmp_path / "source_run.json"
    run.write_text(
        json.dumps(
            {
                "model": "krea2",
                "status": "completed",
                **saved_run_configuration(
                    "krea2",
                    {
                        "model": "krea2",
                        "generate": {"prompt": ["fixture prompt"], "width": 16, "height": 16},
                        "output": {},
                    },
                ),
                "outputs": {"final_latents": latent.name, "text_conditioning": text.name},
                "generation": {"variant": "turbo", "prompts": ["fixture prompt"]},
            }
        )
    )
    return run


def _invocation(run: Path, stage: str, *arguments: str):
    return assemble(
        build_parser().parse_args(
            ["--model", "krea2", "--restart", str(run), "--restart-from", stage, *arguments]
        ),
        base_settings=Settings(),
        base_model_settings=Krea2Settings(),
    )


@pytest.mark.parametrize(
    ("stage", "name"), [("decode", "final_latents"), ("denoise", "text_conditioning")]
)
@pytest.mark.parametrize("produced", [False, True])
def test_restart_prefers_output_then_immediate_consumed_input(
    source_run: Path, stage: str, name: str, produced: bool, caplog
) -> None:
    payload = json.loads(source_run.read_text())
    original = source_run.parent / payload["outputs"][name]
    consumed = original.with_name(f"consumed_{original.name}")
    consumed.write_bytes(original.read_bytes())
    payload["restart"] = {
        "inputs": {name: {"path": consumed.name, "sha256": file_sha256(consumed)}}
    }
    payload["configuration"]["values"]["restart"] = {"run": "unavailable_earlier_run.json"}
    if produced:
        payload["output_fingerprints"] = {name: file_sha256(original)}
    else:
        del payload["outputs"][name]
    source_run.write_text(json.dumps(payload))

    with caplog.at_level("INFO", logger="kinomlx.models.krea2.restart"):
        invocation = _invocation(source_run, stage)
    validate_for_execution(invocation)
    selected = original if produced else consumed
    receipt = invocation.restart.to_record()["inputs"][name]
    assert receipt["path"] == str(selected)
    assert receipt["matches_parent"] is True
    assert consumed in invocation.restart.source_outputs
    assert ("Restart uses recorded input" in caplog.text) is not produced


def test_explicit_latents_override_a_missing_consumed_input(source_run: Path) -> None:
    payload = json.loads(source_run.read_text())
    latent = source_run.parent / payload["outputs"].pop("final_latents")
    payload["restart"] = {
        "inputs": {"final_latents": {"path": "unavailable.safetensors", "sha256": "0" * 64}}
    }
    source_run.write_text(json.dumps(payload))

    manifest = load_restart_manifest(
        RestartConfig(source_run, latents=latent), schema=config_registry().model("krea2")
    )
    assert manifest.selected_latents == latent
    receipt = manifest.to_record()["inputs"]["final_latents"]
    assert receipt["matches_parent"] is False


def test_restart_protects_consumed_input_even_when_not_a_parent_output(source_run: Path) -> None:
    payload = json.loads(source_run.read_text())
    original = source_run.parent / payload["outputs"].pop("final_latents")
    consumed = original.with_name("consumed.png")
    original.rename(consumed)
    payload["restart"] = {"inputs": {"final_latents": {"path": consumed.name}}}
    source_run.write_text(json.dumps(payload))
    invocation = _invocation(source_run, "decode", "--output", str(consumed), "--force")

    with pytest.raises(ConfigError, match="must not overwrite restart source artifacts"):
        validate_for_execution(invocation)


@pytest.mark.parametrize("stage", ["decode", "denoise"])
@pytest.mark.parametrize("layer", ["file", "cli", "set"])
def test_save_all_selects_only_produced_sidecars_and_round_trips(
    source_run: Path, stage: str, layer: str
) -> None:
    if layer == "file":
        config = source_run.with_suffix(".toml")
        config.write_text("[output]\nsave_all_sidecars = true\n")
        arguments = ["--config", str(config)]
    elif layer == "cli":
        arguments = ["--save-all-sidecars"]
    else:
        arguments = ["--set", "output.save_all_sidecars=true"]
    invocation = _invocation(source_run, stage, *arguments)
    validate_for_execution(invocation)
    plan = plan_krea2_output(invocation.request, invocation.variant, invocation.output, stage=stage)
    assert bool(plan.artifacts.initial_noise) is (stage == "denoise")
    assert bool(plan.artifacts.text_conditioning) is (stage == "denoise")
    # Both restart stages re-emit the final latent a further decode needs.
    assert bool(plan.artifacts.final_latents) is True
    assert set(plan.sidecar_paths()) == {"run_log", "execution_log", "effective_config"}
    saved = source_run.with_name("effective.toml")
    saved.write_text(config_registry().model("krea2").dump_config(invocation.resolved_config))
    replay = _invocation(source_run, stage, "--config", str(saved))
    validate_for_execution(replay)
    assert replay.output == invocation.output


@pytest.mark.parametrize(
    ("stage", "field"),
    [
        ("decode", "save_initial_noise"),
        ("decode", "save_text_conditioning"),
    ],
)
@pytest.mark.parametrize("same_layer", [False, True])
def test_save_all_does_not_hide_explicit_unavailable_sidecars(
    source_run: Path, stage: str, field: str, same_layer: bool
) -> None:
    config = source_run.with_suffix(".toml")
    config.write_text(f"[output]\n{field} = true\n")
    arguments = ["--save-all-sidecars"]
    if same_layer:
        arguments.append(f"--{field.replace('_', '-')}")
    else:
        arguments.extend(["--config", str(config)])
    invocation = _invocation(source_run, stage, *arguments)

    with pytest.raises(ConfigError, match=field):
        validate_for_execution(invocation)


def test_save_all_respects_explicit_available_sidecar_opt_out(source_run: Path) -> None:
    invocation = _invocation(
        source_run, "denoise", "--save-all-sidecars", "--set", "output.save_initial_noise=false"
    )
    validate_for_execution(invocation)
    assert invocation.output.save_initial_noise is False
    assert invocation.output.save_final_latents is True


def test_denoise_zero_guidance_ignores_parent_negative_but_rejects_explicit_one(
    source_run: Path,
) -> None:
    payload = json.loads(source_run.read_text())
    payload["generation"]["variant"] = "raw"
    payload["configuration"]["values"]["generate"]["negative_prompt"] = ["saved negative"]
    source_run.write_text(json.dumps(payload))

    inherited = _invocation(source_run, "denoise", "--guidance", "0")
    validate_for_execution(inherited)
    assert inherited.request.negative_prompt is None
    assert inherited.request.negative_prompts == ("saved negative",)
    assert _negative_prompts(inherited.request, cfg_enabled=False) is None

    explicit = _invocation(
        source_run, "denoise", "--guidance", "0", "--negative-prompt", "saved negative"
    )
    with pytest.raises(ValueError, match="negative_prompt is unused"):
        _negative_prompts(explicit.request, cfg_enabled=False)


def test_denoise_replay_allows_an_explicit_unchanged_prompt(source_run: Path) -> None:
    payload = json.loads(source_run.read_text())
    payload["configuration"]["values"]["generate"].pop("prompt")
    source_run.write_text(json.dumps(payload))

    invocation = _invocation(source_run, "denoise", "--prompt", "fixture prompt")
    validate_for_execution(invocation)
    assert invocation.request.prompts == ("fixture prompt",)


@pytest.mark.parametrize("inherited", [False, True])
@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("unused_negative", [False, True])
def test_denoise_replacement_does_not_require_original_text(
    source_run: Path, inherited: bool, rows: int, unused_negative: bool
) -> None:
    payload = json.loads(source_run.read_text())
    original = source_run.parent / payload["outputs"]["text_conditioning"]
    replacement = original.with_name("replacement.safetensors")
    prompts = [f"fixture prompt {index}" for index in range(rows)]
    negatives = ["saved negative"] * rows
    mx.save_safetensors(
        str(replacement),
        {"placeholder": mx.zeros((1,), dtype=mx.float32)},
        metadata={
            "schema_version": "1",
            "artifact": "krea2_text_conditioning",
            "positive_prompts": json.dumps(prompts),
            "negative_prompts": json.dumps(negatives),
        },
    )
    payload["generation"].update(
        prompts=prompts, negative_prompts=None if unused_negative else negatives
    )
    payload["configuration"]["values"]["generate"]["prompt"] = None if inherited else prompts
    source_run.write_text(json.dumps(payload))
    original.unlink()

    invocation = _invocation(source_run, "denoise", "--text-conditioning", str(replacement))
    validate_for_execution(invocation)
    assert invocation.request.prompts == tuple(prompts)
    assert invocation.request.negative_prompts == tuple(negatives)
    assert invocation.request.image_count == rows

    with pytest.raises(ConfigError, match="belongs to an earlier stage"):
        _invocation(
            source_run,
            "denoise",
            "--text-conditioning",
            str(replacement),
            "--prompt",
            "changed prompt",
        )
    with pytest.raises(ConfigError, match="images_per_prompt"):
        _invocation(
            source_run,
            "denoise",
            "--text-conditioning",
            str(replacement),
            "--images-per-prompt",
            "2",
        )
    replacement.unlink()
    with pytest.raises(ConfigError, match=r"\[generate\].text_conditioning"):
        _invocation(source_run, "denoise", "--text-conditioning", str(replacement))


@pytest.fixture
def multi_seed_run(tmp_path: Path) -> Path:
    """A Krea run record shaped the way a two-seed run writes one."""
    names = ["source-001-seed42", "source-002-seed99"]
    for name in names:
        mx.save_safetensors(
            str(tmp_path / f"{name}_final_latents.safetensors"),
            {"final_latents": mx.zeros((1, 16, 2, 2), dtype=mx.bfloat16)},
            metadata={"schema": "kinomlx.krea2.tensor.v1", "artifact": "final_latents"},
        )
    run = tmp_path / "multi_seed_run.json"
    run.write_text(
        json.dumps(
            {
                "model": "krea2",
                "status": "completed",
                **saved_run_configuration(
                    "krea2",
                    {
                        "model": "krea2",
                        "generate": {
                            "prompt": ["fixture prompt"],
                            "seeds": [42, 99],
                            "width": 16,
                            "height": 16,
                        },
                        "output": {},
                    },
                ),
                "outputs": {
                    "images": [str(tmp_path / f"{name}.png") for name in names],
                    "final_latents": [
                        str(tmp_path / f"{name}_final_latents.safetensors") for name in names
                    ],
                },
                "generation": {
                    "variant": "turbo",
                    "prompts": ["fixture prompt", "fixture prompt"],
                    "seeds": [42, 99],
                    "noise": {"per_image_seeds": [42, 99]},
                },
            }
        )
    )
    return run


def _cancel_after(run: Path, published: int) -> Path:
    """Rewrite a completed multi-seed record as one canceled after N images."""
    payload = json.loads(run.read_text(encoding="utf-8"))
    flat = payload.pop("outputs")
    payload["status"] = "aborted"
    payload["planned_image_count"] = len(flat["images"])
    payload["outputs"] = {
        "images": flat["images"][:published],
        "published_artifacts": [
            *flat["images"][:published],
            *flat["final_latents"][:published],
        ],
        "seed_outputs": [
            {
                "index": index,
                "images": [flat["images"][index]],
                "final_latents": flat["final_latents"][index],
            }
            for index in range(published)
        ],
    }
    run.write_text(json.dumps(payload), encoding="utf-8")
    return run


@pytest.mark.parametrize("published", [0, 1, 2])
def test_a_canceled_run_is_restartable_only_for_what_it_published(
    multi_seed_run: Path, published: int
) -> None:
    run = _cancel_after(multi_seed_run, published)

    for index in range(published + 1, 3):
        with pytest.raises(
            ConfigError, match=f"image {index} of 2 was not published before cancellation"
        ):
            load_restart_manifest(
                RestartConfig(run=run, stage="decode", index=index),
                schema=config_registry().model("krea2"),
            )
    for index in range(1, published + 1):
        manifest = load_restart_manifest(
            RestartConfig(run=run, stage="decode", index=index),
            schema=config_registry().model("krea2"),
        )
        assert manifest.selected_latents.is_file()
        assert manifest.source_image_count == 2


def test_a_failed_multi_seed_run_keeps_its_artifacts_to_itself(multi_seed_run: Path) -> None:
    run = _cancel_after(multi_seed_run, 1)
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["status"] = "failed"
    run.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigError, match="has no completed final_latents output"):
        load_restart_manifest(
            RestartConfig(run=run, stage="decode", index=1), schema=config_registry().model("krea2")
        )


@pytest.mark.parametrize(("index", "seed"), [(1, 42), (2, 99)])
def test_multi_seed_restart_selects_one_recorded_image(multi_seed_run: Path, index, seed) -> None:
    manifest = load_restart_manifest(
        RestartConfig(run=multi_seed_run, stage="decode", index=index),
        schema=config_registry().model("krea2"),
    )
    assert (
        manifest.selected_latents.name == f"source-{index:03d}-seed{seed}_final_latents.safetensors"
    )
    generate = manifest.base_config().values["generate"]
    # The inherited seed list would otherwise re-expand the prompt batch.
    assert (generate["seed"], generate["seeds"], generate["images_per_prompt"]) == (seed, None, 1)


def test_multi_seed_restart_without_an_index_names_the_range(multi_seed_run: Path) -> None:
    with pytest.raises(ConfigError, match="recorded 2 images; select one"):
        load_restart_manifest(
            RestartConfig(run=multi_seed_run, stage="decode"),
            schema=config_registry().model("krea2"),
        )


def test_multi_seed_restart_rejects_an_index_past_the_image_count(multi_seed_run: Path) -> None:
    with pytest.raises(ConfigError, match="between 1 and 2"):
        load_restart_manifest(
            RestartConfig(run=multi_seed_run, stage="decode", index=3),
            schema=config_registry().model("krea2"),
        )


def test_multi_seed_restart_keeps_an_explicit_latents_override(multi_seed_run: Path) -> None:
    replacement = multi_seed_run.parent / "edited.safetensors"
    mx.save_safetensors(
        str(replacement),
        {"final_latents": mx.ones((1, 16, 2, 2), dtype=mx.bfloat16)},
        metadata={"schema": "kinomlx.krea2.tensor.v1", "artifact": "final_latents"},
    )
    manifest = load_restart_manifest(
        RestartConfig(run=multi_seed_run, stage="decode", index=2, latents=replacement),
        schema=config_registry().model("krea2"),
    )
    assert manifest.selected_latents == replacement
    assert manifest.base_config().values["generate"]["seed"] == 99


def test_single_image_restart_still_needs_no_index(source_run: Path) -> None:
    manifest = load_restart_manifest(
        RestartConfig(run=source_run, stage="decode"), schema=config_registry().model("krea2")
    )
    assert manifest.selected_latents.name == "source_final_latents.safetensors"
    assert "seeds" not in manifest.base_config().values["generate"]


def _canceled_multi_image_run(source_run: Path, *, record_planned_count: bool) -> Path:
    """Shape a record the way a two-image run canceled after its first image is keyed.

    Every digest such a run wrote carries the planned-count suffix, while the
    published artifact lists hold only the first image.
    """
    payload = json.loads(source_run.read_text())
    latent = source_run.parent / payload["outputs"]["final_latents"]
    text_conditioning = source_run.parent / payload["outputs"]["text_conditioning"]
    payload["outputs"] = {
        "images": [str(source_run.parent / "source-001.png")],
        "final_latents": [latent.name],
        "text_conditioning": [text_conditioning.name],
    }
    payload["output_fingerprints"] = {
        "final_latents.0": file_sha256(latent),
        "text_conditioning.0": file_sha256(text_conditioning),
    }
    payload["generation"]["prompts"] = ["fixture prompt", "second prompt"]
    if record_planned_count:
        payload["planned_image_count"] = 2
    source_run.write_text(json.dumps(payload))
    return source_run


def test_a_canceled_multi_image_run_keeps_its_recorded_identity(source_run: Path, caplog) -> None:
    run = _canceled_multi_image_run(source_run, record_planned_count=True)

    with caplog.at_level(logging.WARNING, logger="kinomlx"):
        manifest = load_restart_manifest(
            RestartConfig(run, index=1), schema=config_registry().model("krea2")
        )

    assert manifest.source_image_count == 2
    receipt = manifest.to_record()["inputs"]["final_latents"]
    assert receipt["matches_parent"] is True
    assert "differs from the parent receipt" not in caplog.text


def test_a_record_without_a_planned_count_still_loads_without_its_digest(
    source_run: Path,
) -> None:
    run = _canceled_multi_image_run(source_run, record_planned_count=False)

    manifest = load_restart_manifest(RestartConfig(run), schema=config_registry().model("krea2"))

    assert manifest.source_image_count == 1
    assert manifest.to_record()["inputs"]["final_latents"]["matches_parent"] is None
