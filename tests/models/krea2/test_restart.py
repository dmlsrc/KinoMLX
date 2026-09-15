"""Krea 2 text-conditioning and final-latent restart contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

import kinomlx.models.krea2.pipelines.restart as restart_module
from kinomlx.components import ComponentLease
from kinomlx.config import ConfigError
from kinomlx.memory_policy import MLXMemoryPolicy
from kinomlx.models.krea2.arguments import build_parser
from kinomlx.models.krea2.catalog import Krea2VAEDecoder, Krea2Variant
from kinomlx.models.krea2.config import assemble, validate_for_execution
from kinomlx.models.krea2.pipelines.restart import load_final_latents, restart_krea2
from kinomlx.models.krea2.pipelines.text_to_image import Krea2GenerationOutput
from kinomlx.models.krea2.restart import Krea2Restart, RestartConfig, load_restart_manifest
from kinomlx.models.krea2.settings import Krea2Settings
from kinomlx.models.krea2.types import Krea2Request
from kinomlx.models.krea2.vae import Krea2DecodedRGB
from kinomlx.reporting import NullReporter
from kinomlx.settings import Settings
from tests.config.saved_records import saved_run_configuration


def _write_latents(
    path: Path,
    *,
    shape: tuple[int, ...] = (1, 16, 2, 2),
    value: float = 0.0,
) -> None:
    mx.save_safetensors(
        str(path),
        {"final_latents": mx.full(shape, value, dtype=mx.bfloat16)},
        metadata={
            "schema": "kinomlx.krea2.tensor.v1",
            "artifact": "final_latents",
        },
    )


def _write_run(
    path: Path,
    latent: Path,
    *,
    model: str = "krea2",
    variant: str = "turbo",
    text_conditioning: Path | None = None,
) -> None:
    payload = {
        "status": "completed",
        "model": model,
        **saved_run_configuration(
            "krea2",
            {
                "model": model,
                "generate": {
                    "prompt": ["fixture prompt"],
                    "width": 16,
                    "height": 16,
                    "seed": 7,
                    "steps": 8,
                    "guidance": 0.0,
                    "mu": 1.15,
                    "images_per_prompt": 1,
                    "noise_backend": "mlx",
                    "vae_tiling": "single",
                    "vae_tile_size": 256,
                    "vae_tile_stride": 192,
                },
                "model_settings": {
                    "transformer_dtype": "bfloat16",
                    "vae_dtype": "bfloat16",
                },
                "output": {
                    "save_all_sidecars": True,
                    "save_initial_noise": True,
                    "save_final_latents": True,
                    "save_run_log": True,
                    "save_console_log": True,
                    "save_effective_config": True,
                },
            },
        ),
        "outputs": {
            "final_latents": latent.name,
            **({} if text_conditioning is None else {"text_conditioning": text_conditioning.name}),
        },
        "generation": {"variant": variant, "decode": {"vae_decoder": "qwen"}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_text_conditioning(path: Path, *, prompt: str = "fixture prompt") -> None:
    mx.save_safetensors(
        str(path),
        {"placeholder": mx.zeros((1,), dtype=mx.float32)},
        metadata={
            "schema_version": "1",
            "artifact": "krea2_text_conditioning",
            "positive_prompts": json.dumps([prompt]),
            "negative_prompts": "null",
        },
    )


def _resources(
    *,
    decoder: Krea2VAEDecoder = Krea2VAEDecoder.QWEN,
) -> SimpleNamespace:
    return SimpleNamespace(
        memory_policy=MLXMemoryPolicy(),
        variant=Krea2Variant.TURBO,
        revision="fixture-revision",
        recipe_identity="fixture-recipe",
        capabilities=SimpleNamespace(
            spatial_compression=8,
            transformer_patch_size=2,
            latent_channels=16,
        ),
        vae=SimpleNamespace(dtype_policy="bfloat16"),
        vae_decoder=decoder,
        vae_config=SimpleNamespace(
            spatial_compression=8,
            base_dim=96,
            dim_mult=(1, 2, 4, 4),
            input_channels=3,
        ),
    )


class _Decoder:
    def __init__(self, events: list[object], *, output_scale: int) -> None:
        self.events = events
        self.output_scale = output_scale

    def __call__(self, latent, *, reporter=None, observer=None, tiling=None):
        del reporter, observer
        self.events.append(("decode", tuple(latent.shape), tiling))
        return Krea2DecodedRGB(
            mx.zeros(
                (
                    latent.shape[0],
                    latent.shape[2] * 8 * self.output_scale,
                    latent.shape[3] * 8 * self.output_scale,
                    3,
                ),
                dtype=latent.dtype,
            )
        )


class _DecoderOnlyComponents:
    def __init__(self, events: list[object]) -> None:
        self.events = events

    def image_decoder(self, resources):
        self.events.append("load-decoder")
        return ComponentLease(
            _Decoder(self.events, output_scale=resources.vae_decoder.output_scale),
            cleanup=lambda: self.events.append("close-decoder"),
        )

    def transformer(self, *_args, **_kwargs):
        raise AssertionError("decode restart must not request a transformer")


def test_restart_manifest_resolves_relative_latents_and_safe_output_base(tmp_path: Path) -> None:
    latent = tmp_path / "source_final_latents.safetensors"
    run = tmp_path / "source_run.json"
    _write_latents(latent)
    _write_run(run, latent)

    manifest = load_restart_manifest(RestartConfig(run))

    assert manifest.selected_latents == latent
    assert manifest.source_variant == "turbo"
    assert manifest.source_vae_decoder == "qwen"
    base = manifest.base_config().values
    assert base["model_settings"]["variant"] == "turbo"
    assert base["model_settings"]["vae_decoder"] == "qwen"
    assert base["output"] == {
        "save_all_sidecars": False,
        "save_initial_noise": False,
        # Decode re-emits the final latent it read, so the parent request survives.
        "save_final_latents": True,
        "save_text_conditioning": False,
        "save_run_log": True,
        "save_console_log": True,
        "save_effective_config": True,
        "path": None,
        "directory": tmp_path,
        "prefix": "source_final_decode",
    }
    assert manifest.to_record()["inputs"]["final_latents"]["path"] == str(latent)


def test_restart_config_inherits_request_and_allows_only_decode_overrides(tmp_path: Path) -> None:
    latent = tmp_path / "source_final_latents.safetensors"
    run = tmp_path / "source_run.json"
    _write_latents(latent)
    _write_run(run, latent)

    invocation = assemble(
        build_parser().parse_args(
            [
                "--model",
                "krea2",
                "--restart",
                str(run),
                "--vae-decoder",
                "upscale2x",
                "--vae-tiling",
                "custom",
                "--vae-tile-size",
                "128",
                "--vae-tile-stride",
                "96",
            ]
        ),
        base_settings=Settings(
            hf_home=tmp_path / "hf",
        ),
        base_model_settings=Krea2Settings(),
    )

    assert invocation.restart is not None
    assert invocation.request is not None
    assert invocation.request.prompts == ("fixture prompt",)
    assert (invocation.request.width, invocation.request.height, invocation.request.seed) == (
        16,
        16,
        7,
    )
    assert invocation.request.vae_tiling == "custom"
    assert invocation.model_settings.vae_decoder == "upscale2x"
    assert invocation.output.save_initial_noise is False
    assert invocation.output.save_final_latents is True
    assert invocation.output.save_run_log is True
    assert invocation.resolved_config["restart"] == {"run": run, "phase": "decode"}
    validate_for_execution(invocation)


def test_denoise_restart_inherits_saved_text_and_allows_downstream_changes(
    tmp_path: Path,
) -> None:
    latent = tmp_path / "source_final_latents.safetensors"
    text = tmp_path / "source_text.safetensors"
    run = tmp_path / "source_run.json"
    _write_latents(latent)
    _write_text_conditioning(text)
    _write_run(run, latent, text_conditioning=text)

    invocation = assemble(
        build_parser().parse_args(
            [
                "--model",
                "krea2",
                "--restart",
                str(run),
                "--restart-from",
                "denoise",
                "--seed",
                "99",
                "--width",
                "32",
                "--steps",
                "2",
            ]
        ),
        base_settings=Settings(),
        base_model_settings=Krea2Settings(),
    )

    assert invocation.restart is not None
    assert invocation.restart.config.phase == "denoise"
    assert invocation.request is not None
    assert invocation.request.text_conditioning == text
    assert invocation.request.prompts == ("fixture prompt",)
    assert (invocation.request.seed, invocation.request.width, invocation.request.steps) == (
        99,
        32,
        2,
    )
    assert invocation.output.prefix == "source_denoise"
    assert invocation.output.save_initial_noise is True
    assert invocation.output.save_final_latents is True
    # Denoise re-emits the saved text it read back, so the parent request survives.
    assert invocation.output.save_text_conditioning is True
    validate_for_execution(invocation)
    record = invocation.restart.to_record(text_conditioning=invocation.request.text_conditioning)
    assert record["phase"] == "denoise"
    assert record["latent_stage"] is None
    assert record["inputs"]["text_conditioning"]["path"] == str(text)


@pytest.mark.parametrize("prompts", [("first prompt", "second prompt"), ("", "")])
def test_decode_restart_of_replay_uses_recorded_expanded_batch(tmp_path: Path, prompts) -> None:
    latent = tmp_path / "replay_final_latents.safetensors"
    run = tmp_path / "replay_run.json"
    _write_latents(latent, shape=(2, 16, 2, 2))
    _write_run(run, latent)
    payload = json.loads(run.read_text())
    generate = payload["configuration"]["values"]["generate"]
    generate.pop("prompt")
    generate["text_conditioning"] = str(tmp_path / "deleted_text.safetensors")
    payload["generation"].update(prompts=list(prompts), negative_prompts=None)
    run.write_text(json.dumps(payload))

    invocation = assemble(
        build_parser().parse_args(["--model", "krea2", "--restart", str(run)]),
        base_settings=Settings(),
        base_model_settings=Krea2Settings(),
    )

    assert invocation.request is not None
    assert invocation.request.expanded_prompts() == prompts
    assert invocation.request.images_per_prompt == 1
    assert invocation.request.text_conditioning is None
    validate_for_execution(invocation)
    with restart_krea2(
        invocation.request,
        _resources(),
        restart=Krea2Restart.decode(latent),
        components=_DecoderOnlyComponents([]),
    ) as result:
        assert len(result.images) == 2


@pytest.mark.parametrize("layer", ["file", "cli", "set"])
@pytest.mark.parametrize("decoder", [None, "qwen", "upscale2x"])
def test_restart_explicit_vae_path_drops_only_inherited_decoder(
    tmp_path: Path, layer: str, decoder: str | None
) -> None:
    latent = tmp_path / "source_final_latents.safetensors"
    run = tmp_path / "source_run.json"
    replacement = tmp_path / "replacement.safetensors"
    _write_latents(latent)
    _write_run(run, latent)
    arguments = ["--model", "krea2", "--restart", str(run)]
    if layer == "file":
        config = tmp_path / "restart.toml"
        config.write_text(f"[model_settings]\nvae_path = {json.dumps(str(replacement))}\n")
        arguments.extend(["--config", str(config)])
    elif layer == "cli":
        arguments.extend(["--vae-path", str(replacement)])
    else:
        arguments.extend(["--set", f"model_settings.vae_path={json.dumps(str(replacement))}"])
    if decoder is not None:
        arguments.extend(["--vae-decoder", decoder])

    invocation = assemble(
        build_parser().parse_args(arguments),
        base_settings=Settings(),
        base_model_settings=Krea2Settings(),
    )

    assert invocation.model_settings.vae_path == replacement
    assert invocation.model_settings.vae_decoder == decoder


def test_denoise_restart_accepts_explicit_text_substitution(tmp_path: Path) -> None:
    latent = tmp_path / "source_final_latents.safetensors"
    replacement = tmp_path / "replacement_text.safetensors"
    run = tmp_path / "source_run.json"
    _write_latents(latent)
    _write_text_conditioning(replacement, prompt="replacement prompt")
    _write_run(run, latent)

    invocation = assemble(
        build_parser().parse_args(
            [
                "--model",
                "krea2",
                "--restart",
                str(run),
                "--restart-from",
                "denoise",
                "--text-conditioning",
                str(replacement),
            ]
        ),
        base_settings=Settings(),
        base_model_settings=Krea2Settings(),
    )

    assert invocation.request is not None
    assert invocation.request.text_conditioning == replacement
    validate_for_execution(invocation)


def test_denoise_restart_requires_saved_or_substituted_text(tmp_path: Path) -> None:
    latent = tmp_path / "source_final_latents.safetensors"
    run = tmp_path / "source_run.json"
    _write_latents(latent)
    _write_run(run, latent)
    invocation = assemble(
        build_parser().parse_args(
            ["--model", "krea2", "--restart", str(run), "--restart-from", "denoise"]
        ),
        base_settings=Settings(),
        base_model_settings=Krea2Settings(),
    )

    with pytest.raises(ConfigError, match="requires the parent text_conditioning"):
        validate_for_execution(invocation)


def test_denoise_restart_accepts_an_explicit_schedule_and_decode_refuses_it(
    tmp_path: Path,
) -> None:
    """Denoise reruns the sampler, so its schedule is an input of that station."""
    latent = tmp_path / "source_final_latents.safetensors"
    text = tmp_path / "source_text.safetensors"
    run = tmp_path / "source_run.json"
    _write_latents(latent)
    _write_text_conditioning(text)
    _write_run(run, latent, text_conditioning=text)

    invocation = assemble(
        build_parser().parse_args(
            [
                "--model",
                "krea2",
                "--restart",
                str(run),
                "--restart-from",
                "denoise",
                "--sigmas",
                "1,0.8,0.4,0",
            ]
        ),
        base_settings=Settings(),
        base_model_settings=Krea2Settings(),
    )
    assert invocation.request is not None
    assert invocation.request.sigmas == (1.0, 0.8, 0.4, 0.0)

    with pytest.raises(ConfigError, match=r"\[generate\]\.sigmas"):
        assemble(
            build_parser().parse_args(
                ["--model", "krea2", "--restart", str(run), "--sigmas", "1,0.8,0.4,0"]
            ),
            base_settings=Settings(),
            base_model_settings=Krea2Settings(),
        )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--prompt", "changed prompt"],
        ["--images-per-prompt", "2"],
    ],
)
def test_denoise_restart_rejects_changes_before_saved_text(
    tmp_path: Path,
    arguments: list[str],
) -> None:
    latent = tmp_path / "source_final_latents.safetensors"
    text = tmp_path / "source_text.safetensors"
    run = tmp_path / "source_run.json"
    _write_latents(latent)
    _write_text_conditioning(text)
    _write_run(run, latent, text_conditioning=text)

    with pytest.raises(ConfigError, match="belongs to an earlier station"):
        assemble(
            build_parser().parse_args(
                [
                    "--model",
                    "krea2",
                    "--restart",
                    str(run),
                    "--restart-from",
                    "denoise",
                    *arguments,
                ]
            ),
            base_settings=Settings(),
            base_model_settings=Krea2Settings(),
        )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--prompt", "changed prompt"],
        ["--seed", "8"],
        ["--width", "32"],
        ["--adapter", "different.safetensors"],
    ],
)
def test_restart_rejects_changes_before_decode(tmp_path: Path, arguments: list[str]) -> None:
    latent = tmp_path / "source_final_latents.safetensors"
    run = tmp_path / "source_run.json"
    _write_latents(latent)
    _write_run(run, latent)

    with pytest.raises(ConfigError, match="belongs to an earlier station"):
        assemble(
            build_parser().parse_args(["--model", "krea2", "--restart", str(run), *arguments]),
            base_settings=Settings(),
            base_model_settings=Krea2Settings(),
        )


def test_restart_refusal_names_the_station_and_quotes_the_declared_reason(
    tmp_path: Path,
) -> None:
    latent = tmp_path / "source_final_latents.safetensors"
    run = tmp_path / "source_run.json"
    _write_latents(latent)
    _write_run(run, latent)

    with pytest.raises(ConfigError) as refusal:
        assemble(
            build_parser().parse_args(
                ["--model", "krea2", "--restart", str(run), "--prompt", "changed prompt"]
            ),
            base_settings=Settings(),
            base_model_settings=Krea2Settings(),
        )

    assert str(refusal.value) == (
        "[restart]: decode restart: [generate].prompt belongs to an earlier station: "
        "Saved latents fix the resolved prompt rows."
    )


def test_restart_rejects_other_models_and_variants(tmp_path: Path) -> None:
    latent = tmp_path / "source_final_latents.safetensors"
    run = tmp_path / "source_run.json"
    _write_latents(latent)
    _write_run(run, latent, model="ltx2")
    with pytest.raises(ConfigError, match="source model 'ltx2'"):
        assemble(
            build_parser().parse_args(["--model", "krea2", "--restart", str(run)]),
            base_settings=Settings(),
            base_model_settings=Krea2Settings(),
        )

    _write_run(run, latent, variant="raw")
    inherited = assemble(
        build_parser().parse_args(["--model", "krea2", "--restart", str(run)]),
        base_settings=Settings(),
        base_model_settings=Krea2Settings(),
    )
    assert inherited.variant is Krea2Variant.RAW
    with pytest.raises(ConfigError, match="source variant 'raw'"):
        assemble(
            build_parser().parse_args(
                ["--model", "krea2", "--restart", str(run), "--variant", "turbo"]
            ),
            base_settings=Settings(),
            base_model_settings=Krea2Settings(),
        )


def test_restart_latent_override_replaces_missing_parent_artifact(tmp_path: Path) -> None:
    recorded = tmp_path / "recorded_final_latents.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    run = tmp_path / "source_run.json"
    _write_latents(replacement)
    _write_run(run, recorded)

    manifest = load_restart_manifest(RestartConfig(run, latents=replacement))

    assert manifest.selected_latents == replacement


def test_restart_refuses_to_replace_a_source_output_even_with_force(tmp_path: Path) -> None:
    latent = tmp_path / "source_final_latents.safetensors"
    source_image = tmp_path / "source.png"
    run = tmp_path / "source_run.json"
    _write_latents(latent)
    source_image.write_bytes(b"source image")
    _write_run(run, latent)
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["outputs"]["images"] = [str(source_image)]
    run.write_text(json.dumps(payload), encoding="utf-8")
    invocation = assemble(
        build_parser().parse_args(
            [
                "--model",
                "krea2",
                "--restart",
                str(run),
                "--output",
                str(source_image),
                "--force",
            ]
        ),
        base_settings=Settings(),
        base_model_settings=Krea2Settings(),
    )

    with pytest.raises(ConfigError, match="must not overwrite restart source artifacts"):
        validate_for_execution(invocation)


def test_decode_restart_loads_only_the_decoder_and_preserves_output_scale(tmp_path: Path) -> None:
    latent = tmp_path / "source_final_latents.safetensors"
    _write_latents(latent)
    events: list[object] = []

    with restart_krea2(
        Krea2Request("fixture prompt", width=16, height=16, seed=7, vae_tiling="single"),
        _resources(decoder=Krea2VAEDecoder.UPSCALE2X),
        restart=Krea2Restart(latent),
        components=_DecoderOnlyComponents(events),
    ) as output:
        assert tuple(output.images[0].pixels.shape) == (32, 32, 3)
        assert output.metadata["execution_mode"] == "restart-decode"
        assert output.metadata["decode"] == {
            "vae_decoder": "upscale2x",
            "output_scale": 2,
            "width": 32,
            "height": 32,
        }
        assert output.initial_noise is None
        # The decoded latent is retained so the station can re-emit it.
        assert output.final_latents is not None
        assert tuple(output.final_latents.shape) == (1, 16, 2, 2)

    assert events == [
        "load-decoder",
        ("decode", (1, 16, 2, 2), None),
        "close-decoder",
    ]


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((1, 8, 2, 2), "does not match"),
        ((2, 16, 2, 2), "does not match"),
    ],
)
def test_restart_rejects_incompatible_latent_shape_before_decode(
    tmp_path: Path,
    shape: tuple[int, ...],
    message: str,
) -> None:
    latent = tmp_path / "wrong.safetensors"
    _write_latents(latent, shape=shape)
    request = Krea2Request("fixture prompt", width=16, height=16, vae_tiling="single")
    resources = _resources()

    with pytest.raises(ValueError, match=message):
        load_final_latents(
            latent,
            request=request,
            resources=resources,
            geometry=SimpleNamespace(latent_height=2, latent_width=2),
            reporter=NullReporter(),
        )


def test_restart_rejects_nonfinite_latents_before_decode(tmp_path: Path) -> None:
    latent = tmp_path / "nonfinite.safetensors"
    _write_latents(latent, value=float("nan"))
    events: list[object] = []

    with pytest.raises(ValueError, match="must contain only finite values"):
        restart_krea2(
            Krea2Request("fixture prompt", width=16, height=16, vae_tiling="single"),
            _resources(),
            restart=Krea2Restart(latent),
            components=_DecoderOnlyComponents(events),
        )
    assert events == []


def test_denoise_restart_replays_saved_text_before_transformer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    text = tmp_path / "source_text.safetensors"
    _write_text_conditioning(text)
    request = Krea2Request("fixture prompt", width=16, height=16, steps=1)
    resources = SimpleNamespace(memory_policy=MLXMemoryPolicy())
    components = object()
    conditioner = object()
    calls = []

    def fake_generate(replay_request, replay_resources, **kwargs):
        calls.append((replay_request, replay_resources, kwargs))
        return Krea2GenerationOutput(images=(), metadata={"warnings": []})

    monkeypatch.setattr(restart_module, "generate_krea2", fake_generate)

    output = restart_krea2(
        request,
        resources,
        restart=Krea2Restart.denoise(text),
        components=components,
        text_conditioner=conditioner,
    )

    assert output.metadata["execution_mode"] == "restart-denoise"
    assert len(calls) == 1
    replay_request, replay_resources, kwargs = calls[0]
    assert replay_request.text_conditioning == text
    assert replay_resources is resources
    assert kwargs["components"] is components
    assert kwargs["text_conditioner"] is conditioner
    assert kwargs["text_replay_policy"] == "observe"


def test_replay_restart_honors_environment_negative_prompt(tmp_path, monkeypatch):
    latent, text, run = (
        tmp_path / name for name in ("latent.safetensors", "text.safetensors", "run.json")
    )
    _write_latents(latent)
    _write_text_conditioning(text)
    _write_run(run, latent, text_conditioning=text)
    args = ["--restart", str(run), "--restart-from", "denoise", "--text-conditioning", str(text)]
    with pytest.raises(ConfigError, match="negative_prompt") as cli_error:
        assemble(
            build_parser().parse_args([*args, "--negative-prompt", "blurry"]),
            base_settings=Settings(),
        )
    monkeypatch.setenv("KINO_KREA2_NEGATIVE_PROMPT", "blurry")
    with pytest.raises(ConfigError, match="negative_prompt") as env_error:
        assemble(build_parser().parse_args(args), base_settings=Settings())
    assert str(env_error.value) == str(cli_error.value)


@pytest.mark.parametrize("phase", ["denoise", "decode"])
@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("step_strengths", "[1,0]", (1.0, 0.0)),
        ("sigma_window", '"0.93"', "0.93"),
    ],
)
def test_restart_community_gate_changes_follow_station_policy(
    tmp_path, phase, field, value, expected
):
    latent = tmp_path / "latent.safetensors"
    text = tmp_path / "text.safetensors"
    run = tmp_path / "run.json"
    _write_latents(latent)
    _write_text_conditioning(text)
    _write_run(run, latent, text_conditioning=text)
    payload = json.loads(run.read_text())
    payload["configuration"]["values"]["generate"].update(
        adapters=[{"path": str(tmp_path / "style.safetensors")}], steps=2
    )
    run.write_text(json.dumps(payload))
    original = run.read_bytes()
    args = build_parser().parse_args(
        [
            "--restart",
            str(run),
            "--restart-from",
            phase,
            "--set",
            f"generate.adapters[0].{field}={value}",
        ]
    )
    if phase == "decode":
        with pytest.raises(ConfigError, match="adapters"):
            assemble(args, base_settings=Settings())
    else:
        invocation = assemble(args, base_settings=Settings())
        assert getattr(invocation.request.adapters[0], field) == expected
        assert invocation.request.text_conditioning == text
        assert invocation.request.prompts == ("fixture prompt",)
    assert run.read_bytes() == original


@pytest.mark.parametrize(
    ("controls", "error"),
    [
        (["--adapter-step-strengths", "1,0", "--adapter-sigma-window", "0.93"], "not both"),
        (["--adapter-step-strengths", "1,0,0"], "step"),
    ],
)
def test_denoise_restart_retains_community_schedule_validation(tmp_path, controls, error):
    latent = tmp_path / "latent.safetensors"
    text = tmp_path / "text.safetensors"
    run = tmp_path / "run.json"
    _write_latents(latent)
    _write_text_conditioning(text)
    _write_run(run, latent, text_conditioning=text)
    args = build_parser().parse_args(
        [
            "--restart",
            str(run),
            "--restart-from",
            "denoise",
            "--steps",
            "2",
            "--adapter",
            "style.safetensors",
            *controls,
        ]
    )
    if error == "not both":
        with pytest.raises(ConfigError, match=error):
            assemble(args, base_settings=Settings())
    else:
        from kinomlx.models.krea2.pipelines.text_to_image import _resolve_adapters

        from .test_generation import _resources as generation_resources

        invocation = assemble(args, base_settings=Settings())
        # Length is checked against the final recipe before adapter preparation,
        # not during configuration assembly. No adapter file exists in this test.
        with pytest.raises(ValueError, match="3 step strengths.*2 steps"):
            _resolve_adapters(
                invocation.request, generation_resources(Krea2Variant.TURBO), NullReporter()
            )


def test_missing_decoder_observation_does_not_invent_a_stock_decoder(tmp_path: Path) -> None:
    latent = tmp_path / "source_final_latents.safetensors"
    run = tmp_path / "source_run.json"
    _write_latents(latent)
    _write_run(run, latent)
    payload = json.loads(run.read_text())
    del payload["generation"]["decode"]
    payload["configuration"]["values"]["model_settings"]["vae_path"] = "alternate.safetensors"
    run.write_text(json.dumps(payload))

    manifest = load_restart_manifest(RestartConfig(run))
    assert manifest.source_vae_decoder is None
    settings = manifest.base_config().values["model_settings"]
    assert "vae_decoder" not in settings
    assert settings["vae_path"] == "alternate.safetensors"
