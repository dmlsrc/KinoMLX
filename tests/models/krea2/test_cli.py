"""Complete Krea 2 command configuration, sidecar, and failure behavior."""

import json
import logging
import tomllib
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.cli.main import main
from kinomlx.config import ConfigError
from kinomlx.io.fingerprints import file_sha256
from kinomlx.model_registry import config_registry
from kinomlx.models.krea2.arguments import build_parser
from kinomlx.models.krea2.command import _log_request
from kinomlx.models.krea2.config import assemble
from kinomlx.models.krea2.resources import prepare_resources
from kinomlx.models.krea2.restart import RestartConfig, load_restart_manifest
from kinomlx.models.krea2.settings import OPTIMIZED_DENOISE_MODE, Krea2Settings
from kinomlx.settings import Settings
from tests.config.saved_records import saved_run_configuration

from ._fixtures import make_snapshot


class _FakeReporter:
    def memory_checkpoint(self, _name: str) -> None:
        pass

    def memory_to_dict(self) -> dict[str, int]:
        return {"observed_run_peak_bytes": 1024}

    def to_dict(self) -> dict[str, object]:
        return {"total_seconds": 0.1, "phases": []}


@pytest.mark.parametrize("kind", ["missing", "directory", "permission"])
def test_krea_unreadable_text_conditioning_is_a_config_error(
    tmp_path: Path, monkeypatch, capsys, kind: str
) -> None:
    sidecar = tmp_path if kind == "directory" else tmp_path / "text.safetensors"
    if kind == "permission":

        def unreadable(_path):
            raise PermissionError("text-conditioning file is not readable")

        monkeypatch.setattr(
            "kinomlx.models.krea2.request_sources.read_text_conditioning_prompts", unreadable
        )

    assert main(["--model", "krea2", "--text-conditioning", str(sidecar), "--print-config"]) == 2
    captured = capsys.readouterr()
    assert "[generate].text_conditioning" in captured.err
    assert "Traceback" not in captured.err


def test_krea_request_logs_positive_and_negative_prompt_blocks(caplog) -> None:
    options = build_parser().parse_args(
        [
            "--model",
            "krea2",
            "--variant",
            "raw",
            "--prompt",
            "first",
            "--prompt",
            "second",
            "--negative-prompt",
            "avoid this",
        ]
    )
    invocation = assemble(options, base_settings=Settings())

    with caplog.at_level("INFO", logger="kinomlx.models.krea2.command"):
        _log_request(invocation)

    messages = [record.getMessage() for record in caplog.records]
    text = "\n".join(messages)
    assert "Run:" in messages
    assert "Model: Krea 2 Raw" in text
    assert "Action: generate" in text
    assert "Compute:" in messages
    assert "Prompt 1:\nfirst" in messages
    assert "Prompt 2:\nsecond" in messages
    assert "Negative prompt:\navoid this" in messages


def test_krea_print_config_preserves_intent_and_resolves_defaults_on_reload(
    tmp_path, capsys
) -> None:
    saved = tmp_path / "krea.toml"
    arguments = [
        "--model",
        "krea2",
        "--variant",
        "raw",
        "--prompt",
        "one",
        "--prompt",
        "two",
        "--negative-prompt",
        "none",
        "--width",
        "513",
        "--height",
        "769",
        "--seed",
        "42",
        "--noise-backend",
        "mlx",
        "--output",
        str(tmp_path / "image.png"),
    ]

    assert main([*arguments, "--print-config"]) == 0
    printed = tomllib.loads(capsys.readouterr().out)
    assert printed["generate"]["prompt"] == ["one", "two"]
    assert "steps" not in printed["generate"]
    assert "guidance" not in printed["generate"]
    assert "mu" not in printed["generate"]
    assert printed["generate"]["noise_backend"] == "mlx"

    assert main([*arguments, "--save-config", str(saved)]) == 0
    capsys.readouterr()
    assert main(["--config", str(saved), "--print-config"]) == 0
    assert tomllib.loads(capsys.readouterr().out) == printed
    replay = assemble(build_parser().parse_args(["--config", str(saved)]))
    assert replay.generate.steps == 52
    assert replay.generate.guidance == 3.5


def test_krea_saved_text_cli_supplies_prompts_without_prompt_flag(tmp_path: Path) -> None:
    sidecar = tmp_path / "prompt_text.safetensors"
    mx.save_safetensors(
        str(sidecar),
        {"placeholder": mx.zeros((1,), dtype=mx.float32)},
        metadata={
            "schema_version": "1",
            "artifact": "krea2_text_conditioning",
            "positive_prompts": '["first","second"]',
            "negative_prompts": "null",
        },
    )

    invocation = assemble(
        build_parser().parse_args(["--model", "krea2", "--text-conditioning", str(sidecar)]),
        base_settings=Settings(),
        base_model_settings=Krea2Settings(),
    )

    assert invocation.request is not None
    assert invocation.request.prompts == ("first", "second")
    assert invocation.request.text_conditioning == sidecar


def test_krea_cli_writes_execution_sidecars_before_generation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "image.png"

    def fake_run(_invocation, _reporter, *, plan, reservation, output_sink):
        assert reservation.active
        assert (tmp_path / "image_config.toml").is_file()
        assert (tmp_path / "image_console.log").is_file()
        assert (tmp_path / "image_run.json").is_file()
        for path in plan.artifacts.paths():
            path.write_bytes(b"artifact")
        return {
            "action": "generate",
            "outputs": plan.artifacts.to_dict(),
            "generation": {"recipe": {"steps": 8, "guidance": 0.0}},
            "output_fingerprints": {"final_latents": file_sha256(plan.artifacts.final_latents[0])},
            "output_fingerprint_errors": {},
            "warnings": ["adapter.safetensors: one target was not mapped"],
        }

    monkeypatch.setattr(
        "kinomlx.models.krea2.command.command_reporter",
        lambda _stack, _settings: _FakeReporter(),
    )
    monkeypatch.setattr("kinomlx.models.krea2.command._run_generate", fake_run)

    assert (
        main(
            [
                "--model",
                "krea2",
                "--prompt",
                "test",
                "--output",
                str(output),
                "--save-all-sidecars",
            ]
        )
        == 0
    )

    record = json.loads((tmp_path / "image_run.json").read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["output_fingerprints"] == {
        "final_latents": file_sha256(tmp_path / "image_final_latents.safetensors")
    }
    assert record["output_fingerprint_errors"] == {}
    assert record["generation"]["recipe"] == {"steps": 8, "guidance": 0.0}
    assert record["warnings"] == ["adapter.safetensors: one target was not mapped"]
    assert record["effective_config"]["generate"]["noise_backend"] == "mlx"
    assert record["diagnostics"]["memory"] == {"observed_run_peak_bytes": 1024}
    assert output.is_file()
    execution_log = (tmp_path / "image_console.log").read_text(encoding="utf-8")
    assert "Prompt:\ntest\n" in execution_log
    assert "Prompt 1:" not in execution_log
    assert "Run:" in execution_log
    assert "Model: Krea 2 Turbo" in execution_log
    assert "Planned outputs:" in execution_log
    assert "IMAGES 1:" in execution_log
    assert "IMAGE 001:" not in execution_log
    assert "Outputs:" in execution_log
    assert (tmp_path / "image_initial_noise.safetensors").is_file()
    assert (tmp_path / "image_final_latents.safetensors").is_file()
    assert (tmp_path / "image_text.safetensors").is_file()
    assert not list(tmp_path.glob(".*.kinomlx-reservation"))


def test_krea_json_surfaces_adapter_warnings(tmp_path: Path, monkeypatch, capsys) -> None:
    output = tmp_path / "image.png"

    def fake_run(_invocation, _reporter, *, plan, reservation, output_sink):
        assert reservation.active
        for path in plan.artifacts.paths():
            path.write_bytes(b"artifact")
        return {
            "action": "generate",
            "outputs": plan.artifacts.to_dict(),
            "generation": {"warnings": ["adapter.safetensors: one target was not mapped"]},
            "warnings": ["adapter.safetensors: one target was not mapped"],
        }

    monkeypatch.setattr(
        "kinomlx.models.krea2.command.command_reporter",
        lambda _stack, _settings: _FakeReporter(),
    )
    monkeypatch.setattr("kinomlx.models.krea2.command._run_generate", fake_run)

    assert (
        main(
            [
                "--model",
                "krea2",
                "--prompt",
                "test",
                "--output",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok"
    assert result["model"] == "krea2"
    assert result["action"] == "generate"
    assert result["outputs"] == {"images": [str(output)]}
    assert result["generation"] == {"warnings": ["adapter.safetensors: one target was not mapped"]}
    assert result["warnings"] == ["adapter.safetensors: one target was not mapped"]
    assert result["sidecar_errors"] == []
    assert result["diagnostics"]["memory"] == {"observed_run_peak_bytes": 1024}
    assert result["timings"] == {"total_seconds": 0.1, "phases": []}


def test_krea_cli_failure_finalizes_run_log_and_releases_reservations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "image.png"

    def fail(_invocation, _reporter, *, plan, reservation, output_sink):
        assert reservation.active
        raise RuntimeError("synthetic generation failure")

    monkeypatch.setattr(
        "kinomlx.models.krea2.command.command_reporter",
        lambda _stack, _settings: _FakeReporter(),
    )
    monkeypatch.setattr("kinomlx.models.krea2.command._run_generate", fail)

    assert (
        main(
            [
                "--model",
                "krea2",
                "--prompt",
                "test",
                "--output",
                str(output),
                "--save-all-sidecars",
            ]
        )
        == 2
    )

    record = json.loads((tmp_path / "image_run.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["error"] == "RuntimeError: synthetic generation failure"
    assert not output.exists()
    assert not (tmp_path / "image_initial_noise.safetensors").exists()
    assert not (tmp_path / "image_final_latents.safetensors").exists()
    assert not (tmp_path / "image_text.safetensors").exists()
    assert not list(tmp_path.glob(".*.kinomlx-reservation"))


def test_krea_cli_restart_records_parent_and_skips_generated_tensors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    latent = tmp_path / "source_final_latents.safetensors"
    latent.write_bytes(b"structural loading is replaced by the command fixture")
    source_run = tmp_path / "source_run.json"
    source_run.write_text(
        json.dumps(
            {
                "status": "completed",
                "model": "krea2",
                **saved_run_configuration(
                    "krea2",
                    {
                        "model": "krea2",
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
                        "output": {},
                    },
                ),
                "outputs": {"final_latents": str(latent)},
                "generation": {"variant": "turbo"},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "decoded.png"

    def fake_run(invocation, _reporter, *, plan, reservation, output_sink):
        assert invocation.restart is not None
        assert invocation.restart.selected_latents == latent
        assert reservation.active
        assert plan.artifacts.initial_noise == ()
        assert plan.artifacts.final_latents == ()
        output.write_bytes(b"decoded image")
        return {
            "action": "restart",
            "outputs": plan.artifacts.to_dict(),
            "generation": {"execution_mode": "restart-decode", "warnings": []},
            "warnings": [],
        }

    monkeypatch.setattr(
        "kinomlx.models.krea2.command.command_reporter",
        lambda _stack, _settings: _FakeReporter(),
    )
    monkeypatch.setattr("kinomlx.models.krea2.command._run_generate", fake_run)

    assert (
        main(
            [
                "--model",
                "krea2",
                "--restart",
                str(source_run),
                "--output",
                str(output),
                "--save-run-log",
            ]
        )
        == 0
    )

    record = json.loads((tmp_path / "decoded_run.json").read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["restart"]["stage"] == "decode"
    assert record["restart"]["source_run"] == str(source_run)
    assert record["restart"]["inputs"]["final_latents"]["path"] == str(latent)
    assert record["generation"]["execution_mode"] == "restart-decode"
    assert record["outputs"]["images"] == [str(output)]


def test_krea_cli_denoise_restart_records_saved_text_input(
    tmp_path: Path,
    monkeypatch,
) -> None:
    text = tmp_path / "source_text.safetensors"
    mx.save_safetensors(
        str(text),
        {"placeholder": mx.zeros((1,), dtype=mx.float32)},
        metadata={
            "schema_version": "1",
            "artifact": "krea2_text_conditioning",
            "positive_prompts": '["fixture prompt"]',
            "negative_prompts": "null",
        },
    )
    source_run = tmp_path / "source_run.json"
    source_run.write_text(
        json.dumps(
            {
                "status": "completed",
                "model": "krea2",
                **saved_run_configuration(
                    "krea2",
                    {
                        "model": "krea2",
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
                        "output": {},
                    },
                ),
                "outputs": {"text_conditioning": str(text)},
                "generation": {"variant": "turbo"},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "rerolled.png"

    def fake_run(invocation, _reporter, *, plan, reservation, output_sink):
        assert invocation.restart is not None
        assert invocation.restart.config.stage == "denoise"
        assert invocation.request.text_conditioning == text
        assert reservation.active
        assert plan.artifacts.text_conditioning == ()
        output.write_bytes(b"rerolled image")
        return {
            "action": "restart",
            "outputs": plan.artifacts.to_dict(),
            "generation": {"execution_mode": "restart-denoise", "warnings": []},
            "warnings": [],
        }

    monkeypatch.setattr(
        "kinomlx.models.krea2.command.command_reporter",
        lambda _stack, _settings: _FakeReporter(),
    )
    monkeypatch.setattr("kinomlx.models.krea2.command._run_generate", fake_run)

    assert (
        main(
            [
                "--model",
                "krea2",
                "--restart",
                str(source_run),
                "--restart-from",
                "denoise",
                "--output",
                str(output),
                "--save-run-log",
            ]
        )
        == 0
    )

    record = json.loads((tmp_path / "rerolled_run.json").read_text(encoding="utf-8"))
    assert record["restart"]["stage"] == "denoise"
    assert record["restart"]["inputs"]["text_conditioning"]["path"] == str(text)
    assert record["generation"]["execution_mode"] == "restart-denoise"


@pytest.mark.parametrize("mode", ["sequential", "stage", "resident"])
@pytest.mark.parametrize("failure", [None, KeyboardInterrupt, RuntimeError])
def test_per_seed_outputs_and_logs_survive_success_or_interrupt(
    tmp_path, monkeypatch, mode, failure
):
    from kinomlx.models.krea2 import command
    from kinomlx.models.krea2.catalog import Krea2Variant
    from kinomlx.models.krea2.runner import Krea2Runner
    from kinomlx.videotoolbox.images import load_image_rgb
    from tests.models.krea2.test_generation import _Components, _Conditioner, _Decoder, _resources

    events = []
    saved = {}
    count = 0
    original = _Decoder.__call__

    def decode(self, *args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            image = tmp_path / "result-001-seed9.png"
            assert load_image_rgb(image).shape == (16, 16, 3)
            sidecars = list(tmp_path.glob("result-001-seed9_*.safetensors"))
            assert len(sidecars) == 3
            for path in sidecars:
                assert mx.load(str(path))
            saved.update({path: file_sha256(path) for path in [image, *sidecars]})
            if failure is not None:
                raise failure("stop remaining seeds")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(_Decoder, "__call__", decode)
    monkeypatch.setattr(command, "_log_resources", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        command,
        "Krea2Runner",
        lambda *args, **kwargs: Krea2Runner(
            resources=_resources(Krea2Variant.TURBO),
            components=_Components(events),
            text_conditioner=_Conditioner(events),
            reporter=kwargs["reporter"],
        ),
    )
    assert main(
        [
            "--model",
            "krea2",
            "--prompt",
            "test",
            "--width",
            "16",
            "--height",
            "16",
            "--steps",
            "1",
            "--seed",
            "9",
            "2",
            "--seed-mode",
            mode,
            "--save-all-sidecars",
            "--output",
            str(tmp_path / "result.png"),
            "--quiet",
        ]
    ) == (0 if failure is None else 130 if failure is KeyboardInterrupt else 2)
    assert saved
    assert all(file_sha256(path) == digest for path, digest in saved.items())
    assert (tmp_path / "result-002-seed2.png").exists() is (failure is None)
    assert (tmp_path / "result_console.log").stat().st_size > 0
    record = json.loads((tmp_path / "result_run.json").read_text())
    assert record["status"] == (
        "completed" if failure is None else "aborted" if failure is KeyboardInterrupt else "failed"
    )
    # Recorded from the plan, so an interrupted run still states how many images
    # its per-image digest keys were built for.
    assert record["planned_image_count"] == 2
    expected = [str(tmp_path / "result-001-seed9.png")]
    if failure is None:
        expected.append(str(tmp_path / "result-002-seed2.png"))
        for name in ("initial_noise", "final_latents", "text_conditioning"):
            assert len(record["outputs"][name]) == 2
            assert all(Path(path).is_file() for path in record["outputs"][name])
    assert record["outputs"]["images"] == expected
    assert not list(tmp_path.glob("result_*.safetensors"))

    source = tmp_path / "result_run.json"
    if failure is KeyboardInterrupt:
        # A canceled run is restartable for the image it finished. Its digests
        # were keyed by the planned count, so the receipt still matches.
        manifest = load_restart_manifest(
            RestartConfig(run=source, stage="decode", index=1),
            schema=config_registry().model("krea2"),
        )
        assert manifest.selected_latents == (
            tmp_path / "result-001-seed9_final_latents.safetensors"
        )
        assert manifest.to_record()["inputs"]["final_latents"]["matches_parent"] is True
        assert manifest.base_config().values["generate"]["seed"] == 9
        # Image 2 was planned and never written.
        with pytest.raises(ConfigError, match="image 2 of 2 was not published before cancellation"):
            load_restart_manifest(
                RestartConfig(run=source, stage="decode", index=2),
                schema=config_registry().model("krea2"),
            )
    elif failure is RuntimeError:
        # A failed run stopped for a reason it cannot describe; nothing it left
        # behind is offered to a restart.
        with pytest.raises(ConfigError, match="no completed final_latents output"):
            load_restart_manifest(
                RestartConfig(run=source, stage="decode", index=1),
                schema=config_registry().model("krea2"),
            )


def test_krea_dry_run_reports_the_inventory_and_reserves_no_output(
    tmp_path: Path, capsys, caplog
) -> None:
    source = make_snapshot(tmp_path / "community", variant="turbo")
    outputs = tmp_path / "renders"
    arguments = [
        "--model",
        "krea2",
        "--dry-run",
        "--source-path",
        str(source),
        "--output-dir",
        str(outputs),
    ]

    with caplog.at_level(logging.INFO, logger="kinomlx"):
        assert main(arguments) == 0
    assert "Resources" in caplog.text
    assert capsys.readouterr().out == ""

    assert main([*arguments, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["model"] == "krea2"
    assert [entry["kind"] for entry in report["components"]] == [
        "tokenizer",
        "text_encoder",
        "transformer",
        "vae",
    ]
    assert not outputs.exists()


def test_krea_without_generation_input_asks_for_a_prompt_or_a_dry_run(caplog) -> None:
    """One resource-only entry point, named the way the other three name theirs.

    A prompt-less Krea invocation used to inspect resources, which made the
    flag and the omission two ways to reach one behavior. It is now the
    located configuration error Klein, LTX and GMNet already produced, and it
    names the flag that does inspect.
    """
    with caplog.at_level(logging.ERROR, logger="kinomlx"):
        exit_code = main(["--model", "krea2"])

    assert exit_code == 2
    assert "config error: Krea 2 generation needs --prompt" in caplog.text
    assert "--dry-run" in caplog.text
    assert "Traceback" not in caplog.text


def test_krea_dry_run_refuses_a_decoder_the_checkpoint_contradicts(tmp_path: Path, caplog) -> None:
    """A rule only checkable against the checkpoint is still the user's mistake.

    Krea detects its decoder from the VAE source, so this contradiction can
    only surface inside preparation. The dry run reports it as a configuration
    error rather than as a Krea failure.
    """
    source = make_snapshot(tmp_path / "community", variant="turbo")

    with caplog.at_level(logging.ERROR, logger="kinomlx"):
        exit_code = main(
            [
                "--model",
                "krea2",
                "--dry-run",
                "--source-path",
                str(source),
                "--vae-path",
                str(source / "vae"),
                "--vae-decoder",
                "upscale2x",
            ]
        )

    assert exit_code == 2
    assert "config error: [model_settings]:" in caplog.text
    assert "contradicts" in caplog.text
    assert "preparation failed" not in caplog.text


def test_krea_console_and_report_name_the_denoiser_the_same_way(tmp_path: Path) -> None:
    source = make_snapshot(tmp_path / "community", variant="turbo")
    resources = prepare_resources(
        Krea2Settings(source_path=source),
        infrastructure=Settings(hf_home=tmp_path / "hf"),
    )

    assert resources.denoise_mode == OPTIMIZED_DENOISE_MODE
    assert resources.describe()["denoise_mode"] == OPTIMIZED_DENOISE_MODE
    assert Krea2Settings().denoise_mode == resources.denoise_mode


@pytest.mark.parametrize(
    "generation",
    [
        ["--prompt", "a still"],
        ["--prompt-unconditional"],
        ["--text-conditioning", "saved.safetensors"],
    ],
)
def test_krea_dry_run_refuses_generation_input(generation: list[str]) -> None:
    with pytest.raises(ConfigError, match="Resource-only operations"):
        assemble(
            build_parser().parse_args(["--dry-run", *generation]),
            base_settings=Settings(),
        )
