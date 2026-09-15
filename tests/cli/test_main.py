"""CLI printing, error-boundary, and dispatch tests."""

import importlib
import json
import tomllib
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.adapters.selection import AdapterSelection
from kinomlx.artifacts import TensorArtifact
from kinomlx.cli.arguments import build_root_parser
from kinomlx.cli.invocation import (
    bootstrap_flag_present,
    bootstrap_json_from_arguments,
)
from kinomlx.cli.main import main
from kinomlx.cli.presentation import (
    format_elapsed,
    log_adapter_selections,
    log_output_paths,
    log_prompt_blocks,
    model_failure_message,
    render_model_error,
    render_run_success,
)
from kinomlx.debug import SidecarError
from kinomlx.media.frames import VideoFrameStream
from kinomlx.models.ltx2.adapters.selection import LTX2AdapterSelection
from kinomlx.models.ltx2.artifacts import (
    STAGE_1_CONDITIONING,
    STAGE_2_CONDITIONING,
    distilled_stage_latents_artifact,
    text_conditioning_artifact,
)
from kinomlx.models.ltx2.generation import GenerationOutput
from kinomlx.models.ltx2.run_logging import _stage_strength_details
from kinomlx.models.ltx2.runner import LTX2Error
from kinomlx.models.ltx2.signals import ltx23_sdr_signal
from kinomlx.settings import Settings


def _frame_stream() -> VideoFrameStream:
    signal = ltx23_sdr_signal(width=64, height=64, fps=24.0)
    return VideoFrameStream(
        lambda: iter((mx.zeros((64, 64, 3), dtype=mx.float16),)),
        spec=signal,
        frame_count=1,
    )


def _generation_output(
    *,
    audio_waveform=None,
    audio_sample_rate: int | None = None,
    metadata: dict[str, object] | None = None,
) -> GenerationOutput:
    return GenerationOutput(
        frames=_frame_stream(),
        audio_waveform=audio_waveform,
        audio_sample_rate=audio_sample_rate,
        metadata={} if metadata is None else metadata,
    )


def _inject_operational_failure(monkeypatch) -> None:
    import kinomlx.models.ltx2.command as main_module

    class _Runner:
        def run(self, _recipe, _config):
            raise LTX2Error("injected model failure")

    monkeypatch.setattr(
        main_module,
        "create_runner",
        lambda *_args, **_kwargs: _Runner(),
    )


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (-1.0, "0.0s"),
        (59.94, "59.9s"),
        (59.96, "1m 0.0s"),
        (3599.96, "1h 0m 0.0s"),
        (3661.24, "1h 1m 1.2s"),
    ],
)
def test_format_elapsed_carries_rounded_seconds(seconds: float, expected: str) -> None:
    assert format_elapsed(seconds) == expected


def test_bootstrap_flag_scans_stop_at_the_option_terminator() -> None:
    assert not bootstrap_flag_present(["--", "--help"], {"-h", "--help"})
    assert not bootstrap_json_from_arguments(["--", "--json"], Settings())


def test_prompt_blocks_put_exact_bodies_on_copyable_lines(caplog) -> None:
    import logging

    logger = logging.getLogger("kinomlx.test.prompt_blocks")
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_prompt_blocks(logger, ("first line\nsecond line",))
        log_prompt_blocks(logger, ("avoid this", "avoid that"), label="Negative prompt")

    assert [record.getMessage() for record in caplog.records] == [
        "Prompt:\nfirst line\nsecond line",
        "Negative prompt 1:\navoid this",
        "Negative prompt 2:\navoid that",
    ]


def test_common_human_run_rendering_is_rich_and_consistent(caplog, tmp_path: Path) -> None:
    import logging

    logger = logging.getLogger("kinomlx.test.run_rendering")
    adapter = AdapterSelection(
        tmp_path / "style.safetensors",
        0.75,
        exclude=("audio",),
    )
    outputs = {
        "images": [tmp_path / "one.png", tmp_path / "two.png"],
        "run_log": tmp_path / "run.json",
    }
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_adapter_selections(
            logger,
            (adapter,),
            strength_details=lambda _selection: ("profile strength=0.5",),
        )
        log_output_paths(logger, outputs, heading="Planned outputs")
        payload = render_run_success(
            logger,
            model="krea2",
            action="generate",
            outputs=outputs,
            elapsed_seconds=12.5,
            timings={"total_seconds": 12.0, "phases": []},
            diagnostics={"memory": {"observed_run_peak_bytes": 1024}},
            json_output=False,
            generation={"seed": 42},
            warnings=("warning",),
            sidecar_errors=(),
        )

    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "Adapters:" in text
    assert "profile strength=0.5" in text
    assert "strength=0.75" not in text
    assert "exclude=audio" in text
    assert "Planned outputs:" in text
    assert "IMAGES 1:" in text
    assert "RUN LOG:" in text
    assert "Outputs:" in text
    assert "Total runtime: 12.5s" in text
    assert payload["outputs"] == outputs
    assert payload["warnings"] == ["warning"]


@pytest.mark.parametrize(
    ("stage_1", "stage_2", "expected"),
    [
        (0.25, -0.5, ("stage 1 strength=0.25", "stage 2 strength=-0.5")),
        (None, 0.5, ("stage 1 strength=0.75", "stage 2 strength=0.5")),
        (0, None, ("stage 1 strength=0", "stage 2 strength=0.75")),
        (None, None, ("strength=0.75",)),
    ],
)
def test_ltx_adapter_strength_details_name_what_each_stage_applies(
    tmp_path: Path, stage_1: float | None, stage_2: float | None, expected: tuple[str, ...]
) -> None:
    """A stage override is the whole strength for that stage; an unread scalar is not shown."""
    selected = LTX2AdapterSelection(
        tmp_path / "adapter.safetensors",
        strength=0.75,
        stage_1_strength=stage_1,
        stage_2_strength=stage_2,
    )
    assert _stage_strength_details(selected) == expected


def test_model_failure_message_uses_one_canonical_prefix() -> None:
    assert model_failure_message("ltx2", "generation", "bad input") == (
        "LTX-2 generation failed: bad input"
    )
    assert (
        model_failure_message(
            "gmnet",
            "expansion",
            "GMNet expansion failed: bad input",
        )
        == "GMNet expansion failed: bad input"
    )
    assert (
        model_failure_message(
            "ltx2",
            "restart",
            LTX2Error("LTX-2 generation failed: shape mismatch in saved latents"),
        )
        == "LTX-2 restart failed: shape mismatch in saved latents"
    )


def test_verbose_error_record_serializes_internal_traceback(caplog, capsys) -> None:
    import logging

    try:
        raise RuntimeError("internal detail")
    except RuntimeError as exc:
        with caplog.at_level(logging.DEBUG, logger="kinomlx.cli.presentation"):
            exit_code = render_model_error("krea2", "generation", exc, json_output=False)

    assert exit_code == 2
    debug = next(record for record in caplog.records if record.levelno == logging.DEBUG)
    assert debug.exc_info is None
    assert "Traceback (most recent call last)" in debug.getMessage()
    assert "RuntimeError: internal detail" in debug.getMessage()
    assert "Krea 2 generation failed: internal detail" in caplog.text
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("model", "module_name"),
    [
        ("ltx2", "kinomlx.models.ltx2.arguments"),
        ("krea2", "kinomlx.models.krea2.arguments"),
        ("gmnet", "kinomlx.models.gmnet.arguments"),
    ],
)
def test_model_help_includes_its_starter_config_command(
    model: str,
    module_name: str,
    capsys,
) -> None:
    builder = importlib.import_module(module_name).build_parser
    with pytest.raises(SystemExit) as exit_info:
        builder().parse_args(["--help"])
    assert exit_info.value.code == 0
    assert f"kinomlx config init --model {model}" in capsys.readouterr().out


def test_root_help_describes_every_registered_model(capsys) -> None:
    parser = build_root_parser()
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "ltx2   generate distilled LTX-2.3/LTX-2.5 video" in output
    assert "gmnet  expand one SDR still to HDR (EXR / PQ HEIC) with GMNet" in output
    assert "krea2  generate Krea 2 Raw or Turbo still images" in output


def test_print_config_round_trips_complete_invocation(tmp_path: Path, capsys) -> None:
    output = tmp_path / "out.mp4"
    assert main(["--prompt", "test", "--output", str(output), "--print-config"]) == 0
    first_text = capsys.readouterr().out
    first = tomllib.loads(first_text)
    assert first["generate"]["prompt"] == "test\n"
    assert first["output"]["path"] == str(output)

    resolved = tmp_path / "resolved.toml"
    resolved.write_text(first_text, encoding="utf-8")
    assert main(["--config", str(resolved), "--print-config"]) == 0
    second = tomllib.loads(capsys.readouterr().out)
    assert second == first


def test_save_config_matches_print_config_without_running_inference(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "out.mp4"
    saved = tmp_path / "resolved.toml"
    arguments = ["--prompt", "test", "--output", str(output)]

    assert main([*arguments, "--save-config", str(saved)]) == 0
    saved_config = tomllib.loads(saved.read_text(encoding="utf-8"))
    capsys.readouterr()
    assert main([*arguments, "--print-config"]) == 0

    assert tomllib.loads(capsys.readouterr().out) == saved_config


def test_duration_save_config_round_trips_the_selected_duration(
    tmp_path: Path,
    capsys,
) -> None:
    saved = tmp_path / "duration.toml"

    assert main(["--prompt", "test", "--duration", "6", "--save-config", str(saved)]) == 0
    first = tomllib.loads(saved.read_text(encoding="utf-8"))
    assert first["generate"]["duration"] == 6.0
    assert "frames" not in first["generate"]

    capsys.readouterr()
    assert main(["--config", str(saved), "--print-config"]) == 0
    assert tomllib.loads(capsys.readouterr().out) == first


def test_resolved_duration_pair_is_rejected_even_when_consistent(
    tmp_path: Path,
    capsys,
) -> None:
    legacy = tmp_path / "duration-pair.toml"
    legacy.write_text(
        """
[generate]
prompt = "test"
frames = 145
duration = 6.0
fps = 24.0
""".strip(),
        encoding="utf-8",
    )

    assert main(["--config", str(legacy), "--print-config"]) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert "cannot be combined" in " ".join(captured.err.split())


def test_legacy_full_dump_with_inactive_vae_geometry_reloads(
    tmp_path: Path,
    capsys,
) -> None:
    legacy = tmp_path / "legacy.toml"
    legacy.write_text(
        """
[generate]
prompt = "test"

[generate.vae_tiling]
mode = "auto"
temporal_overlap_frames = 24
spatial_overlap_pixels = 64
""".strip(),
        encoding="utf-8",
    )

    assert main(["--config", str(legacy), "--print-config"]) == 0
    resolved = tomllib.loads(capsys.readouterr().out)
    assert resolved["generate"]["vae_tiling"] == {"mode": "auto"}


def test_save_config_refuses_to_replace_an_existing_file(tmp_path: Path, caplog) -> None:
    saved = tmp_path / "existing.toml"
    saved.write_text("keep me\n", encoding="utf-8")

    assert main(["--prompt", "test", "--save-config", str(saved)]) == 2

    assert saved.read_text(encoding="utf-8") == "keep me\n"
    assert f"output already exists: {saved}" in caplog.text
    assert "choose another path or remove the existing file" in caplog.text


def test_sparse_config_preserves_cli_and_environment_non_defaults(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("KINO_CHECK_ADAPTER_WEIGHTS", "false")
    monkeypatch.setenv("KINO_LTX2_OUTPUT_DIR", str(tmp_path / "renders"))
    sparse = tmp_path / "sparse.toml"
    arguments = ["--prompt", "test", "--seed", "7"]

    assert main([*arguments, "--print-config"]) == 0
    full = tomllib.loads(capsys.readouterr().out)
    assert (
        main(
            [
                *arguments,
                "--save-config",
                str(sparse),
                "--only-non-defaults",
            ]
        )
        == 0
    )
    sparse_config = tomllib.loads(sparse.read_text(encoding="utf-8"))
    assert sparse_config["model"] == "ltx2"
    assert sparse_config["generate"] == {"prompt": "test\n", "seed": 7}
    assert not sparse_config["settings"]["check_adapter_weights"]
    assert sparse_config["output"]["directory"] == str(tmp_path / "renders")
    assert "width" not in sparse_config["generate"]

    capsys.readouterr()
    assert main(["--config", str(sparse), "--print-config"]) == 0
    assert tomllib.loads(capsys.readouterr().out) == full


def test_sparse_config_preserves_builtins_that_mask_environment_values(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("KINO_LTX2_OUTPUT_DIR", str(tmp_path / "from-environment"))
    monkeypatch.setenv("KINO_LTX2_FAST_MODE", "0")
    sparse = tmp_path / "environment-masks.toml"
    arguments = [
        "--prompt",
        "test",
        "--output-dir",
        "outputs",
        "--fast-mode",
    ]

    assert main([*arguments, "--print-config"]) == 0
    full = tomllib.loads(capsys.readouterr().out)
    assert (
        main(
            [
                *arguments,
                "--save-config",
                str(sparse),
                "--only-non-defaults",
            ]
        )
        == 0
    )
    exported = tomllib.loads(sparse.read_text(encoding="utf-8"))
    assert exported["output"]["directory"] == "outputs"
    assert exported["model_settings"]["fast_mode"] is True

    capsys.readouterr()
    assert main(["--config", str(sparse), "--print-config"]) == 0
    assert tomllib.loads(capsys.readouterr().out) == full


def test_sparse_save_all_keeps_selector_and_explicit_opt_out(tmp_path: Path) -> None:
    sparse = tmp_path / "sidecars.toml"
    assert (
        main(
            [
                "--prompt",
                "test",
                "--save-all-sidecars",
                "--no-save-console-log",
                "--save-config",
                str(sparse),
                "--only-non-defaults",
            ]
        )
        == 0
    )

    output = tomllib.loads(sparse.read_text(encoding="utf-8"))["output"]
    assert output["save_all_sidecars"] is True
    assert output["save_console_log"] is False
    assert "save_run_log" not in output
    assert "save_effective_config" not in output


def test_sparse_auto_duration_round_trips_the_virtual_selector(
    tmp_path: Path,
    capsys,
) -> None:
    sparse = tmp_path / "automatic.toml"
    assert (
        main(
            [
                "--prompt",
                "test",
                "--auto-duration",
                "--save-config",
                str(sparse),
                "--only-non-defaults",
            ]
        )
        == 0
    )
    assert tomllib.loads(sparse.read_text(encoding="utf-8"))["generate"] == {
        "prompt": "test\n",
        "auto_duration": True,
    }

    assert main(["--config", str(sparse), "--print-config"]) == 0
    assert tomllib.loads(capsys.readouterr().out)["generate"]["auto_duration"] is True


def test_only_non_defaults_requires_a_config_output_surface(capfd) -> None:
    assert main(["--prompt", "test", "--only-non-defaults"]) == 2
    error = capfd.readouterr().err
    assert "--only-non-defaults requires --print-config" in error
    assert "--save-config" in error


def test_invalid_auto_duration_is_a_clean_config_error(
    tmp_path: Path,
    capfd,
) -> None:
    config = tmp_path / "invalid-duration.toml"
    config.write_text(
        '[generate]\nprompt = "test"\nauto_duration = "yes"\n',
        encoding="utf-8",
    )

    assert main(["--config", str(config), "--print-config"]) == 2
    captured = capfd.readouterr()
    assert "config error:" in captured.err
    assert "auto_duration: expected bool, got str" in " ".join(captured.err.split())
    assert "Traceback" not in captured.err


def test_model_failure_is_an_operational_error(
    monkeypatch,
    tmp_path: Path,
    capfd,
) -> None:
    _inject_operational_failure(monkeypatch)
    result = main(["--prompt", "test", "--output", str(tmp_path / "out.mp4")])
    captured = capfd.readouterr()
    assert result == 2
    assert captured.err.count("generation failed") == 1
    assert "injected model failure" in captured.err
    assert "Traceback" not in captured.err


def test_model_failure_releases_generated_output_reservation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _inject_operational_failure(monkeypatch)
    result = main(
        [
            "--quiet",
            "--prompt",
            "test",
            "--output-dir",
            str(tmp_path),
            "--output-prefix",
            "failed",
        ]
    )

    assert result == 2
    assert list(tmp_path.glob("failed_*.mp4")) == []


def test_model_failure_finalizes_requested_run_log(monkeypatch, tmp_path: Path) -> None:
    _inject_operational_failure(monkeypatch)
    output = tmp_path / "failed.mp4"
    assert (
        main(
            [
                "--quiet",
                "--prompt",
                "test",
                "--output",
                str(output),
                "--save-run-log",
            ]
        )
        == 2
    )
    payload = json.loads((tmp_path / "failed_run.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error"] == "LTX2Error: injected model failure"
    assert payload["outputs"] == {"run_log": str(tmp_path / "failed_run.json")}


def test_effective_config_is_written_before_model_execution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _inject_operational_failure(monkeypatch)
    monkeypatch.setenv("KINO_CHECK_ADAPTER_WEIGHTS", "false")
    output = tmp_path / "failed.mp4"

    assert (
        main(
            [
                "--quiet",
                "--prompt",
                "test",
                "--seed",
                "7",
                "--output",
                str(output),
                "--save-effective-config",
            ]
        )
        == 2
    )

    effective = tomllib.loads((tmp_path / "failed_config.toml").read_text(encoding="utf-8"))
    assert effective["generate"]["prompt"] == "test\n"
    assert effective["generate"]["seed"] == 7
    assert effective["output"]["path"] == str(output)
    assert not effective["settings"]["check_adapter_weights"]


def test_effective_config_records_adapter_intent_and_the_resolved_frame_count(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The per-seed projection must not manufacture record defaults or a second duration."""
    _inject_operational_failure(monkeypatch)
    monkeypatch.setenv("KINO_CHECK_ADAPTER_WEIGHTS", "false")
    style = tmp_path / "style.safetensors"
    distill = tmp_path / "distill.safetensors"
    style.write_bytes(b"")
    distill.write_bytes(b"")
    output = tmp_path / "failed.mp4"

    assert (
        main(
            [
                "--quiet",
                "--prompt",
                "test",
                "--duration",
                "2",
                "--seed",
                "7",
                "--adapter",
                str(style),
                "--adapter-exclude",
                "audio",
                "--adapter",
                f"{distill}:0.25:0.5",
                "--output",
                str(output),
                "--save-effective-config",
                "--save-run-log",
            ]
        )
        == 2
    )

    adapters = [
        {"path": str(style), "exclude": ["audio"]},
        {"path": str(distill), "stage1_strength": 0.25, "stage2_strength": 0.5},
    ]
    effective = tomllib.loads((tmp_path / "failed_config.toml").read_text(encoding="utf-8"))
    assert effective["generate"]["adapters"] == adapters
    assert "duration" not in effective["generate"]
    assert effective["generate"]["frames"] == 49
    payload = json.loads((tmp_path / "failed_run.json").read_text(encoding="utf-8"))
    assert payload["effective_config"]["generate"]["adapters"] == adapters
    assert "duration" not in payload["effective_config"]["generate"]


def test_failed_run_does_not_claim_an_execution_log_that_failed_to_initialize(
    monkeypatch,
    tmp_path: Path,
) -> None:

    _inject_operational_failure(monkeypatch)
    monkeypatch.setattr(
        "kinomlx.models.ltx2.command.initialize_execution_log",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(SidecarError("disk full")),
    )
    output = tmp_path / "failed.mp4"
    assert (
        main(
            [
                "--quiet",
                "--prompt",
                "test",
                "--output",
                str(output),
                "--save-run-log",
                "--save-console-log",
            ]
        )
        == 2
    )

    payload = json.loads((tmp_path / "failed_run.json").read_text(encoding="utf-8"))
    assert payload["outputs"] == {"run_log": str(tmp_path / "failed_run.json")}
    assert payload["sidecar_errors"] == [
        {
            "artifact": "execution_log",
            "path": str(tmp_path / "failed_console.log"),
            "error": "SidecarError: disk full",
        }
    ]


def test_json_mode_emits_one_structured_error(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    _inject_operational_failure(monkeypatch)
    result = main(["--json", "--prompt", "test", "--output", str(tmp_path / "out.mp4")])
    captured = capsys.readouterr()
    assert result == 2
    assert json.loads(captured.out) == {
        "status": "error",
        "error": "LTX-2 generation failed: injected model failure",
    }


def test_environment_json_mode_applies_to_assembly_errors(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.setenv("KINO_JSON", "1")
    missing = tmp_path / "missing.toml"
    result = main(["--config", str(missing)])
    captured = capsys.readouterr()
    assert result == 2
    payload = json.loads(captured.out)
    assert payload["status"] == "error"
    assert payload["error"].startswith(f"config error: cannot read config {missing}")
    assert captured.err == ""


def test_explicit_no_json_overrides_environment_for_assembly_errors(
    monkeypatch,
    tmp_path: Path,
    capfd,
) -> None:
    monkeypatch.setenv("KINO_JSON", "1")
    missing = tmp_path / "missing.toml"
    result = main(["--no-json", "--config", str(missing)])
    captured = capfd.readouterr()
    assert result == 2
    assert captured.out == ""
    assert "config error" in captured.err


def test_invalid_environment_setting_is_a_clean_config_error(monkeypatch, capfd) -> None:
    monkeypatch.setenv("KINO_LTX2_TRANSFORMER_RESIDENT_BLOCKS", "many")
    result = main(["--print-config"])
    captured = capfd.readouterr()
    assert result == 2
    assert "config error" in captured.err
    assert "transformer_resident_blocks" in captured.err
    assert "Traceback" not in captured.err


def test_json_environment_survives_another_invalid_environment_setting(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("KINO_JSON", "1")
    monkeypatch.setenv("KINO_LTX2_TRANSFORMER_RESIDENT_BLOCKS", "many")
    result = main(["--print-config"])
    captured = capsys.readouterr()
    assert result == 2
    payload = json.loads(captured.out)
    assert payload["status"] == "error"
    assert "transformer_resident_blocks" in payload["error"]
    assert captured.err == ""


def test_quiet_dispatch_keeps_reporter_accounting_without_live_rows(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import kinomlx.models.ltx2.command as main_module

    final = tmp_path / "final.mp4"
    reporter_calls = []

    class _Reporter:
        def __init__(self, *, disable=None, phase_labels=None):
            reporter_calls.append(disable)

        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return None

        def phase_start(self, *_args, **_kwargs):
            pass

        def phase_advance(self, *_args, **_kwargs):
            pass

        def phase_end(self, *_args, **_kwargs):
            pass

    class _Runner:
        def run(self, _recipe, _config):
            return _generation_output()

    monkeypatch.setattr("kinomlx.reporter_stack.RichReporter", _Reporter)
    monkeypatch.setattr(main_module, "create_runner", lambda *_args, **_kwargs: _Runner())
    monkeypatch.setattr(main_module, "write_generation", lambda *_args, **_kwargs: final)
    assert main(["--quiet", "--prompt", "test", "--output", str(final)]) == 0
    assert reporter_calls == [True]


def test_success_dispatch_reaches_generation_output_adapter(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import kinomlx.models.ltx2.command as main_module

    output = tmp_path / "final.mp4"
    frames = _frame_stream()
    waveform = object()
    generated_configs = []
    encoded_calls = []

    class _Runner:
        def run(self, _recipe, config):
            generated_configs.append(config)
            return GenerationOutput(
                frames=frames,
                audio_waveform=waveform,
                audio_sample_rate=48_000,
            )

    def fake_encode(encoded_frames, path, **kwargs):
        encoded_calls.append((encoded_frames, path, kwargs))
        return Path(path)

    monkeypatch.setattr(main_module, "create_runner", lambda *_args, **_kwargs: _Runner())
    monkeypatch.setattr("kinomlx.output.encode_video_videotoolbox", fake_encode)

    assert (
        main(
            [
                "--quiet",
                "--prompt",
                "test",
                "--generate-audio",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert generated_configs[0].generate_audio is True
    encoded_frames, encoded_path, kwargs = encoded_calls[0]
    assert encoded_frames is frames
    assert encoded_path == output
    assert kwargs["audio_waveform"] is waveform
    assert kwargs["audio_sample_rate"] == 48_000
    assert kwargs["fps"] == 24.0


def test_success_dispatch_can_resolve_timestamped_output_without_exact_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import kinomlx.models.ltx2.command as main_module
    import kinomlx.models.ltx2.config as config_module

    expected = tmp_path / "sample_20260817_193500.mp4"
    generated_paths = []

    class _Runner:
        def run(self, _recipe, _config):
            return _generation_output()

    def fake_build(directory, prefix, *, now=None):
        assert Path(directory) == tmp_path
        assert prefix == "sample"
        assert now is None
        return expected.with_suffix("")

    def fake_write(_generation, output, **_kwargs):
        generated_paths.append(output.path)
        expected.write_bytes(b"video")
        return expected

    monkeypatch.setattr(config_module, "timestamped_output_base", fake_build)
    monkeypatch.setattr(main_module, "create_runner", lambda *_args, **_kwargs: _Runner())
    monkeypatch.setattr(main_module, "write_generation", fake_write)

    assert (
        main(
            [
                "--quiet",
                "--prompt",
                "test",
                "--output-dir",
                str(tmp_path),
                "--output-prefix",
                "sample",
                "--save-run-log",
            ]
        )
        == 0
    )
    assert generated_paths == [expected]
    assert expected.is_file()
    run_log = tmp_path / "sample_20260817_193500_run.json"
    payload = json.loads(run_log.read_text(encoding="utf-8"))
    assert payload["effective_config"]["output"]["path"] == str(expected)
    assert payload["outputs"]["run_log"] == str(run_log)


def test_signpost_native_build_uses_the_configured_app_cache_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import kinomlx.models.ltx2.command as main_module

    output = tmp_path / "out.mp4"
    cache = tmp_path / "cache"
    build_directories = []

    class _Runner:
        def run(self, _recipe, _config):
            return _generation_output()

    class _SignpostReporter:
        def __init__(self, delegate, *, log_path, build_dir):
            assert log_path is None
            build_directories.append(build_dir)
            self._delegate = delegate

        def __enter__(self):
            return self._delegate

        def __exit__(self, *_exc_info):
            return None

    monkeypatch.setattr("kinomlx.reporter_stack.SignpostReporter", _SignpostReporter)
    monkeypatch.setattr(main_module, "create_runner", lambda *_args, **_kwargs: _Runner())
    monkeypatch.setattr(main_module, "write_generation", lambda *_args, **_kwargs: output)

    assert (
        main(
            [
                "--quiet",
                "--prompt",
                "test",
                "--output",
                str(output),
                "--app-cache-dir",
                str(cache),
                "--profile-signposts",
            ]
        )
        == 0
    )
    assert build_directories == [cache / "_native" / "signpost"]


def test_cli_writes_structured_run_and_human_execution_logs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import kinomlx.models.ltx2.command as main_module
    import kinomlx.videotoolbox as videotoolbox

    output = tmp_path / "final.mp4"
    (tmp_path / "final.wav").touch()

    class _Runner:
        def run(self, _recipe, _config):
            return _generation_output(
                metadata={
                    "model_version": "2.3",
                    "warnings": ("adapter.safetensors: low target coverage (20.0%)",),
                }
            )

    def fake_encode(_frames, path, **_kwargs):
        Path(path).touch()
        return Path(path)

    monkeypatch.setattr(main_module, "create_runner", lambda *_args, **_kwargs: _Runner())
    monkeypatch.setattr(videotoolbox, "encode_video_videotoolbox", fake_encode)

    assert (
        main(
            [
                "--quiet",
                "--prompt",
                "test",
                "--output",
                str(output),
                "--save-run-log",
                "--save-console-log",
            ]
        )
        == 0
    )

    run_log = tmp_path / "final_run.json"
    execution_log = tmp_path / "final_console.log"
    payload = json.loads(run_log.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    # Every run records its seed selection, single-seed runs included.
    assert payload["generation"] == {
        "model_version": "2.3",
        "warnings": ["adapter.safetensors: low target coverage (20.0%)"],
        "seed": 42,
        "seeds": [42],
        "seed_index": 0,
    }
    assert payload["warnings"] == ["adapter.safetensors: low target coverage (20.0%)"]
    assert payload["outputs"] == {
        "execution_log": str(execution_log),
        "run_log": str(run_log),
        "video": str(output),
    }
    text = execution_log.read_text(encoding="utf-8")
    assert text.startswith("Command: kinomlx --quiet --prompt test")
    assert "Run:" in text
    assert "Model: LTX-2" in text
    assert "Sampling:" in text
    assert "Compute:" in text
    assert "Conditioning:" in text
    assert "Delivery:" in text
    assert "Prompt:\ntest\n" in text
    assert "Planned outputs:" in text
    assert "Outputs:" in text
    assert "Total runtime:" in text
    assert text.rfind("Outputs:") < text.rfind("Total runtime:")


def test_run_record_lists_the_separate_vae_frame_dump(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import kinomlx.models.ltx2.command as main_module

    output = tmp_path / "final.mp4"
    frame_directory = tmp_path / "final_vae_frames"

    class _Runner:
        def run(self, _recipe, _config):
            return _generation_output()

    def fake_write_generation(_generation, config, **_kwargs):
        assert config.save_vae_frames
        output.touch()
        frame_directory.mkdir()
        (frame_directory / "manifest.json").write_text("{}\n")
        return output

    monkeypatch.setattr(main_module, "create_runner", lambda *_args, **_kwargs: _Runner())
    monkeypatch.setattr(main_module, "write_generation", fake_write_generation)

    assert (
        main(
            [
                "--quiet",
                "--prompt",
                "test",
                "--output",
                str(output),
                "--save-run-log",
                "--save-vae-frames",
            ]
        )
        == 0
    )
    payload = json.loads((tmp_path / "final_run.json").read_text(encoding="utf-8"))
    assert payload["outputs"]["vae_frames"] == str(frame_directory)
    assert payload["planned_outputs"]["vae_frames"] == str(frame_directory)


def test_steel_probe_trace_summary_is_written_to_run_record(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import kinomlx.models.ltx2.command as main_module

    output = tmp_path / "final.mp4"
    summary = {
        "enabled": True,
        "scope": "process_compiled_traces",
        "counter_unit": "compiled_trace",
        "hit_d128": 4,
    }

    class _Runner:
        def run(self, _recipe, _config):
            return _generation_output()

    def fake_write_generation(*_args, **_kwargs):
        output.touch()
        return output

    monkeypatch.setenv("KINO_LTX2_STEEL_ATTENTION_PROBE", "1")
    monkeypatch.setattr(main_module, "create_runner", lambda *_args, **_kwargs: _Runner())
    monkeypatch.setattr(main_module, "write_generation", fake_write_generation)
    monkeypatch.setattr("kinomlx.models.ltx2.command.steel_attention_summary", lambda: summary)

    assert (
        main(
            [
                "--quiet",
                "--prompt",
                "test",
                "--output",
                str(output),
                "--save-run-log",
            ]
        )
        == 0
    )
    payload = json.loads((tmp_path / "final_run.json").read_text(encoding="utf-8"))
    assert payload["diagnostics"]["steel_attention"] == summary
    assert payload["diagnostics"]["memory"]["counter_source"] == "mlx_allocator"


def test_run_record_captures_stage_memory_and_lazy_generation_diagnostics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import kinomlx.debug as debug_module
    import kinomlx.models.ltx2.command as main_module

    output = tmp_path / "final.mp4"
    counter = [0]

    def sample_memory() -> dict[str, int]:
        counter[0] += 1
        return {
            "active_bytes": counter[0] * 10,
            "cache_bytes": counter[0] * 20,
            "peak_bytes": counter[0] * 30,
        }

    expected_vae = {
        "vae_decode": {
            "entry_memory": {"unaccounted_active_bytes": 0},
            "tiling": {"requested_mode": True, "total_tiles": 1},
        }
    }

    class _Runner:
        def run(self, _recipe, _config):
            return GenerationOutput(
                frames=_frame_stream(),
                diagnostics_provider=lambda: expected_vae,
            )

    def fake_write_generation(*_args, **_kwargs):
        output.touch()
        return output

    monkeypatch.setattr(
        debug_module,
        "create_mlx_memory_sampler",
        lambda: sample_memory,
    )
    monkeypatch.setattr(main_module, "create_runner", lambda *_args, **_kwargs: _Runner())
    monkeypatch.setattr(main_module, "write_generation", fake_write_generation)

    assert (
        main(
            [
                "--quiet",
                "--prompt",
                "test",
                "--output",
                str(output),
                "--save-run-log",
            ]
        )
        == 0
    )
    diagnostics = json.loads((tmp_path / "final_run.json").read_text())["diagnostics"]
    assert diagnostics["vae_decode"] == expected_vae["vae_decode"]
    memory = diagnostics["memory"]
    assert memory["counter_source"] == "mlx_allocator"
    assert memory["unit"] == "bytes"
    assert memory["synchronizes_device"] is False
    assert memory["sampling_errors"] == []
    assert [sample["label"] for sample in memory["samples"]] == [
        "runner_start",
        "generation_ready",
        "output_complete",
    ]


def test_run_record_identifies_preexisting_sidecars_not_replaced_by_current_run(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import kinomlx.models.ltx2.command as main_module

    output = tmp_path / "final.mp4"
    stale = tmp_path / "final_stage1.safetensors"
    stale.write_bytes(b"previous run")

    class _Runner:
        def run(self, _recipe, _config):
            return _generation_output()

    def fake_write_generation(*_args, **_kwargs):
        output.touch()
        return output

    monkeypatch.setattr(main_module, "create_runner", lambda *_args, **_kwargs: _Runner())
    monkeypatch.setattr(main_module, "write_generation", fake_write_generation)

    assert (
        main(
            [
                "--quiet",
                "--prompt",
                "test",
                "--output",
                str(output),
                "--save-run-log",
            ]
        )
        == 0
    )
    payload = json.loads((tmp_path / "final_run.json").read_text(encoding="utf-8"))
    expected = {"stage_1_latents": str(stale)}
    assert payload["preexisting_sidecars"] == expected
    assert payload["stale_sidecars"] == expected


def test_save_all_sidecars_reaches_every_pipeline_artifact(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    import kinomlx.debug.sidecars as sidecars_module
    import kinomlx.models.ltx2.command as main_module

    output = tmp_path / "final.mp4"
    condition = tmp_path / "condition.png"
    condition.touch()
    written: list[Path] = []

    class _Runner:
        def __init__(self, artifact_sink):
            self._artifacts = artifact_sink

        def run(self, _recipe, _config):
            self._artifacts.save(
                text_conditioning_artifact(
                    prompt="neutral prompt",
                    video_encoding=object(),
                    audio_encoding=object(),
                    attention_mask=object(),
                    provenance={
                        "model_generation": "ltx-2.3",
                        "text_encoder_identity": "gemma-3-12b-it",
                        "projection_identity": "connector:test",
                    },
                )
            )
            self._artifacts.save(
                distilled_stage_latents_artifact(
                    1,
                    video_latent=object(),
                    audio_latent=None,
                    final=False,
                )
            )
            self._artifacts.save(
                distilled_stage_latents_artifact(
                    2,
                    video_latent=object(),
                    audio_latent=None,
                    final=True,
                )
            )
            self._artifacts.save(
                TensorArtifact(
                    STAGE_1_CONDITIONING,
                    (("condition_0_latent", object()),),
                )
            )
            self._artifacts.save(
                TensorArtifact(
                    STAGE_2_CONDITIONING,
                    (("condition_0_latent", object()),),
                )
            )
            return _generation_output()

    def fake_save(path, *_args, **_kwargs):
        path = Path(path)
        path.touch()
        written.append(path)

    def fake_write_generation(*_args, **_kwargs):
        output.touch()
        return output

    monkeypatch.setattr(
        main_module,
        "create_runner",
        lambda *_args, **kwargs: _Runner(kwargs["artifact_sink"]),
    )
    monkeypatch.setattr(main_module, "write_generation", fake_write_generation)
    monkeypatch.setattr(sidecars_module, "save_weights", fake_save)

    assert (
        main(
            [
                "--json",
                "--prompt",
                "test",
                "--output",
                str(output),
                "--image",
                str(condition),
                "--save-all-sidecars",
            ]
        )
        == 0
    )
    cli_result = json.loads(capsys.readouterr().out)
    assert cli_result["outputs"]["video"] == str(output)
    assert cli_result["action"] == "generate"
    assert set(written) == {
        tmp_path / "final_text.safetensors",
        tmp_path / "final_stage1.safetensors",
        tmp_path / "final.safetensors",
        tmp_path / "final_stage1_conditioning.safetensors",
        tmp_path / "final_stage2_conditioning.safetensors",
    }
    payload = json.loads((tmp_path / "final_run.json").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["sidecar_errors"] == []
    assert set(payload["outputs"]) >= {
        "video",
        "text_conditioning",
        "stage_1_latents",
        "final_latents",
        "stage_1_conditioning",
        "stage_2_conditioning",
        "run_log",
        "execution_log",
        "effective_config",
    }
    assert (tmp_path / "final_config.toml").is_file()


def test_hdr_heic_sequence_is_recorded_as_a_completed_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import kinomlx.models.ltx2.command as main_module

    output = tmp_path / "hdr.mp4"

    class _Runner:
        def run(self, _recipe, _config):
            return _generation_output()

    def fake_write_generation(generation, *_args, **_kwargs):
        generation.close()
        output.touch()
        for directory in (tmp_path / "hdr_exr", tmp_path / "hdr_heic"):
            directory.mkdir()
            (directory / "manifest.json").write_text("{}\n", encoding="utf-8")
        return output

    monkeypatch.setattr(main_module, "create_runner", lambda *_args, **_kwargs: _Runner())
    monkeypatch.setattr(main_module, "write_generation", fake_write_generation)

    assert (
        main(
            [
                "--quiet",
                "--prompt",
                "test",
                "--hdr",
                "ACESCG",
                "--output",
                str(output),
                "--save-hdr-heic-frames",
                "--save-run-log",
            ]
        )
        == 0
    )
    payload = json.loads((tmp_path / "hdr_run.json").read_text(encoding="utf-8"))
    assert payload["outputs"]["exr_frames"] == str(tmp_path / "hdr_exr")
    assert payload["outputs"]["heic_frames"] == str(tmp_path / "hdr_heic")


def test_sidecar_write_failures_do_not_abort_primary_output(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    import kinomlx.debug.sidecars as sidecars_module
    import kinomlx.models.ltx2.command as main_module

    output = tmp_path / "final.mp4"

    class _Runner:
        def __init__(self, artifact_sink):
            self._artifacts = artifact_sink

        def run(self, _recipe, _config):
            # --save-latents plans both stage products, so a run that finishes
            # normally offers both; the sink is what fails here, not the run.
            for stage, final in ((1, False), (2, True)):
                self._artifacts.save(
                    distilled_stage_latents_artifact(
                        stage,
                        video_latent=object(),
                        audio_latent=None,
                        final=final,
                    )
                )
            return _generation_output()

    def fake_write_generation(*_args, **_kwargs):
        output.touch()
        return output

    def fail_sidecar(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        main_module,
        "create_runner",
        lambda *_args, **kwargs: _Runner(kwargs["artifact_sink"]),
    )
    monkeypatch.setattr(main_module, "write_generation", fake_write_generation)
    monkeypatch.setattr(sidecars_module, "save_weights", fail_sidecar)

    assert (
        main(
            [
                "--json",
                "--prompt",
                "test",
                "--output",
                str(output),
                "--save-latents",
                "--save-run-log",
            ]
        )
        == 0
    )
    assert output.is_file()
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
    payload = json.loads((tmp_path / "final_run.json").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["sidecar_errors"] == [
        {
            "artifact": "stage_1_latents",
            "path": str(tmp_path / "final_stage1.safetensors"),
            "error": "OSError: disk full",
        },
        {
            "artifact": "final_latents",
            "path": str(tmp_path / "final.safetensors"),
            "error": "OSError: disk full",
        },
    ]
    assert "stage_1_latents" not in payload["outputs"]
    assert "final_latents" not in payload["outputs"]


def test_effective_config_failure_is_nonfatal_and_recorded(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    import kinomlx.models.ltx2.command as main_module

    output = tmp_path / "final.mp4"

    class _Runner:
        def run(self, _recipe, _config):
            return _generation_output()

    def fake_write_generation(*_args, **_kwargs):
        output.touch()
        return output

    monkeypatch.setattr(main_module, "create_runner", lambda *_args, **_kwargs: _Runner())
    monkeypatch.setattr(main_module, "write_generation", fake_write_generation)
    monkeypatch.setattr(
        "kinomlx.models.ltx2.command.write_effective_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(SidecarError("disk full")),
    )

    assert (
        main(
            [
                "--json",
                "--prompt",
                "test",
                "--output",
                str(output),
                "--save-effective-config",
                "--save-run-log",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
    payload = json.loads((tmp_path / "final_run.json").read_text(encoding="utf-8"))
    assert payload["sidecar_errors"] == [
        {
            "artifact": "effective_config",
            "path": str(tmp_path / "final_config.toml"),
            "error": "SidecarError: disk full",
        }
    ]
    assert "effective_config" not in payload["outputs"]


def test_completed_run_survives_run_log_finalize_failure(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    import kinomlx.models.ltx2.command as main_module

    output = tmp_path / "final.mp4"

    class _Runner:
        def run(self, _recipe, _config):
            return _generation_output()

    class _FailingRunRecord:
        def __init__(self, *_args, **_kwargs):
            pass

        def write(self, *, status, **_kwargs):
            if status == "completed":
                raise SidecarError("injected run-log failure")

    def fake_write_generation(*_args, **_kwargs):
        output.touch()
        return output

    monkeypatch.setattr(main_module, "create_runner", lambda *_args, **_kwargs: _Runner())
    monkeypatch.setattr(main_module, "write_generation", fake_write_generation)
    monkeypatch.setattr("kinomlx.models.ltx2.command.RunRecord", _FailingRunRecord)

    assert (
        main(
            [
                "--json",
                "--prompt",
                "test",
                "--output",
                str(output),
                "--save-run-log",
            ]
        )
        == 0
    )
    assert output.is_file()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok"
    assert result["model"] == "ltx2"
    assert result["action"] == "generate"
    assert result["outputs"] == {"video": str(output)}
    assert result["generation"] == {"seed": 42, "seeds": [42], "seed_index": 0}
    assert result["warnings"] == []
    assert result["sidecar_errors"] == [
        {
            "artifact": "run_log",
            "path": str(tmp_path / "final_run.json"),
            "error": "SidecarError: injected run-log failure",
        }
    ]


def test_programmer_failure_propagates(monkeypatch, tmp_path: Path) -> None:
    import kinomlx.models.ltx2.command as main_module

    failure = AssertionError("injected defect")

    class _Runner:
        def run(self, _recipe, _config):
            raise failure

    monkeypatch.setattr(main_module, "create_runner", lambda *_args, **_kwargs: _Runner())
    with pytest.raises(AssertionError) as caught:
        main(["--quiet", "--prompt", "test", "--output", str(tmp_path / "x.mp4")])
    assert caught.value is failure


def _offering_runner(monkeypatch, tmp_path: Path, *, offers: tuple[bool, ...], interrupt=None):
    """Install an LTX runner that offers the named latent stages and stops."""
    import kinomlx.models.ltx2.command as main_module

    output = tmp_path / "final.mp4"

    class _Runner:
        def __init__(self, artifact_sink):
            self._artifacts = artifact_sink

        def run(self, _recipe, _config):
            for stage, final in zip((1, 2), (False, True), strict=True):
                if not offers[stage - 1]:
                    continue
                self._artifacts.save(
                    distilled_stage_latents_artifact(
                        stage,
                        video_latent=mx.zeros((1, 1, 1, 1), dtype=mx.bfloat16),
                        audio_latent=None,
                        final=final,
                    )
                )
            if interrupt is not None:
                raise interrupt
            return _generation_output()

    def fake_write_generation(*_args, **_kwargs):
        output.touch()
        return output

    monkeypatch.setattr(
        main_module, "create_runner", lambda *_args, **kwargs: _Runner(kwargs["artifact_sink"])
    )
    monkeypatch.setattr(main_module, "write_generation", fake_write_generation)
    return output


def test_a_planned_artifact_the_run_never_produced_fails_the_run(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    import logging

    output = _offering_runner(monkeypatch, tmp_path, offers=(True, False))

    with caplog.at_level(logging.ERROR, logger="kinomlx"):
        code = main(
            [
                "--quiet",
                "--prompt",
                "test",
                "--output",
                str(output),
                "--save-latents",
                "--save-run-log",
            ]
        )

    assert code == 2
    assert (
        "LTX-2 generation failed: planned artifact final_latents was never produced by this run"
    ) in caplog.text
    assert "Traceback" not in caplog.text
    record = json.loads((tmp_path / "final_run.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert "final_latents was never produced" in record["error"]
    # The stage the run did produce is still recorded as published.
    assert "stage_1_latents" in record["outputs"]


def test_a_run_that_produces_every_planned_artifact_is_unaffected(
    monkeypatch, tmp_path: Path
) -> None:
    output = _offering_runner(monkeypatch, tmp_path, offers=(True, True))

    assert (
        main(
            [
                "--quiet",
                "--prompt",
                "test",
                "--output",
                str(output),
                "--save-latents",
                "--save-run-log",
            ]
        )
        == 0
    )
    record = json.loads((tmp_path / "final_run.json").read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert {"stage_1_latents", "final_latents"} <= record["outputs"].keys()


def test_a_canceled_run_is_not_checked_for_unproduced_artifacts(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    import logging

    output = _offering_runner(
        monkeypatch, tmp_path, offers=(False, False), interrupt=KeyboardInterrupt()
    )

    with caplog.at_level(logging.ERROR, logger="kinomlx"):
        code = main(
            [
                "--quiet",
                "--prompt",
                "test",
                "--output",
                str(output),
                "--save-latents",
                "--save-run-log",
            ]
        )

    assert code == 130
    assert "never produced by this run" not in caplog.text
    record = json.loads((tmp_path / "final_run.json").read_text(encoding="utf-8"))
    assert record["status"] == "aborted"


@pytest.mark.parametrize("json_output", [False, True])
def test_keyboard_interrupt_returns_130_without_a_traceback(
    monkeypatch,
    caplog,
    capsys,
    json_output,
) -> None:
    import kinomlx.cli.main as main_module

    def interrupt(_arguments):
        raise KeyboardInterrupt

    monkeypatch.setattr(main_module, "_run_main", interrupt)
    arguments = ["--json"] if json_output else []
    assert main(arguments) == 130
    if json_output:
        assert json.loads(capsys.readouterr().out) == {"status": "canceled"}
    else:
        assert "Canceled." in caplog.text


def test_pack_fields_keeps_every_item_whole_and_in_order() -> None:
    from kinomlx.cli.presentation import pack_fields

    fields = (
        ("Model", "LTX-2.3"),
        ("Action", "generate"),
        ("Resolution", "768x448"),
        ("Frames", 121),
        ("Seed", 42),
        ("Checkpoint", "x" * 50),
        ("Prompt", "first line\nsecond line"),
        ("FPS", "24"),
    )
    lines = pack_fields(fields, width=40)

    assert lines[:2] == (
        "Model: LTX-2.3 | Action: generate",
        "Resolution: 768x448 | Frames: 121",
    )
    assert all(len(line) <= 40 or " | " not in line for line in lines)
    assert "Prompt: first line\nsecond line" in lines
    items = [item for line in lines for item in line.split(" | ")]
    assert items == [f"{label}: {value}" for label, value in fields]


def test_log_section_packs_fields_within_the_console_message_width(caplog, monkeypatch) -> None:
    import logging

    from kinomlx.cli import presentation

    monkeypatch.setattr(presentation, "console_message_width", lambda: 42)
    logger = logging.getLogger("kinomlx.test.section_packing")
    with caplog.at_level(logging.INFO, logger=logger.name):
        presentation.log_section(
            logger,
            "Run",
            (
                ("Model", "LTX-2.3"),
                ("Action", "generate"),
                ("Resolution", "768x448"),
                ("Frames", 121),
                ("Seed", 42),
            ),
        )

    messages = [record.getMessage() for record in caplog.records]
    assert messages[0] == "Run:"
    body = messages[1:]
    assert 1 < len(body) < 5
    assert all(line.startswith("  ") and len(line) <= 42 for line in body)
    items = [item for line in body for item in line[2:].split(" | ")]
    assert items == [
        "Model: LTX-2.3",
        "Action: generate",
        "Resolution: 768x448",
        "Frames: 121",
        "Seed: 42",
    ]


def test_console_message_width_follows_the_rich_time_cell() -> None:
    from kinomlx.ui import get_console
    from kinomlx.ui.logging import configure_logging, console_message_width

    configure_logging(show_date=True)
    try:
        assert console_message_width() == max(24, get_console().width - 31)
    finally:
        configure_logging()
    assert console_message_width() == max(24, get_console().width - 20)


def test_render_run_success_collapses_outputs_that_match_the_plan(caplog, tmp_path: Path) -> None:
    import logging

    from kinomlx.cli.presentation import render_run_success

    logger = logging.getLogger("kinomlx.test.outputs_as_planned")
    planned = {"video": str(tmp_path / "clip.mp4"), "run_log": str(tmp_path / "clip_run.json")}
    common = {
        "model": "ltx2",
        "action": "generate",
        "elapsed_seconds": 1.0,
        "timings": {},
        "diagnostics": {},
        "json_output": False,
        "planned_outputs": planned,
    }
    with caplog.at_level(logging.INFO, logger=logger.name):
        render_run_success(
            logger,
            outputs={"run_log": tmp_path / "clip_run.json", "video": tmp_path / "clip.mp4"},
            **common,
        )
        render_run_success(logger, outputs={"video": tmp_path / "other.mp4"}, **common)

    messages = [record.getMessage() for record in caplog.records]
    assert messages[0] == "Outputs: as planned"
    assert messages[1].startswith("Total runtime:")
    assert messages[2] == "Outputs:"
    assert any("VIDEO:" in message and "other.mp4" in message for message in messages[3:])


def test_multi_seed_json_emits_one_record_covering_every_seed(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    import kinomlx.models.ltx2.command as main_module

    seen: list[int] = []

    class _Runner:
        resources = object()

        def run(self, _recipe, config):
            seen.append(config.seed)
            return _generation_output()

    written: list[Path] = []

    def _write(_generation, output, **_kwargs):
        written.append(output.path)
        output.path.write_bytes(b"video")
        return output.path

    monkeypatch.setattr(main_module, "create_runner", lambda *_args, **_kwargs: _Runner())
    monkeypatch.setattr(main_module, "prepare_distilled_recipe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module, "write_generation", _write)
    result = main(
        [
            "--json",
            "--prompt",
            "test",
            "--seed",
            "7,8,9",
            "--output",
            str(tmp_path / "out.mp4"),
            "--save-run-log",
        ]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert seen == [7, 8, 9]
    payload = json.loads(captured.out)
    assert payload["status"] == "ok"
    assert payload["seeds"] == [7, 8, 9]
    assert [run["seed"] for run in payload["runs"]] == [7, 8, 9]
    assert [run["seed_index"] for run in payload["runs"]] == [0, 1, 2]
    assert [path.name for path in written] == [
        "out-001-seed7.mp4",
        "out-002-seed8.mp4",
        "out-003-seed9.mp4",
    ]

    records = sorted(tmp_path.glob("*_run.json"))
    assert len(records) == 3
    for seed, record in zip((7, 8, 9), records, strict=True):
        saved = json.loads(record.read_text())
        assert saved["configuration"]["values"]["generate"]["seed"] == seed
        assert saved["configuration"]["values"]["generate"]["seeds"] is None


@pytest.mark.parametrize("model", ["krea2", "flux2"])
def test_cancellation_reports_the_same_status_and_completed_outputs(
    monkeypatch, tmp_path, capsys, model
):
    """Every model renders one canceled record naming what it had finished."""
    from kinomlx.errors import RunCanceled

    def cancel(*_args, **_kwargs):
        raise RunCanceled({"images": [str(tmp_path / "done-001.png")]})

    monkeypatch.setattr(f"kinomlx.models.{model}.command.run_{model}_command", cancel)
    result = main(["--model", model, "--json", "--prompt", "test"])
    captured = capsys.readouterr()

    assert result == 130
    assert json.loads(captured.out) == {
        "status": "canceled",
        "outputs": {"images": [str(tmp_path / "done-001.png")]},
    }


def test_cancellation_without_completed_outputs_stays_minimal(monkeypatch, capsys):
    def cancel(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("kinomlx.models.krea2.command.run_krea2_command", cancel)
    assert main(["--model", "krea2", "--json", "--prompt", "test"]) == 130
    assert json.loads(capsys.readouterr().out) == {"status": "canceled"}


@pytest.mark.parametrize(
    ("argv", "message", "banned"),
    [
        (
            ["--model", "flux2", "--prompt", "x", "--width", "100", "--print-config"],
            "[generate]: explicit Klein output dimensions must be multiples of 16",
            ("Flux2Request", "reference_geometry"),
        ),
        (
            ["--model", "krea2", "--prompt", "x", "--turbo-sda-strength", "1.5", "--print-config"],
            "[generate]: turbo_sda_strength and turbo_sda_exclusions require turbo_sda",
            ("Krea2GenerateConfig", "Krea2Request", "profiles"),
        ),
        (
            ["--model", "ltx2", "--prompt", "x", "--prompt-unconditional", "--print-config"],
            "[generate]: prompt_unconditional cannot be combined with prompt or prompt_file",
            ("DistilledRequest", "unconditional_prompt"),
        ),
        (
            [
                "--model",
                "krea2",
                "--prompt",
                "x",
                "--set",
                "restart.run=absent.json",
                "--set",
                "restart.phase=denoise",
                "--set",
                "restart.latents=latents.safetensors",
                "--print-config",
            ],
            "[restart]: restart latents apply only when phase=decode",
            ("RestartConfig", "restart_selection"),
        ),
    ],
)
def test_record_rule_violations_name_the_table_on_every_model(argv, message, banned, capfd) -> None:
    assert main(argv) == 2
    captured = " ".join(capfd.readouterr().err.split())
    assert message in captured
    assert "Traceback" not in captured
    for name in banned:
        assert name not in captured
