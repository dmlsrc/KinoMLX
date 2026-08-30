"""CLI printing, error-boundary, and dispatch tests."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.artifacts import TensorArtifact
from kinomlx.cli.common import bootstrap_flag_present, bootstrap_json_from_arguments
from kinomlx.cli.main import _format_elapsed, main
from kinomlx.debug import SidecarError
from kinomlx.media.frames import VideoFrameStream
from kinomlx.models.ltx2.artifacts import (
    STAGE_1_CONDITIONING,
    STAGE_2_CONDITIONING,
    distilled_stage_latents_artifact,
    text_conditioning_artifact,
)
from kinomlx.models.ltx2.runner import GenerationOutput, LTX2Error
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
    import kinomlx.cli.main as main_module

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
    assert _format_elapsed(seconds) == expected


def test_bootstrap_flag_scans_stop_at_the_option_terminator() -> None:
    assert not bootstrap_flag_present(["--", "--help"], {"-h", "--help"})
    assert not bootstrap_json_from_arguments(["--", "--json"], Settings())


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


def test_duration_save_config_round_trips_as_resolved_frames(
    tmp_path: Path,
    capsys,
) -> None:
    saved = tmp_path / "duration.toml"

    assert main(["--prompt", "test", "--duration", "6", "--save-config", str(saved)]) == 0
    first = tomllib.loads(saved.read_text(encoding="utf-8"))
    assert first["generate"]["frames"] == 145
    assert "duration" not in first["generate"]

    capsys.readouterr()
    assert main(["--config", str(saved), "--print-config"]) == 0
    assert tomllib.loads(capsys.readouterr().out) == first


def test_legacy_resolved_duration_pair_reloads_when_consistent(
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

    assert main(["--config", str(legacy), "--print-config"]) == 0
    resolved = tomllib.loads(capsys.readouterr().out)
    assert resolved["generate"]["frames"] == 145
    assert "duration" not in resolved["generate"]


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
    monkeypatch.setenv("KINO_CACHE_MODE", "rebuild")
    monkeypatch.setenv("KINO_OUTPUT_DIR", str(tmp_path / "renders"))
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
    assert sparse_config["settings"]["cache_mode"] == "rebuild"
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
    monkeypatch.setenv("KINO_OUTPUT_DIR", str(tmp_path / "from-environment"))
    monkeypatch.setenv("KINO_FAST_MODE", "0")
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
    assert "config error: generate.auto_duration must be a boolean" in captured.err
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
    monkeypatch.setenv("KINO_CACHE_MODE", "rebuild")
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
    assert effective["settings"]["cache_mode"] == "rebuild"


def test_failed_run_does_not_claim_an_execution_log_that_failed_to_initialize(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import kinomlx.debug as debug_module

    _inject_operational_failure(monkeypatch)
    monkeypatch.setattr(
        debug_module,
        "initialize_execution_log",
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
        "error": "generation failed: injected model failure",
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
    monkeypatch.setenv("KINO_TRANSFORMER_RESIDENT_BLOCKS", "many")
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
    monkeypatch.setenv("KINO_TRANSFORMER_RESIDENT_BLOCKS", "many")
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
    import kinomlx.cli.main as main_module
    import kinomlx.ui as ui

    final = tmp_path / "final.mp4"
    reporter_calls = []

    class _Reporter:
        def __init__(self, *, disable=None):
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

    monkeypatch.setattr(ui, "RichReporter", _Reporter)
    monkeypatch.setattr(main_module, "create_runner", lambda *_args, **_kwargs: _Runner())
    monkeypatch.setattr(main_module, "write_generation", lambda *_args, **_kwargs: final)
    assert main(["--quiet", "--prompt", "test", "--output", str(final)]) == 0
    assert reporter_calls == [True]


def test_success_dispatch_reaches_generation_output_adapter(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import kinomlx.cli.main as main_module
    import kinomlx.videotoolbox as videotoolbox

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
    monkeypatch.setattr(videotoolbox, "encode_video_videotoolbox", fake_encode)

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
    import kinomlx.cli.config as config_module
    import kinomlx.cli.main as main_module

    expected = tmp_path / "kitten_20260817_193500.mp4"
    generated_paths = []

    class _Runner:
        def run(self, _recipe, _config):
            return _generation_output()

    def fake_build(directory, prefix, *, now=None):
        assert Path(directory) == tmp_path
        assert prefix == "kitten"
        assert now is None
        return expected

    def fake_write(_generation, output, **_kwargs):
        generated_paths.append(output.path)
        expected.write_bytes(b"video")
        return expected

    monkeypatch.setattr(config_module, "build_timestamped_output_path", fake_build)
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
                "kitten",
                "--save-run-log",
            ]
        )
        == 0
    )
    assert generated_paths == [expected]
    assert expected.is_file()
    run_log = tmp_path / "kitten_20260817_193500_run.json"
    payload = json.loads(run_log.read_text(encoding="utf-8"))
    assert payload["invocation"]["output"]["path"] == str(expected)
    assert payload["outputs"]["run_log"] == str(run_log)


def test_signpost_native_build_uses_the_configured_cache_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import kinomlx.cli.main as main_module
    import kinomlx.profiling as profiling_module

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

    monkeypatch.setattr(profiling_module, "SignpostReporter", _SignpostReporter)
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
                "--cache-dir",
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
    import kinomlx.cli.main as main_module
    import kinomlx.videotoolbox as videotoolbox

    output = tmp_path / "final.mp4"
    (tmp_path / "final.wav").touch()

    class _Runner:
        def run(self, _recipe, _config):
            return _generation_output(metadata={"model_version": "2.3"})

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
    assert payload["generation"] == {"model_version": "2.3"}
    assert payload["outputs"] == {
        "execution_log": str(execution_log),
        "run_log": str(run_log),
        "video": str(output),
    }
    text = execution_log.read_text(encoding="utf-8")
    assert text.startswith("Command: kinomlx --quiet --prompt test")
    assert "Output:" in text
    assert "Total runtime:" in text
    assert text.rfind("Output:") < text.rfind("Total runtime:")


def test_run_record_lists_the_separate_vae_frame_dump(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import kinomlx.cli.main as main_module

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
    import kinomlx.cli.main as main_module
    import kinomlx.kernels.steel_attention as steel_module

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

    monkeypatch.setenv("KINO_STEEL_ATTENTION_PROBE", "1")
    monkeypatch.setattr(main_module, "create_runner", lambda *_args, **_kwargs: _Runner())
    monkeypatch.setattr(main_module, "write_generation", fake_write_generation)
    monkeypatch.setattr(steel_module, "steel_attention_summary", lambda: summary)

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
    import kinomlx.cli.main as main_module
    import kinomlx.debug as debug_module

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
            "tiling": {"requested_mode": "auto", "total_tiles": 1},
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
    import kinomlx.cli.main as main_module

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
    import kinomlx.cli.main as main_module
    import kinomlx.debug.sidecars as sidecars_module

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
    assert json.loads(capsys.readouterr().out)["output"] == str(output)
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
    import kinomlx.cli.main as main_module

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
    import kinomlx.cli.main as main_module
    import kinomlx.debug.sidecars as sidecars_module

    output = tmp_path / "final.mp4"

    class _Runner:
        def __init__(self, artifact_sink):
            self._artifacts = artifact_sink

        def run(self, _recipe, _config):
            self._artifacts.save(
                distilled_stage_latents_artifact(
                    1,
                    video_latent=object(),
                    audio_latent=None,
                    final=False,
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
        }
    ]
    assert "stage_1_latents" not in payload["outputs"]


def test_effective_config_failure_is_nonfatal_and_recorded(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    import kinomlx.cli.main as main_module
    import kinomlx.debug as debug_module

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
        debug_module,
        "write_effective_config",
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
    import kinomlx.cli.main as main_module
    import kinomlx.debug as debug_module

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
    monkeypatch.setattr(debug_module, "RunRecord", _FailingRunRecord)

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
    assert json.loads(capsys.readouterr().out) == {
        "status": "ok",
        "model": "ltx2",
        "output": str(output),
    }


def test_programmer_failure_propagates(monkeypatch, tmp_path: Path) -> None:
    import kinomlx.cli.main as main_module

    failure = AssertionError("injected defect")

    class _Runner:
        def run(self, _recipe, _config):
            raise failure

    monkeypatch.setattr(main_module, "create_runner", lambda *_args, **_kwargs: _Runner())
    with pytest.raises(AssertionError) as caught:
        main(["--quiet", "--prompt", "test", "--output", str(tmp_path / "x.mp4")])
    assert caught.value is failure
