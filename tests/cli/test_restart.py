"""CLI contracts for restarting from saved KinoMLX station products."""

import json
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.config import ConfigError, dump_config
from kinomlx.config.assembly import config_table
from kinomlx.io.fingerprints import file_sha256
from kinomlx.models.ltx2.arguments import build_parser
from kinomlx.models.ltx2.artifacts import distilled_stage_latents_artifact
from kinomlx.models.ltx2.config import (
    assemble,
    resolve_for_execution,
    validate_for_execution,
)
from kinomlx.models.ltx2.restart import DistilledRestart
from kinomlx.models.ltx2.settings import STREAM_TRANSFORMER_RESIDENT_BLOCKS
from kinomlx.settings import Settings
from tests.cli.test_main import _generation_output
from tests.config.saved_records import saved_run_configuration


def _source_run(
    tmp_path: Path,
    *,
    generation: str = "2.5",
    prompt: str = "parent prompt",
    generate: dict[str, object] | None = None,
    model_settings: dict[str, object] | None = None,
    include_stage_1: bool = True,
    include_final: bool = True,
    include_text: bool = True,
    output_fingerprints: dict[str, str] | None = None,
) -> tuple[Path, dict[str, Path]]:
    paths = {
        "video": tmp_path / "parent.mp4",
        "stage_1_latents": tmp_path / "parent_stage1.safetensors",
        "final_latents": tmp_path / "parent.safetensors",
        "text_conditioning": tmp_path / "parent_text.safetensors",
    }
    paths["video"].write_bytes(b"video")
    if include_stage_1:
        paths["stage_1_latents"].write_bytes(b"stage one")
    if include_final:
        paths["final_latents"].write_bytes(b"final")
    if include_text:
        paths["text_conditioning"].write_bytes(b"text")
    outputs = {name: str(path) for name, path in paths.items() if path.is_file()}
    payload = {
        "status": "completed",
        "model": "ltx2",
        **saved_run_configuration(
            "ltx2",
            {
                "model": "ltx2",
                "generate": {
                    "prompt": prompt,
                    "width": 768,
                    "height": 448,
                    "frames": 121,
                    "fps": 24.0,
                    "seed": 42,
                    "generate_audio": True,
                    **(generate or {}),
                },
                "output": {
                    "path": str(paths["video"]),
                    "save_run_log": True,
                },
                "model_settings": {
                    "model_generation": generation,
                    "video_vae": "conv",
                    "video_vae_path": str(tmp_path / "parent-conv-vae.safetensors"),
                    **(model_settings or {}),
                },
                "model_artifacts": {
                    "save_latents": True,
                    "save_text_conditioning": True,
                },
            },
        ),
        "outputs": outputs,
        "generation": {
            "model_generation": generation,
            "video_shape": [1, 3, 121, 448, 768],
        },
    }
    if output_fingerprints is not None:
        payload["output_fingerprints"] = output_fingerprints
    run = tmp_path / "parent_run.json"
    run.write_text(json.dumps(payload), encoding="utf-8")
    return run, paths


def _assemble(run: Path, *arguments: str):
    return assemble(
        build_parser().parse_args(["--restart", str(run), *arguments]),
        base_settings=Settings(),
    )


def test_restart_defaults_to_final_latent_decode_and_safe_new_output(tmp_path: Path) -> None:
    run, paths = _source_run(tmp_path)
    invocation = _assemble(run)

    assert invocation.restart is not None
    assert invocation.restart.config.phase == "decode"
    assert invocation.restart.selected_latent_stage == "final"
    assert invocation.restart.selected_latents == paths["final_latents"]
    assert invocation.request.seed == 42
    assert (invocation.request.width, invocation.request.height, invocation.request.frames) == (
        768,
        448,
        121,
    )
    assert invocation.output.path is None
    assert invocation.output.directory == tmp_path
    assert invocation.output.prefix == "parent_final_decode"
    # Decode re-emits the final latent it read, so the parent's request survives.
    assert invocation.model_artifacts.save_latents is True
    assert invocation.model_artifacts.save_text_conditioning is False

    resolved = resolve_for_execution(invocation, now=datetime(2026, 8, 22, 20, 0, 0))
    assert resolved.output.path == tmp_path / "parent_final_decode_20260822_200000.mp4"
    assert resolved.output.path != paths["video"]


def test_restart_decode_allows_seed_and_downstream_overrides(tmp_path: Path) -> None:
    run, _paths = _source_run(tmp_path, generation="2.3")
    invocation = _assemble(
        run,
        "--seed",
        "99",
        "--video-vae",
        "diffusion",
        "--vae-decode-dtype",
        "bfloat16",
        "--vae-tiling",
        "single",
        "--vsr-spatial-mode",
        "balanced",
        "--target-fps",
        "60",
        "--encode-quality",
        "0.8",
        "--output",
        str(tmp_path / "rerolled.mp4"),
    )

    assert invocation.request.seed == 99
    assert invocation.request.vae_decode_dtype == "bfloat16"
    assert invocation.request.vae_tiling.mode == "single"
    assert invocation.model_settings.video_vae == "diffusion"
    assert invocation.model_settings.video_vae_path is None
    assert invocation.model_settings.model_generation == "2.3"
    assert invocation.output.vsr_spatial_mode == "balanced"
    assert invocation.output.target_fps == 60
    assert invocation.output.encode_quality == pytest.approx(0.8)
    validate_for_execution(invocation)


@pytest.mark.parametrize(
    "arguments",
    [
        ("--prompt", "new prompt"),
        ("--width", "832"),
        ("--lora", "different.safetensors"),
        ("--sampler", "deterministic"),
    ],
)
def test_decode_restart_rejects_changes_to_stations_it_skips(
    arguments: tuple[str, str],
    tmp_path: Path,
) -> None:
    run, _paths = _source_run(tmp_path)
    with pytest.raises(ConfigError, match="belongs to an earlier station"):
        _assemble(run, *arguments)


def test_restart_refusal_names_the_station_and_quotes_the_declared_reason(
    tmp_path: Path,
) -> None:
    run, _paths = _source_run(tmp_path)

    with pytest.raises(ConfigError) as refusal:
        _assemble(run, "--sigmas-stage-2", "0.95,0.8,0.6,0.3,0")

    assert str(refusal.value) == (
        "[restart]: decode restart: [generate].stage_2_sigmas belongs to an earlier "
        "station: The saved final latent already embodies the stage-2 schedule."
    )


def test_decode_restart_accepts_a_parent_prompt_with_trailing_whitespace(tmp_path: Path) -> None:
    """A generated multiline template block parses back with a trailing newline."""
    run, _paths = _source_run(tmp_path, prompt="parent prompt\n")

    invocation = _assemble(run)

    assert invocation.request.prompt == "parent prompt"


def test_decode_restart_accepts_a_streamed_transformer_parent(tmp_path: Path) -> None:
    """The saved preset expands on both sides, so no flags means no change."""
    run, _paths = _source_run(tmp_path, model_settings={"stream_transformer": True})

    invocation = _assemble(run)

    assert invocation.model_settings.stream_transformer is True
    assert (
        invocation.model_settings.transformer_resident_blocks == STREAM_TRANSFORMER_RESIDENT_BLOCKS
    )


def test_restart_rejects_audio_length_policy_change_after_latent_generation(
    tmp_path: Path,
) -> None:
    run, _paths = _source_run(tmp_path)
    with pytest.raises(ConfigError, match=r"\[generate\]\.reference_aligned_audio"):
        _assemble(run, "--reference-aligned-audio")


def test_stage_1_direct_decode_uses_selected_source_product(tmp_path: Path) -> None:
    run, paths = _source_run(tmp_path)
    invocation = _assemble(run, "--latent-stage", "stage-1")

    assert invocation.restart is not None
    assert invocation.restart.selected_latent_stage == "stage-1"
    assert invocation.restart.selected_latents == paths["stage_1_latents"]
    assert invocation.output.prefix == "parent_stage1_decode"


def test_stage_2_restart_requires_and_inherits_saved_text(tmp_path: Path) -> None:
    run, paths = _source_run(tmp_path)
    invocation = _assemble(run, "--restart-from", "stage-2")

    assert invocation.restart is not None
    assert invocation.restart.selected_latents == paths["stage_1_latents"]
    assert invocation.request.text_conditioning == paths["text_conditioning"]
    # Stage-2 re-emits the saved text it consumed, so the parent's request survives.
    assert invocation.model_artifacts.save_text_conditioning is True
    validate_for_execution(invocation)


def test_stage_2_restart_accepts_explicit_text_substitution(tmp_path: Path) -> None:
    run, _paths = _source_run(tmp_path, include_text=False)
    replacement = tmp_path / "replacement_text.safetensors"
    replacement.write_bytes(b"replacement text")
    invocation = _assemble(
        run,
        "--restart-from",
        "stage-2",
        "--text-conditioning",
        str(replacement),
    )

    assert invocation.request.text_conditioning == replacement
    validate_for_execution(invocation)


def test_stage_2_restart_accepts_a_new_stage_2_schedule_and_decode_refuses_it(
    tmp_path: Path,
) -> None:
    """Stage-2 reruns its own schedule; the saved final latent already embodies it."""
    run, _paths = _source_run(tmp_path)
    schedule = "0.95,0.8,0.6,0.3,0"

    invocation = _assemble(run, "--restart-from", "stage-2", "--sigmas-stage-2", schedule)

    assert invocation.request.stage_2_sigmas == (0.95, 0.8, 0.6, 0.3, 0.0)
    assert invocation.request.stage_1_sigmas == ()

    with pytest.raises(ConfigError, match=r"\[generate\]\.stage_2_sigmas"):
        _assemble(run, "--sigmas-stage-2", schedule)


def test_stage_2_restart_rejects_geometry_and_prompt_changes(tmp_path: Path) -> None:
    run, _paths = _source_run(tmp_path)
    with pytest.raises(ConfigError, match=r"\[generate\]\.frames"):
        _assemble(run, "--restart-from", "stage-2", "--frames", "129")
    with pytest.raises(ConfigError, match=r"\[generate\]\.prompt"):
        _assemble(run, "--restart-from", "stage-2", "--prompt", "new prompt")
    with pytest.raises(ConfigError, match=r"\[generate\]\.sampler"):
        _assemble(run, "--restart-from", "stage-2", "--sampler", "deterministic")


def test_restart_print_config_round_trips_as_a_complete_invocation(tmp_path: Path) -> None:
    run, _paths = _source_run(tmp_path)
    first = _assemble(
        run,
        "--seed",
        "99",
        "--video-vae",
        "diffusion",
        "--output",
        str(tmp_path / "rerolled.mp4"),
    )
    config_path = tmp_path / "restart.toml"
    config_path.write_text(dump_config(first.resolved_config), encoding="utf-8")

    second = assemble(
        build_parser().parse_args(["--config", str(config_path), "--print-config"]),
        base_settings=Settings(),
    )

    assert second.resolved_config == first.resolved_config


def test_typed_set_can_select_the_same_restart_contract(tmp_path: Path) -> None:
    run, paths = _source_run(tmp_path)
    invocation = assemble(
        build_parser().parse_args(
            [
                "--set",
                f'restart.run="{run}"',
                "--set",
                'restart.phase="stage-2"',
            ]
        ),
        base_settings=Settings(),
    )

    assert invocation.restart is not None
    assert invocation.restart.config.phase == "stage-2"
    assert invocation.restart.selected_latents == paths["stage_1_latents"]


def test_typed_set_rejects_non_table_restart() -> None:
    options = build_parser().parse_args(["--set", "restart=1", "--print-config"])

    with pytest.raises(ConfigError, match=r"\[restart\] must be a table"):
        assemble(options, base_settings=Settings())


def test_file_config_rejects_non_table_restart(tmp_path: Path) -> None:
    config_path = tmp_path / "restart.toml"
    config_path.write_text("restart = 1\n", encoding="utf-8")
    options = build_parser().parse_args(["--config", str(config_path), "--print-config"])

    with pytest.raises(ConfigError, match=r"\[restart\] must be a table"):
        assemble(options, base_settings=Settings())


def test_restart_table_rejects_non_string_keys() -> None:
    with pytest.raises(ConfigError, match=r"\[restart\] keys must be strings"):
        config_table({"restart": {1: "invalid"}}, "restart")


def test_stage_2_restart_without_text_sidecar_fails_before_execution(tmp_path: Path) -> None:
    run, _paths = _source_run(tmp_path, include_text=False)
    invocation = _assemble(run, "--restart-from", "stage-2")
    with pytest.raises(ConfigError, match="stage-2 requires.*text_conditioning"):
        validate_for_execution(invocation)


def test_stage_2_restart_may_resave_the_text_it_consumed(tmp_path: Path) -> None:
    run, _paths = _source_run(tmp_path)
    invocation = _assemble(
        run,
        "--restart-from",
        "stage-2",
        "--save-text-conditioning",
    )

    validate_for_execution(invocation)
    assert invocation.model_artifacts.save_text_conditioning is True


def test_decode_restart_still_refuses_artifacts_it_cannot_produce(tmp_path: Path) -> None:
    run, _paths = _source_run(tmp_path)
    invocation = _assemble(run, "--save-text-conditioning")

    with pytest.raises(
        ConfigError, match="save_text_conditioning is not produced by a decode restart"
    ):
        validate_for_execution(invocation)


def test_decode_restart_may_resave_the_latent_it_consumed(tmp_path: Path) -> None:
    run, _paths = _source_run(tmp_path)
    invocation = _assemble(run, "--save-latents")

    validate_for_execution(invocation)
    assert invocation.model_artifacts.save_latents is True


def test_restart_latent_substitution_warns_and_records_hash_without_veto(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    run, paths = _source_run(tmp_path)
    expected = file_sha256(paths["final_latents"])
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["output_fingerprints"] = {"final_latents": expected}
    run.write_text(json.dumps(payload), encoding="utf-8")
    replacement = tmp_path / "other-run-final.safetensors"
    replacement.write_bytes(b"different but potentially fitting tensors")

    invocation = _assemble(run, "--restart-latents", str(replacement))
    assert invocation.restart is not None
    with caplog.at_level("WARNING", logger="kinomlx.restart_receipts"):
        record = invocation.restart.to_record(
            text_conditioning=invocation.request.text_conditioning,
            decoder_seed=invocation.request.seed,
        )

    receipt = record["inputs"]["final_latents"]
    assert receipt["path"] == str(replacement)
    assert receipt["sha256"] == file_sha256(replacement)
    assert receipt["parent_sha256"] == expected
    assert receipt["matches_parent"] is False
    assert record["identity_mismatch"] is True
    assert "continuing because identity is observational" in caplog.text


def test_old_parent_without_hash_still_records_consumed_hash(tmp_path: Path) -> None:
    run, paths = _source_run(tmp_path)
    invocation = _assemble(run)
    assert invocation.restart is not None
    record = invocation.restart.to_record(
        text_conditioning=invocation.request.text_conditioning,
        decoder_seed=invocation.request.seed,
    )
    receipt = record["inputs"]["final_latents"]

    assert receipt["sha256"] == file_sha256(paths["final_latents"])
    assert receipt["parent_sha256"] is None
    assert receipt["matches_parent"] is None
    assert record["identity_mismatch"] is False


def test_restart_hash_read_failure_is_observed_but_never_a_load_veto(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import kinomlx.restart_receipts as receipts_module

    run, paths = _source_run(tmp_path)
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["output_fingerprints"] = {"final_latents": file_sha256(paths["final_latents"])}
    run.write_text(json.dumps(payload), encoding="utf-8")
    invocation = _assemble(run)
    assert invocation.restart is not None
    monkeypatch.setattr(
        receipts_module,
        "file_sha256",
        lambda _path: (_ for _ in ()).throw(OSError("hash probe failed")),
    )

    with caplog.at_level("WARNING", logger="kinomlx.restart_receipts"):
        record = invocation.restart.to_record(
            text_conditioning=invocation.request.text_conditioning,
            decoder_seed=invocation.request.seed,
        )

    receipt = record["inputs"]["final_latents"]
    assert receipt["sha256"] is None
    assert receipt["matches_parent"] is None
    assert receipt["hash_error"] == "OSError: hash probe failed"
    assert record["identity_mismatch"] is False
    assert "continuing to structural loading" in caplog.text


def test_restart_rejects_only_missing_selected_artifact_at_manifest_boundary(
    tmp_path: Path,
) -> None:
    run, _paths = _source_run(tmp_path, include_final=False)
    with pytest.raises(ConfigError, match="no completed final_latents output"):
        _assemble(run)


def test_restart_cannot_overwrite_parent_video(tmp_path: Path) -> None:
    run, paths = _source_run(tmp_path)
    invocation = _assemble(run, "--output", str(paths["video"]))
    with pytest.raises(ConfigError, match="must not overwrite"):
        resolve_for_execution(invocation)


def test_restart_flags_without_parent_run_are_rejected() -> None:
    with pytest.raises(ConfigError, match=r"\[restart\].run is required"):
        assemble(
            build_parser().parse_args(["--restart-from", "decode", "--print-config"]),
            base_settings=Settings(),
        )


def test_cli_dispatches_the_public_restart_recipe_and_records_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kinomlx.cli.main as main_module
    import kinomlx.models.ltx2.command as command_module

    run, paths = _source_run(tmp_path)
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["output_fingerprints"] = {"final_latents": file_sha256(paths["final_latents"])}
    run.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "restarted.mp4"
    calls = {}

    class _Runner:
        def __init__(self, artifact_sink):
            self._artifacts = artifact_sink

        def restart(self, request, restart):
            calls["restart"] = restart
            calls["request"] = request
            # A decode restart re-emits the latent product it consumed, so a
            # run that finishes normally hands it to the sink.
            self._artifacts.save(
                distilled_stage_latents_artifact(
                    2,
                    video_latent=mx.zeros((1, 1, 1, 1), dtype=mx.bfloat16),
                    audio_latent=None,
                    final=True,
                )
            )
            return _generation_output()

        def run(self, recipe, request):
            pytest.fail("restart CLI must use the public runner restart method")

    def fake_write_generation(*_args, **_kwargs):
        output.touch()
        return output

    monkeypatch.setattr(
        command_module, "create_runner", lambda *_args, **kwargs: _Runner(kwargs["artifact_sink"])
    )
    monkeypatch.setattr(command_module, "write_generation", fake_write_generation)

    assert (
        main_module.main(
            [
                "--quiet",
                "--restart",
                str(run),
                "--seed",
                "99",
                "--no-generate-audio",
                "--output",
                str(output),
                "--save-run-log",
            ]
        )
        == 0
    )

    restart = calls["restart"]
    assert isinstance(restart, DistilledRestart)
    assert restart == DistilledRestart.decode(
        paths["final_latents"],
        source_model_generation="2.5",
    )
    assert calls["request"].seed == 99
    record = json.loads((tmp_path / "restarted_run.json").read_text(encoding="utf-8"))
    assert record["restart"]["decoder_seed"] == 99
    assert record["restart"]["identity_mismatch"] is False
    assert record["restart"]["inputs"]["final_latents"]["sha256"] == file_sha256(
        paths["final_latents"]
    )
    # The re-emitted latent is a fresh output of this run, fingerprinted as one.
    assert set(record["output_fingerprints"]) == {"final_latents"}


def test_a_layer_asking_for_both_tiling_modes_is_refused(tmp_path: Path) -> None:
    """F-29: a tile value and a non-custom mode in one layer say two things."""
    config = tmp_path / "tiling.toml"
    config.write_text(
        '[generate]\nprompt = "a clip"\n\n'
        '[generate.vae_tiling]\nmode = "single"\nspatial_tile_pixels = 256\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as error:
        assemble(
            build_parser().parse_args(["--config", str(config), "--print-config"]),
            base_settings=Settings(),
        )
    assert str(error.value) == (
        f'{config}: [generate].vae_tiling: tile values require mode = "custom"'
    )


def test_an_overlap_beside_an_inactive_mode_stays_a_redundancy(tmp_path: Path) -> None:
    """F-29 boundary: an overlap has a default that legacy whole dumps wrote out."""
    config = tmp_path / "overlap.toml"
    config.write_text(
        '[generate]\nprompt = "a clip"\n\n'
        '[generate.vae_tiling]\nmode = "single"\ntemporal_overlap_frames = 16\n',
        encoding="utf-8",
    )
    invocation = assemble(
        build_parser().parse_args(["--config", str(config), "--print-config"]),
        base_settings=Settings(),
    )
    assert invocation.request.vae_tiling.mode == "single"


def test_a_tile_value_alone_still_selects_custom_tiling(tmp_path: Path) -> None:
    invocation = assemble(
        build_parser().parse_args(
            ["--prompt", "a clip", "--vae-spatial-tile-pixels", "256", "--print-config"]
        ),
        base_settings=Settings(),
    )
    assert invocation.request.vae_tiling.mode == "custom"
    assert invocation.request.vae_tiling.spatial_tile_pixels == 256


def test_a_decode_restart_may_switch_a_custom_parent_to_single(tmp_path: Path) -> None:
    """F-29: a lower layer's geometry is precedence, not a contradiction."""
    run, _paths = _source_run(
        tmp_path,
        generate={"vae_tiling": {"mode": "custom", "spatial_tile_pixels": 256}},
    )
    invocation = _assemble(run, "--vae-tiling", "single")
    assert invocation.request.vae_tiling.mode == "single"
    assert invocation.request.vae_tiling.spatial_tile_pixels is None
