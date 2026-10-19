"""Common image command behavior, precedence, and saved-stage execution."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from kinomlx.cli.main import main
from kinomlx.config import ConfigError
from kinomlx.io.image import save_image
from kinomlx.io.output_bundle import BundleOutputError
from kinomlx.io.output_config import ImageOutputConfig
from kinomlx.memory_policy import MLXMemoryPolicy
from kinomlx.model_registry import config_registry
from kinomlx.models.flux2 import command
from kinomlx.models.flux2.arguments import build_parser
from kinomlx.models.flux2.catalog import Flux2Variant
from kinomlx.models.flux2.config import assemble
from kinomlx.models.flux2.generation import Flux2ImageResult
from kinomlx.models.flux2.output import Flux2OutputSink, plan_output
from kinomlx.models.flux2.pipelines import restart as restart_pipeline
from kinomlx.models.flux2.pipelines import text_to_image as pipeline
from kinomlx.models.flux2.restart import RestartManifest
from kinomlx.models.flux2.sampling import SamplingRecipe, schedule
from kinomlx.models.flux2.types import Flux2RestartConfig
from kinomlx.models.krea2.arguments import build_parser as krea_parser
from kinomlx.restart_selection import fingerprint_key
from tests.config.saved_records import saved_run_configuration
from tests.models.flux2.test_text_conditioning import conditioning


@dataclass
class _Resources:
    profile: object
    memory_policy: MLXMemoryPolicy = MLXMemoryPolicy()
    describe: object = dict
    source_root: object = None
    revision: str = "fixture"
    recipe_identity: str = "fixture"
    components: tuple = ()
    component: object = None


def test_dry_run_prints_for_a_human_unless_json_is_selected(monkeypatch, capsys, caplog):
    """A resource-only run honors the switch every other command honors."""
    report = {"model": "flux2", "variant": "distilled", "components": []}
    monkeypatch.setattr(
        command,
        "prepare_resources",
        lambda *a, **k: _Resources(
            describe=lambda: report,
            profile=SimpleNamespace(variant=Flux2Variant.DISTILLED, kv=False),
        ),
    )

    with caplog.at_level(logging.INFO, logger="kinomlx"):
        assert main(["--model", "flux2", "--dry-run"]) == 0
    assert "Resources" in caplog.text
    assert capsys.readouterr().out == ""

    assert main(["--model", "flux2", "--dry-run", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == report


def test_common_non_adapter_image_flags_have_identical_parser_behavior():
    parsers = [build_parser(), krea_parser()]
    controls = [
        {flag: action for action in parser._actions for flag in action.option_strings}
        for parser in parsers
    ]
    for flag in (
        "--prompt",
        "--prompt-file",
        "--negative-prompt",
        "--source-image",
        "--match-source-aspect",
        "--reference-fit",
        "--output",
        "--output-dir",
        "--output-prefix",
        "--seed",
        "--images-per-prompt",
        "--save-all-sidecars",
        "--no-save-run-log",
        "--restart",
        "--restart-from",
        "--restart-latents",
        "--text-conditioning",
        "--json",
        "--no-json",
    ):
        left, right = (control[flag] for control in controls)
        assert (type(left), left.dest, left.type, left.nargs, left.const, left.default) == (
            type(right),
            right.dest,
            right.type,
            right.nargs,
            right.const,
            right.default,
        ), flag
    assert parsers[0].parse_args(["--match-source-aspect"]).match_source_aspect == "long"


def test_prompt_file_layers_and_output_selection(tmp_path):
    text = tmp_path / "prompt.txt"
    text.write_text("A complete\nmultiline prompt.")
    config = tmp_path / "config.toml"
    config.write_text(
        '[generate]\nprompt="lower"\nseed=7\n[output]\npath="lower.png"\nsave_run_log=false\n'
    )
    options = build_parser().parse_args(
        [
            "--config",
            str(config),
            "--prompt-file",
            str(text),
            "--seed",
            "11",
            "--set",
            "generate.seed=42",
            "--output-dir",
            str(tmp_path),
            "--save-all-sidecars",
            "--no-save-console-log",
        ]
    )
    invocation = assemble(options)
    assert invocation.generate.prompts() == ("A complete\nmultiline prompt.",)
    assert invocation.generate.seed == 42
    assert invocation.input_paths == (text,)
    assert invocation.output.path is None
    assert invocation.output.save_run_log is True
    assert invocation.output.save_console_log is False
    assert "prompt_file" not in invocation.resolved_config["generate"]
    text.unlink()
    assert invocation.generate.prompts()[0].startswith("A complete")


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--prompt", "test", "--negative-prompt", "bad"], "active base CFG"),
        (["--prompt", "test", "--guidance", "4"], "base recipe only"),
        (["--prompt", "test", "--cfg-mode", "batched"], "base model"),
        (["--prompt", "test", "--reference-fit", "crop"], "reference_fit requires source_images"),
        (["--prompt", "test", "--width", "65"], "multiples of 16"),
        (["--prompt", "test", "--images-per-prompt", "0"], "greater than or equal to 1"),
    ],
)
def test_ineffective_controls_fail_before_model_loading(arguments, message):
    with pytest.raises(ValueError, match=message):
        assemble(build_parser().parse_args(arguments))


def _result(*, base=False):
    stage = conditioning(base=base)
    latent = mx.zeros((1, 16, 128), dtype=mx.float16)
    return Flux2ImageResult(
        mx.full((64, 64, 3), 0.5), latent, 64, 64, (), (), stage, latent, schedule(16, 4)
    )


def _fake_generation(monkeypatch, *, base=False):
    calls = []
    monkeypatch.setattr(
        command,
        "prepare_resources",
        lambda *a, **k: _Resources(
            memory_policy=MLXMemoryPolicy(),
            describe=lambda: {"variant": "base" if base else "distilled", "components": []},
            profile=SimpleNamespace(
                variant=Flux2Variant.BASE if base else Flux2Variant.DISTILLED, kv=False
            ),
            source_root=None,
            revision="fixture",
            recipe_identity="fixture",
            components=(),
        ),
    )

    def generate(*args, **kwargs):
        calls.append(kwargs)
        yield kwargs.get("conditioning")
        yield None
        return _result(base=base)

    monkeypatch.setattr(pipeline, "image_stages", generate)
    return calls


@pytest.mark.parametrize(
    "changes",
    [
        ["--negative-prompt", "blurry"],
        ["--reference-fit", "crop"],
        ["--match-source-aspect"],
        ["--negative-prompt", "blurry", "--reference-fit", "fit"],
    ],
)
def test_restart_optional_changes_report_earlier_stage(tmp_path, monkeypatch, changes):
    _fake_generation(monkeypatch)
    assert (
        command.run_flux2_command(
            [
                "--prompt",
                "example",
                "--width",
                "64",
                "--height",
                "64",
                "--save-all-sidecars",
                "--output",
                str(tmp_path / "first.png"),
                "--quiet",
            ]
        )
        == 0
    )
    with pytest.raises(ValueError, match="earlier stage"):
        assemble(
            build_parser().parse_args(
                [
                    "--restart",
                    str(tmp_path / "first_run.json"),
                    "--restart-from",
                    "denoise",
                    *changes,
                ]
            )
        )


def test_arithmetic_failure_is_rendered_as_json(tmp_path, monkeypatch, capsys):
    _fake_generation(monkeypatch)

    def fail(*args, **kwargs):
        raise FloatingPointError("nonfinite latents at step 1")

    monkeypatch.setattr(pipeline, "image_stages", fail)
    assert (
        command.run_flux2_command(
            [
                "--prompt",
                "example",
                "--width",
                "64",
                "--height",
                "64",
                "--output",
                str(tmp_path / "failed.png"),
                "--json",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "nonfinite latents" in captured.out
    assert "Traceback" not in captured.out + captured.err
    json.loads(captured.out)


@pytest.mark.parametrize(
    "arguments", [["--restart-from", "denoise"], ["--restart-latents", "replacement.safetensors"]]
)
def test_restart_switch_requires_run(arguments):
    with pytest.raises(ValueError, match="run.*required"):
        assemble(build_parser().parse_args(arguments))


def test_cli_publishes_complete_bundle_and_restarts_without_encoders(tmp_path, monkeypatch):
    calls = _fake_generation(monkeypatch, base=True)
    output = tmp_path / "first.png"
    arguments = [
        "--variant",
        "base",
        "--prompt",
        "a painted bowl",
        "--width",
        "64",
        "--height",
        "64",
        "--steps",
        "4",
        "--save-all-sidecars",
        "--output",
        str(output),
        "--quiet",
    ]
    assert command.run_flux2_command(arguments) == 0
    run = tmp_path / "first_run.json"
    data = json.loads(run.read_text())
    assert data["status"] == "completed"
    assert set(data["output_fingerprints"]) == {
        "images",
        "initial_noise",
        "final_latents",
        "text_conditioning",
    }
    assert data["generation"]["images"][0]["schedule"][-1] == 0
    monkeypatch.setattr(
        command,
        "prepare_resources",
        lambda *a, **k: pytest.fail("restart reopens source resources"),
    )
    monkeypatch.setattr(
        "kinomlx.models.flux2.restart.RestartManifest.resources",
        lambda *a, **k: _Resources(
            memory_policy=MLXMemoryPolicy(),
            profile=SimpleNamespace(variant=Flux2Variant.BASE),
            component=lambda kind: SimpleNamespace(source_path=tmp_path),
        ),
    )
    assert (
        command.run_flux2_command(
            [
                "--restart",
                str(run),
                "--restart-from",
                "denoise",
                "--output",
                str(tmp_path / "replay.png"),
                "--quiet",
            ]
        )
        == 0
    )
    assert calls[-1]["conditioning"].negative is not None
    monkeypatch.setattr(restart_pipeline, "decode_latents", lambda *a, **k: mx.zeros((64, 64, 3)))
    assert (
        command.run_flux2_command(
            ["--restart", str(run), "--output", str(tmp_path / "decode.png"), "--quiet"]
        )
        == 0
    )
    assert len(calls) == 2
    decoded = json.loads((tmp_path / "decode_run.json").read_text())
    assert "initial_noise" not in decoded["outputs"]
    assert "text_conditioning" not in decoded["outputs"]


def test_output_protects_source_hardlinks_even_with_force(tmp_path):
    source = tmp_path / "source.png"
    alias = tmp_path / "alias.png"
    save_image(source, mx.zeros((64, 64, 3)))
    alias.hardlink_to(source)
    with pytest.raises(BundleOutputError, match="overwrite an input"):
        plan_output(
            ImageOutputConfig(path=alias, force=True),
            variant="kv",
            seed=0,
            image_count=1,
            sources=(source,),
        )


def test_failed_bundle_publish_restores_prior_files(tmp_path, monkeypatch):
    plan = plan_output(
        ImageOutputConfig(path=tmp_path / "image.png", force=True, save_final_latents=True),
        variant="distilled",
        seed=0,
        image_count=1,
    )
    for target in plan.targets():
        target.write_bytes(b"prior")
    rename = Path.rename
    failed = False

    def fail_one_publish(path, target):
        nonlocal failed
        if (
            not failed
            and path.name.endswith("_final_latents.safetensors")
            and path.parent.name.startswith(".flux2-output-")
        ):
            failed = True
            raise OSError("disk full")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_one_publish)
    with plan.reserve() as reservation:
        sink = Flux2OutputSink(plan)
        with pytest.raises(OSError, match="disk full"):
            sink.write([_result()], SamplingRecipe(), reservation=reservation)
        assert sink.fingerprints == {}
    assert all(path.read_bytes() == b"prior" for path in plan.targets())
    assert not list(tmp_path.glob(".flux2-output-*"))


def test_documented_downloads_skip_only_duplicate_monoliths():
    from kinomlx.models.flux2.catalog import PROFILES

    root = Path(__file__).resolve().parents[3]
    guide = (root / "docs/flux2/downloads.md").read_text()
    for profile in PROFILES:
        assert f'hf download {profile.repo_id} --exclude "flux-2-klein-*.safetensors"' in guide
    assert "--include" not in guide


def test_console_sidecar_failure_keeps_successful_image(tmp_path, monkeypatch):
    _fake_generation(monkeypatch)

    def fail_log(*args, **kwargs):
        raise OSError("log device full")

    monkeypatch.setattr(command, "initialize_execution_log", fail_log)
    assert (
        command.run_flux2_command(
            [
                "--prompt",
                "a painted bowl",
                "--output",
                str(tmp_path / "image.png"),
                "--save-all-sidecars",
                "--quiet",
            ]
        )
        == 0
    )
    record = json.loads((tmp_path / "image_run.json").read_text())
    assert record["status"] == "completed"
    assert record["sidecar_errors"][0]["artifact"] == "execution_log"
    assert "execution_log" not in record["outputs"]
    assert record["diagnostics"]["memory"]["counter_source"] == "mlx_allocator"


def test_cancel_marks_run_aborted_and_preserves_primaries(tmp_path, monkeypatch):
    _fake_generation(monkeypatch)

    def cancel(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "image_stages", cancel)
    output = tmp_path / "image.png"
    output.write_bytes(b"prior image")
    # Cancellation is rendered once at the CLI boundary, as it is for every model.
    assert (
        main(
            [
                "--model",
                "flux2",
                "--prompt",
                "a painted bowl",
                "--output",
                str(output),
                "--force",
                "--save-all-sidecars",
                "--quiet",
            ]
        )
        == 130
    )
    assert output.read_bytes() == b"prior image"
    record = json.loads((tmp_path / "image_run.json").read_text())
    assert record["status"] == "aborted"
    assert not list(tmp_path.glob("*.lock"))


def test_replacement_conditioning_does_not_reopen_missing_original(tmp_path, monkeypatch):
    _fake_generation(monkeypatch, base=True)
    assert (
        command.run_flux2_command(
            [
                "--variant",
                "base",
                "--prompt",
                "a painted bowl",
                "--width",
                "64",
                "--height",
                "64",
                "--output",
                str(tmp_path / "image.png"),
                "--save-all-sidecars",
                "--quiet",
            ]
        )
        == 0
    )
    original = tmp_path / "image_text.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    original.rename(replacement)
    selected = assemble(
        build_parser().parse_args(
            [
                "--restart",
                str(tmp_path / "image_run.json"),
                "--restart-from",
                "denoise",
                "--text-conditioning",
                str(replacement),
            ]
        )
    )
    assert selected.conditioning.negative is not None
    assert selected.generate.text_conditioning == replacement


def test_saved_conditioning_cannot_silently_replace_explicit_sources(tmp_path):
    from kinomlx.models.flux2.text_conditioning import save_conditioning

    path = tmp_path / "text.safetensors"
    save_conditioning(path, conditioning(), SamplingRecipe())
    with pytest.raises(ValueError, match="replaces source images"):
        assemble(
            build_parser().parse_args(
                [
                    "--text-conditioning",
                    str(path),
                    "--source-image",
                    str(tmp_path / "source.png"),
                ]
            )
        )


def test_explicit_matching_negative_without_prompt_uses_saved_prompt(tmp_path):
    from kinomlx.models.flux2.catalog import Flux2Variant
    from kinomlx.models.flux2.text_conditioning import save_conditioning

    path = tmp_path / "text.safetensors"
    save_conditioning(path, conditioning(base=True), SamplingRecipe(Flux2Variant.BASE))
    result = assemble(
        build_parser().parse_args(
            [
                "--variant",
                "base",
                "--text-conditioning",
                str(path),
                "--negative-prompt",
                "",
            ]
        )
    )
    assert result.generate.prompts() == ("a painted bowl",)


def test_no_generation_selection_requires_a_prompt_or_inspection():
    # The message names the configuration paths, not one surface's flags.
    with pytest.raises(
        ValueError,
        match=(
            r"\[generate\].prompt, \[generate\].prompt_file, "
            r"\[generate\].text_conditioning, or \[restart\].run is required"
        ),
    ):
        assemble(build_parser().parse_args(["--variant", "kv"]))


def test_decode_after_text_replay_does_not_reopen_original_text(tmp_path, monkeypatch):
    from kinomlx.models.flux2.text_conditioning import save_conditioning

    _fake_generation(monkeypatch)
    text = tmp_path / "input.safetensors"
    save_conditioning(text, conditioning(), SamplingRecipe())
    assert (
        command.run_flux2_command(
            [
                "--text-conditioning",
                str(text),
                "--output",
                str(tmp_path / "image.png"),
                "--save-all-sidecars",
                "--quiet",
            ]
        )
        == 0
    )
    text.unlink()
    run = tmp_path / "image_run.json"
    payload = json.loads(run.read_text())
    # Explicit paths remain typed paths after the JSON round trip, even when sources are gone.
    payload["configuration"]["values"].setdefault("model_settings", {})["source_path"] = str(
        tmp_path / "deleted-source"
    )
    run.write_text(json.dumps(payload))
    monkeypatch.setattr(
        command, "prepare_resources", lambda *a, **k: pytest.fail("decode reopens sources")
    )
    monkeypatch.setattr(
        "kinomlx.models.flux2.restart.RestartManifest.resources",
        lambda *a, **k: _Resources(
            memory_policy=MLXMemoryPolicy(),
            profile=SimpleNamespace(variant=Flux2Variant.DISTILLED),
            component=lambda kind: SimpleNamespace(source_path=tmp_path),
        ),
    )
    monkeypatch.setattr(restart_pipeline, "decode_latents", lambda *a, **k: mx.zeros((64, 64, 3)))
    assert (
        command.run_flux2_command(
            [
                "--restart",
                str(run),
                "--output",
                str(tmp_path / "decode.png"),
                "--quiet",
            ]
        )
        == 0
    )


def test_grouped_adapters_keep_default_and_explicit_strengths_independent():
    invocation = assemble(
        build_parser().parse_args(
            [
                "--prompt",
                "a painted bowl",
                "--adapter",
                "turbo.safetensors",
                "--adapter",
                "style.safetensors",
                "--adapter-strength",
                "0.8",
            ]
        )
    )
    assert [item.strength for item in invocation.generate.resolved_adapters()] == [1.0, 0.8]


def test_klein_header_matches_image_run_sections(tmp_path, monkeypatch, caplog):
    import logging

    _fake_generation(monkeypatch)
    with caplog.at_level(logging.INFO, logger="kinomlx"):
        assert (
            command.run_flux2_command(
                [
                    "--prompt",
                    "example",
                    "--width",
                    "64",
                    "--height",
                    "64",
                    "--output",
                    str(tmp_path / "image.png"),
                ]
            )
            == 0
        )
    text = caplog.text
    headings = ["Run:", "Compute:", "Prompt:", "Planned outputs:", "Resources:", "Sampling:"]
    offsets = [text.index(heading) for heading in headings]
    assert offsets == sorted(offsets)
    assert "Model: FLUX.2 Klein Distilled | Action: generate" in text
    assert "Resolution: 64x64" in text
    assert "Noise backend: mlx" in text
    assert "Source aspect:" not in text


@pytest.mark.parametrize("json_mode", [False, True])
def test_restart_without_current_image_metadata_reports_error(tmp_path, capsys, json_mode):
    run = tmp_path / "run.json"
    run.write_text(
        json.dumps(
            {
                "model": "flux2",
                "status": "completed",
                **saved_run_configuration(
                    "flux2", {"model": "flux2", "generate": {"prompt": "scene"}}
                ),
                "generation": {},
            }
        )
    )
    from kinomlx.ui import configure_logging

    args = ["--restart", str(run), "--restart-from", "decode", "--transformer-dtype", "float16"]
    if json_mode:
        args.append("--json")
    # kinomlx.cli.main configures logging before it dispatches to a model command,
    # so a test that calls the command directly supplies that step itself instead
    # of depending on a console handler another test happened to leave installed.
    configure_logging()
    assert command.run_flux2_command(args) == 2
    output = capsys.readouterr()
    # The console wraps at its own width, which varies with the temporary path.
    assert "records no image 1" in " ".join((output.out + output.err).split())
    if json_mode:
        assert isinstance(json.loads(output.out), dict)


@pytest.mark.parametrize("mode", ["sequential", "stage", "resident"])
@pytest.mark.parametrize("failure", [KeyboardInterrupt, RuntimeError])
def test_finished_seed_files_survive_next_seed_failure(tmp_path, monkeypatch, mode, failure):
    from kinomlx.io.fingerprints import file_sha256
    from kinomlx.videotoolbox.images import load_image_rgb

    _fake_generation(monkeypatch)
    saved = {}

    def generate(*args, **kwargs):
        yield kwargs.get("conditioning")
        yield None
        if kwargs["seed"] == 2:
            image = tmp_path / "result-001-seed9.png"
            assert load_image_rgb(image).shape == (64, 64, 3)
            sidecars = list(tmp_path.glob("result-001-seed9_*.safetensors"))
            assert len(sidecars) == 3
            for path in sidecars:
                assert mx.load(str(path))
            saved.update({path: file_sha256(path) for path in [image, *sidecars]})
            raise failure("stop remaining seeds")
        return _result()

    monkeypatch.setattr(pipeline, "image_stages", generate)
    assert main(
        [
            "--model",
            "flux2",
            "--prompt",
            "test",
            "--width",
            "64",
            "--height",
            "64",
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
    ) == (130 if failure is KeyboardInterrupt else 2)
    assert (tmp_path / "result_console.log").stat().st_size > 0
    assert saved
    assert all(file_sha256(path) == digest for path, digest in saved.items())
    assert not (tmp_path / "result-002-seed2.png").exists()
    record = json.loads((tmp_path / "result_run.json").read_text())
    assert record["status"] == ("aborted" if failure is KeyboardInterrupt else "failed")
    assert record["outputs"]["images"] == [str(tmp_path / "result-001-seed9.png")]
    # The digests below carry the planned-count suffix, so the record states it.
    assert record["planned_image_count"] == 2
    assert sorted(record["output_fingerprints"]) == [
        "final_latents.0",
        "images.0",
        "initial_noise.0",
        "text_conditioning.0",
    ]

    source = tmp_path / "result_run.json"
    if failure is KeyboardInterrupt:
        # A canceled run is restartable for the image it finished. Its digests
        # were keyed by the planned count, so the artifact still resolves.
        manifest = RestartManifest.load(
            Flux2RestartConfig(run=source, index=1), schema=config_registry().model("flux2")
        )
        assert manifest.source_image_count == 2
        assert manifest.artifact("final_latents") == (
            tmp_path / "result-001-seed9_final_latents.safetensors"
        )
        assert (
            manifest.base_config(schema=config_registry().model("flux2")).values["generate"]["seed"]
            == 9
        )
        # Image 2 was planned and never written.
        with pytest.raises(ValueError, match="image 2 of 2 was not published before cancellation"):
            RestartManifest.load(
                Flux2RestartConfig(run=source, index=2), schema=config_registry().model("flux2")
            )
    else:
        # A failed run stopped for a reason it cannot describe; the record is
        # not offered to a restart at all.
        with pytest.raises(ValueError, match="completed or canceled"):
            RestartManifest.load(
                Flux2RestartConfig(run=source, index=1), schema=config_registry().model("flux2")
            )


@pytest.mark.parametrize("count", [1, 3])
def test_per_image_digest_keys_are_the_shared_restart_key(tmp_path, count):
    plan = plan_output(
        ImageOutputConfig(
            directory=tmp_path,
            prefix="bundle",
            save_initial_noise=True,
            save_final_latents=True,
        ),
        variant="distilled",
        seed=9,
        image_count=count,
    )
    sink = Flux2OutputSink(plan)
    with plan.reserve() as reservation:
        for index in range(count):
            sink.write_image(index, _result(), SamplingRecipe(), reservation=reservation)

    # A restart reads these digests back through fingerprint_key, so the writer
    # must spell them with that one function, not a second inline convention.
    assert set(sink.fingerprints) == {
        fingerprint_key(name, index=index, count=count)
        for name in ("images", "initial_noise", "final_latents")
        for index in range(count)
    }


def test_completed_sidecar_survives_later_sidecar_write_failure(tmp_path, monkeypatch):
    import kinomlx.models.flux2.output as output_module

    plan = plan_output(
        ImageOutputConfig(path=tmp_path / "result.png", save_all_sidecars=True),
        variant="distilled",
        seed=9,
        image_count=1,
    )
    sink = Flux2OutputSink(plan)
    original = output_module.save_weights

    def write(path, tensors, metadata):
        if "final_latents" in tensors:
            raise OSError("disk full")
        original(path, tensors, metadata)

    monkeypatch.setattr(output_module, "save_weights", write)
    with plan.reserve() as reservation, pytest.raises(OSError, match="disk full"):
        sink.write_image(0, _result(), SamplingRecipe(), reservation=reservation)
    noise = dict(plan.artifacts)["initial_noise"][0]
    assert "initial_noise" in mx.load(str(noise))
    assert str(noise) in sink.completed_outputs()["initial_noise"]


def test_klein_configuration_errors_carry_the_shared_prefix(caplog):
    import logging

    with caplog.at_level(logging.ERROR, logger="kinomlx"):
        exit_code = command.run_flux2_command(
            ["--prompt", "scene", "--steps", "0", "--print-config"]
        )
    assert exit_code == 2
    assert (
        "config error: command line: [generate].steps: must be greater than or equal to 1"
        in caplog.text
    )
    assert "Traceback" not in caplog.text


def test_klein_internal_failure_is_presented_as_a_model_failure(tmp_path, monkeypatch, caplog):
    import logging

    _fake_generation(monkeypatch)

    def fail(*args, **kwargs):
        raise RuntimeError("injected internal defect")

    monkeypatch.setattr(pipeline, "image_stages", fail)
    with caplog.at_level(logging.ERROR, logger="kinomlx"):
        exit_code = command.run_flux2_command(
            [
                "--prompt",
                "example",
                "--width",
                "64",
                "--height",
                "64",
                "--output",
                str(tmp_path / "failed.png"),
            ]
        )
    assert exit_code == 2
    rendered = " ".join(caplog.text.split())
    assert "config error:" not in rendered
    assert "FLUX.2 Klein generation failed: injected internal defect" in rendered


@pytest.mark.parametrize(
    ("layer", "expected_source"),
    [("--set", "--set"), ("--config", None)],
)
def test_both_prompt_keys_in_one_layer_name_that_layer(tmp_path, layer, expected_source):
    if layer == "--set":
        arguments = ["--set", "generate.prompt=a", "--set", "generate.prompt_file=b.txt"]
        source = expected_source
    else:
        config = tmp_path / "both.toml"
        config.write_text('[generate]\nprompt = "a"\nprompt_file = "b.txt"\n', encoding="utf-8")
        arguments = ["--config", str(config)]
        source = str(config)
    with pytest.raises(ConfigError) as error:
        assemble(build_parser().parse_args([*arguments, "--print-config"]))
    assert str(error.value) == (
        f"{source}: [generate]: prompt and prompt_file cannot both be set in the same layer"
    )


def test_a_configuration_error_reaches_the_product_console_without_main(capsys):
    """A direct caller of the Klein command sees the same console as `kinomlx`.

    The command configures logging before assembling, so a refusal raised by
    assembly is rendered by the product handler with its level label rather than
    falling through to logging's last-resort handler, which prints the bare text.
    """
    exit_code = command.run_flux2_command(["--prompt", "scene", "--steps", "0", "--print-config"])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "config error: command line: [generate].steps: must be greater than or equal to 1" in (
        " ".join(err.split())
    )
    assert "ERROR" in err


def test_recompute_reference_kv_is_a_public_input_on_every_surface(tmp_path, monkeypatch):
    """The K/V recomputation switch reaches the recipe from the CLI, TOML and the environment."""
    cli = assemble(
        build_parser().parse_args(
            ["--prompt", "scene", "--variant", "kv", "--recompute-reference-kv", "--print-config"]
        )
    )
    assert cli.generate.recompute_reference_kv is True
    assert cli.generate.recipe("kv").recompute_reference_kv is True
    assert cli.resolved_config["generate"]["recompute_reference_kv"] is True

    config = tmp_path / "kv.toml"
    config.write_text(
        '[generate]\nprompt = "scene"\nrecompute_reference_kv = true\n'
        '[model_settings]\nvariant = "kv"\n',
        encoding="utf-8",
    )
    toml = assemble(build_parser().parse_args(["--config", str(config), "--print-config"]))
    assert toml.generate.recompute_reference_kv is True

    monkeypatch.setenv("KINO_FLUX2_RECOMPUTE_REFERENCE_KV", "true")
    env = assemble(
        build_parser().parse_args(["--prompt", "scene", "--variant", "kv", "--print-config"])
    )
    assert env.generate.recompute_reference_kv is True
    off = assemble(
        build_parser().parse_args(
            [
                "--prompt",
                "scene",
                "--variant",
                "kv",
                "--no-recompute-reference-kv",
                "--print-config",
            ]
        )
    )
    assert off.generate.recompute_reference_kv is False


@pytest.mark.parametrize("variant", ["base", "distilled"])
def test_recompute_reference_kv_requires_the_kv_variant(variant):
    with pytest.raises(ConfigError, match="recompute_reference_kv requires the KV variant"):
        assemble(
            build_parser().parse_args(
                [
                    "--prompt",
                    "scene",
                    "--variant",
                    variant,
                    "--recompute-reference-kv",
                    "--print-config",
                ]
            )
        )
