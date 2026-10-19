"""Four-step recipe admission and offline adapter selection."""

from dataclasses import replace
from pathlib import Path

import pytest

from kinomlx.adapters.selection import AdapterSelection
from kinomlx.config import ConfigError
from kinomlx.config_rules import validate_record
from kinomlx.models.krea2.adapters.turbo_profile import (
    TURBO_ADAPTER_FILENAME,
    TURBO_ADAPTER_REPO,
    resolve_turbo_adapters,
)
from kinomlx.models.krea2.arguments import build_parser
from kinomlx.models.krea2.config import assemble
from kinomlx.models.krea2.settings import Krea2Settings
from kinomlx.models.krea2.types import Krea2Request
from kinomlx.settings import Settings


@pytest.fixture
def profile_settings(tmp_path):
    repo = tmp_path / "hub" / ("models--" + TURBO_ADAPTER_REPO.replace("/", "--"))
    snapshot = repo / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / TURBO_ADAPTER_FILENAME).touch()
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("a" * 40)
    return Settings(hf_home=tmp_path)


def test_profile_cli_environment_and_matching_overrides(monkeypatch, profile_settings):
    monkeypatch.setenv("KINO_KREA2_TURBO_4STEP", "true")
    monkeypatch.setenv("KINO_KREA2_TURBO_4STEP_STRENGTH", "0.75")
    invocation = assemble(
        build_parser().parse_args(["--model", "krea2", "--prompt", "scene"]),
        base_settings=profile_settings,
        base_model_settings=Krea2Settings(),
    )
    request = invocation.request
    assert request is not None
    assert request.turbo_4step
    assert request.turbo_4step_strength == 0.75
    assert (request.steps, request.guidance, request.mu) == (4, 0, 1.15)
    # The recipe is a default, not a lock: an explicit step count is honored and
    # reported, because leaving a distilled recipe is a choice worth announcing.
    overridden = assemble(
        build_parser().parse_args(["--model", "krea2", "--prompt", "scene", "--steps", "8"]),
        base_settings=profile_settings,
        base_model_settings=Krea2Settings(),
    ).request
    assert overridden is not None
    assert overridden.steps == 8
    assert overridden.profile_deviations() == ("steps 8 instead of 4",)


@pytest.mark.parametrize(
    "change",
    [
        {"turbo_4step_strength": 0},
        {"turbo_4step_strength": float("nan")},
    ],
)
def test_profile_rejects_an_unusable_strength(change):
    with pytest.raises(ValueError, match="turbo_4step_strength"):
        Krea2Request(prompt="scene", turbo_4step=True, **change)


@pytest.mark.parametrize(
    ("change", "reported"),
    [
        ({"steps": 8}, "steps 8 instead of 4"),
        ({"guidance": 1.0}, "guidance 1.0 instead of 0.0"),
        ({"mu": 0.8}, "mu 0.8 instead of 1.15"),
        (
            {"sigmas": (1.0, 0.9, 0.5, 0.0)},
            "an explicit sigma schedule instead of the distilled one",
        ),
    ],
)
def test_profile_recipe_values_are_defaults_a_request_may_leave(change, reported):
    request = Krea2Request(prompt="scene", turbo_4step=True, **change)
    assert reported in request.profile_deviations()


def test_a_request_on_the_distilled_recipe_reports_no_deviation():
    assert Krea2Request(prompt="scene", turbo_4step=True).profile_deviations() == ()


def test_profile_strength_needs_enablement():
    with pytest.raises(ValueError, match="require"):
        validate_record(
            Krea2Request(prompt="scene", turbo_4step_strength=0.75), location="[generate]"
        )


def test_offline_selection_stacking_and_duplicate_alias(tmp_path):
    repo = tmp_path / "hub" / ("models--" + TURBO_ADAPTER_REPO.replace("/", "--"))
    snapshot = repo / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    path = snapshot / TURBO_ADAPTER_FILENAME
    path.touch()
    with pytest.raises(FileNotFoundError, match="hf download"):
        resolve_turbo_adapters(tmp_path, enabled=True, strength=1, additional=())
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("a" * 40)
    style = AdapterSelection(tmp_path / "style.safetensors", 0.8)
    selections, receipt = resolve_turbo_adapters(
        tmp_path,
        enabled=True,
        strength=0.75,
        additional=(style,),
    )
    assert selections == (AdapterSelection(path, 0.75), style)
    assert receipt["revision"] == "a" * 40
    alias = tmp_path / "alias.safetensors"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="also listed"):
        resolve_turbo_adapters(
            tmp_path,
            enabled=True,
            strength=1,
            additional=(replace(style, path=alias),),
        )
    assert resolve_turbo_adapters(None, enabled=False, strength=1, additional=(style,)) == (
        (style,),
        None,
    )


def test_profile_raw_variant_uses_profile_defaults(profile_settings):
    invocation = assemble(
        build_parser().parse_args(
            [
                "--model",
                "krea2",
                "--variant",
                "raw",
                "--turbo-4step",
                "--prompt",
                "scene",
            ]
        ),
        base_settings=profile_settings,
        base_model_settings=Krea2Settings(),
    )
    assert invocation.request is not None
    assert (invocation.request.steps, invocation.request.guidance, invocation.request.mu) == (
        4,
        0,
        1.15,
    )


def test_exclusions_have_independent_cli_toml_and_env_controls(
    tmp_path, monkeypatch, profile_settings
):
    monkeypatch.setenv("KINO_KREA2_TURBO_4STEP", "true")
    monkeypatch.setenv("KINO_KREA2_TURBO_4STEP_EXCLUSIONS", "ff")
    config = tmp_path / "recipe.toml"
    config.write_text(
        'model = "krea2"\n[generate]\nprompt = "scene"\nturbo_4step_exclusions = "output"\n'
    )
    invocation = assemble(
        build_parser().parse_args(
            [
                "--model",
                "krea2",
                "--config",
                str(config),
                "--turbo-4step-exclude",
                "to_q,to_k",
                "--adapter",
                "style.safetensors:0.8",
            ]
        ),
        base_settings=profile_settings,
        base_model_settings=Krea2Settings(),
    )
    request = invocation.request
    assert request is not None
    assert request.turbo_4step_exclusions == "to_q,to_k"
    assert request.resolved_adapters()[0].strength == 0.8
    assert request.resolved_adapters()[0].exclude == ()
    with pytest.raises(ValueError, match="turbo_4step_exclusions.*not_a_category"):
        replace(request, turbo_4step_exclusions="not_a_category")


def test_raw_profile_warns_once_in_header(caplog, profile_settings):
    from kinomlx.models.krea2.command import _log_request

    invocation = assemble(
        build_parser().parse_args(["--variant", "raw", "--prompt", "scene", "--turbo-4step"]),
        base_settings=profile_settings,
    )
    _log_request(invocation)
    assert caplog.text.count("standard 8-step Turbo LoRA") == 1


@pytest.mark.parametrize("editing", [False, True])
def test_decode_logging_does_not_resolve_turbo_adapter(
    monkeypatch, caplog, profile_settings, editing
):
    from types import SimpleNamespace

    from kinomlx.models.krea2 import command

    invocation = assemble(
        build_parser().parse_args(["--model", "krea2", "--turbo-4step", "--prompt", "scene"]),
        base_settings=profile_settings,
        base_model_settings=Krea2Settings(),
    )
    restart = SimpleNamespace(
        config=SimpleNamespace(stage="decode"),
        source_run=Path("run.json"),
        selected_latents=Path("latents.safetensors"),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("decode-only restart resolved an unused adapter")

    monkeypatch.setattr(command, "resolve_turbo_adapters", forbidden)
    monkeypatch.setattr(command, "resolve_edit_adapters", forbidden)
    if editing:
        invocation = replace(
            invocation, request=replace(invocation.request, source_images=(Path("reference.png"),))
        )
    command._log_request(replace(invocation, restart=restart))


def test_profile_preflight_rejects_duplicate_and_missing_release(profile_settings):
    selections, _ = resolve_turbo_adapters(
        profile_settings.hf_home, enabled=True, strength=1, additional=()
    )
    args = ["--prompt", "scene", "--turbo-4step"]
    with pytest.raises(ConfigError, match="also listed"):
        assemble(
            build_parser().parse_args([*args, "--adapter", str(selections[0].path)]),
            base_settings=profile_settings,
        )
    selections[0].path.unlink()
    with pytest.raises(ConfigError, match="hf download"):
        assemble(build_parser().parse_args(args), base_settings=profile_settings)


def test_profile_can_be_disabled_from_cli(monkeypatch):
    monkeypatch.setenv("KINO_KREA2_TURBO_4STEP", "true")
    invocation = assemble(build_parser().parse_args(["--prompt", "scene", "--no-turbo-4step"]))
    assert not invocation.request.turbo_4step
    assert invocation.request.steps == 8


def test_four_step_profile_keeps_explicit_cli_sampler_values(profile_settings) -> None:
    invocation = assemble(
        build_parser().parse_args(
            [
                "--prompt",
                "scene",
                "--turbo-4step",
                "--steps",
                "6",
                "--guidance",
                "1.25",
                "--mu",
                "0.8",
            ]
        ),
        base_settings=profile_settings,
    )
    assert invocation.request is not None
    assert (invocation.request.steps, invocation.request.guidance, invocation.request.mu) == (
        6,
        1.25,
        0.8,
    )


@pytest.mark.parametrize("strength", [1.0, 0.75])
def test_denoise_restart_can_select_turbo_recipe(tmp_path, profile_settings, strength):
    from ..test_restart import _write_latents, _write_run, _write_text_conditioning

    latent, text, run = (
        tmp_path / name for name in ("latent.safetensors", "text.safetensors", "run.json")
    )
    _write_latents(latent)
    _write_text_conditioning(text)
    _write_run(run, latent, text_conditioning=text)
    invocation = assemble(
        build_parser().parse_args(
            [
                "--restart",
                str(run),
                "--restart-from",
                "denoise",
                "--turbo-4step",
                "--turbo-4step-strength",
                str(strength),
            ]
        ),
        base_settings=profile_settings,
    )
    assert invocation.request.turbo_4step_strength == strength
    assert (invocation.request.steps, invocation.request.guidance, invocation.request.mu) == (
        4,
        0,
        1.15,
    )


@pytest.mark.parametrize("change", ["strength", "off", "release"])
def test_existing_turbo_restart_changes_inputs_but_pins_release(tmp_path, profile_settings, change):
    import json

    from ..test_restart import _write_latents, _write_run, _write_text_conditioning

    latent, text, run = (
        tmp_path / name for name in ("latent.safetensors", "text.safetensors", "run.json")
    )
    _write_latents(latent)
    _write_text_conditioning(text)
    _write_run(run, latent, text_conditioning=text)
    payload = json.loads(run.read_text())
    payload["configuration"]["values"]["generate"].update(
        turbo_4step=True, steps=4, guidance=0, mu=1.15
    )
    _, profile = resolve_turbo_adapters(
        profile_settings.hf_home, enabled=True, strength=1.0, additional=()
    )
    payload["generation"]["turbo_4step"] = profile
    if change == "release":
        payload["generation"]["turbo_4step"]["revision"] = "b" * 40
    run.write_text(json.dumps(payload))
    args = ["--restart", str(run), "--restart-from", "denoise"]
    args += ["--no-turbo-4step"] if change == "off" else ["--turbo-4step-strength", "0.75"]
    if change == "release":
        with pytest.raises(ConfigError, match="release differs"):
            assemble(build_parser().parse_args(args), base_settings=profile_settings)
    else:
        invocation = assemble(build_parser().parse_args(args), base_settings=profile_settings)
        assert invocation.request.turbo_4step == (change != "off")
        assert invocation.request.steps == (8 if change == "off" else 4)
        if change == "strength":
            assert invocation.request.turbo_4step_strength == 0.75
