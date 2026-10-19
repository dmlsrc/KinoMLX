"""Saved-source selection projects declared fields without evaluating record edits."""

import argparse
import contextlib
import io
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from kinomlx.cli.main import main
from kinomlx.config import ConfigError, ConfigGroup, SetOverrides
from kinomlx.config.assembly import collect_invocation_layers, collect_invocation_sources
from kinomlx.config.provenance import PreparedConfig
from kinomlx.config.transforms import LayerTransform
from kinomlx.model_registry import config_registry


@pytest.mark.parametrize("model", ["flux2", "krea2", "ltx2"])
def test_restart_selector_coverage_is_owned_by_the_registered_fields(model):
    schema = config_registry().model(model)
    table = schema.table(("restart",))
    assert {(*table.path, field.name) for field in table.fields} <= set(
        schema.paths_in_group(ConfigGroup.SOURCE_SELECTOR)
    )


def test_selector_projection_ignores_record_execution_but_preserves_parent_assignment_order():
    overrides = SetOverrides.parse(
        [
            "generate.adapters[99].strength=0.5",
            'restart={run="first.json",stage="decode"}',
            "generate.adapters=[]",
            'restart.run="second.json"',
            'restart={stage="denoise"}',
        ]
    )
    selected = overrides.project(frozenset({("restart", "run"), ("restart", "stage")}))
    assert selected.config == {"restart": {"run": "second.json", "stage": "denoise"}}
    assert selected.origins[("restart", "run")].ordinal == 4
    assert selected.origins[("restart", "stage")].ordinal == 5
    assert overrides.assignments[0].path == ("generate", "adapters", 99, "strength")


@pytest.mark.parametrize("model", ["flux2", "krea2", "ltx2"])
def test_source_projection_agrees_with_complete_precedence(model, tmp_path):
    schema = config_registry().model(model)
    config = tmp_path / "run.toml"
    config.write_text('[restart]\nrun="file.json"\nstage="decode"\n')
    options = argparse.Namespace(
        config=config,
        restart_run=Path("cli.json"),
        set_overrides=['restart={run="set.json"}', "settings.verbose=true"],
    )
    inputs = collect_invocation_layers(schema, options)
    selected = inputs.selection()
    assert selected == {"restart": {"run": "set.json", "stage": "decode"}}
    assert selected["restart"] == inputs.state().values["restart"]
    assert "settings" not in selected


def test_empty_parent_overlay_does_not_clear_an_earlier_selector():
    selected = SetOverrides.parse(['restart.run="source.json"', "restart={}"]).project(
        frozenset({("restart", "run")})
    )
    assert selected.config == {"restart": {"run": "source.json"}}


@pytest.mark.parametrize("model", ["flux2", "krea2", "ltx2"])
def test_admitted_sources_select_restart_before_prompt_file_effects(model, tmp_path):
    schema = config_registry().model(model)
    options = argparse.Namespace(
        config=None,
        set_overrides=[
            'restart.run="selected.json"',
            f'generate.prompt_file="{tmp_path / "not-opened.txt"}"',
        ],
    )
    sources = collect_invocation_sources(schema, options)
    assert sources.selection() == {"restart": {"run": "selected.json"}}
    with pytest.raises(ValueError, match="not-opened.txt"):
        sources.prepare()


def test_final_source_projection_cannot_silently_change_the_selected_stage():
    schema = config_registry().model("flux2")
    path = ("restart", "stage")
    rule = LayerTransform(
        "fixture_rewrite_stage",
        frozenset({path}),
        frozenset({path}),
        lambda values: {path: "denoise"} if path in values else {},
    )
    tables = tuple(
        replace(
            table,
            fields=tuple(
                replace(field, layer_transforms=(*field.layer_transforms, rule))
                if table.path == ("restart",) and field.name == "stage"
                else field
                for field in table.fields
            ),
        )
        if table.path == ("restart",)
        else table
        for table in schema.tables
    )
    changed = replace(schema, tables=tables)
    sources = collect_invocation_sources(
        changed,
        argparse.Namespace(
            config=None,
            set_overrides=[
                'restart={run="selected.json",stage="decode"}',
            ],
        ),
    )
    assert sources.selection()["restart"]["stage"] == "decode"
    with pytest.raises(ConfigError, match="source selectors changed"):
        sources.prepare()


@pytest.mark.parametrize("override", [False, True])
def test_saved_component_selector_masks_environment_before_final_projection(monkeypatch, override):
    schema = config_registry().model("krea2")
    monkeypatch.setenv("KINO_KREA2_VAE_PATH", "environment/vae")
    saved = PreparedConfig({"model_settings": {"vae_path": "saved/vae"}}, {})
    options = argparse.Namespace(
        config=None,
        set_overrides=['model_settings.vae_path="current/vae"'] if override else [],
    )
    sources = collect_invocation_sources(schema, options)
    assert sources.selection() == {
        "model_settings": {"vae_path": "current/vae" if override else Path("environment/vae")}
    }
    layers = sources.prepare(saved, stage="decode")
    expected = "current/vae" if override else "saved/vae"
    assert layers.merge()["model_settings"]["vae_path"] == expected
    assert layers.selection() == {"model_settings": {"vae_path": expected}}
    assert (("model_settings", "vae_path") in layers.active_explicit_paths()) is override
    assert saved.values == {"model_settings": {"vae_path": "saved/vae"}}


@pytest.mark.parametrize(
    ("model", "variable", "value"),
    [
        ("krea2", "KINO_KREA2_RESTART_STAGE", "decode"),
        ("ltx2", "KINO_LTX2_LATENT_STAGE", "stage-2"),
        ("flux2", "KINO_FLUX2_RESTART_INDEX", "1"),
    ],
)
def test_an_environment_only_restart_table_without_run_is_not_a_restart(
    monkeypatch, model, variable, value, caplog, tmp_path
):
    """An exported selector must not put every later ordinary run into restart mode."""
    import logging

    for name in config_registry().environment_variables():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variable, value)
    arguments = ["--model", model, "--print-config", "--prompt", "a still"]
    out, err = io.StringIO(), io.StringIO()
    with (
        caplog.at_level(logging.DEBUG, logger="kinomlx.config.assembly"),
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        assert main(arguments) == 0, err.getvalue()
    # An ignored selector is not intent: the printed configuration describes the
    # ordinary run that happens, and says so by carrying no restart table.
    assert "restart" not in tomllib.loads(out.getvalue())
    assert variable in caplog.text

    saved = tmp_path / f"{model}.toml"
    err = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
        code = main(["--model", model, "--prompt", "a still", "--save-config", str(saved)])
    assert code == 0, err.getvalue()
    # Reloading the saved file must not fail on a restart key a file layer wrote:
    # a file [restart] table without a run is refused, so the table must be gone.
    reloaded = io.StringIO()
    with contextlib.redirect_stdout(reloaded), contextlib.redirect_stderr(err):
        code = main(["--model", model, "--print-config", "--config", str(saved)])
    assert code == 0, err.getvalue()
    assert "restart" not in tomllib.loads(reloaded.getvalue())


def test_an_environment_selector_beside_a_real_restart_run_is_kept(monkeypatch, tmp_path):
    from tests.cli.test_restart import _source_run

    for name in config_registry().environment_variables():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KINO_LTX2_LATENT_STAGE", "stage-1")
    run, _paths = _source_run(tmp_path)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(["--model", "ltx2", "--print-config", "--restart", str(run)])
    assert code == 0, err.getvalue()
    # Only an ignored table is dropped; a selector that chose this restart is
    # intent and stays in the printed configuration.
    assert tomllib.loads(out.getvalue())["restart"]["latent_stage"] == "stage-1"


@pytest.mark.parametrize(
    ("model", "arguments", "expected"),
    [
        (
            "krea2",
            ["--set", "restart.stage=decode"],
            "[restart].run is required; [restart].stage was set by --set",
        ),
        (
            "ltx2",
            ["--restart-from", "decode"],
            "[restart].run is required; [restart].stage was set by command line",
        ),
        (
            "flux2",
            ["--set", "restart.index=1"],
            "[restart].run is required; [restart].index was set by --set",
        ),
    ],
)
def test_a_restart_key_above_the_environment_still_requires_run(
    monkeypatch, model, arguments, expected
):
    for name in config_registry().environment_variables():
        monkeypatch.delenv(name, raising=False)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        assert main(["--model", model, "--print-config", "--prompt", "a still", *arguments]) != 0
    assert expected in " ".join(err.getvalue().split())


def test_a_file_restart_key_names_the_environment_variable_beside_it(monkeypatch, tmp_path, caplog):
    for name in config_registry().environment_variables():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KINO_KREA2_RESTART_STAGE", "decode")
    config = tmp_path / "restart.toml"
    config.write_text('[restart]\nlatents = "latents.safetensors"\n', encoding="utf-8")
    import logging

    out, err = io.StringIO(), io.StringIO()
    with (
        caplog.at_level(logging.ERROR, logger="kinomlx"),
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        code = main(
            ["--model", "krea2", "--print-config", "--prompt", "a still", "--config", str(config)]
        )
    assert code != 0
    rendered = caplog.text
    assert f"{config}: [restart].run is required" in rendered
    assert f"[restart].latents was set by {config}" in rendered
    assert "[restart].stage was set by environment KINO_KREA2_RESTART_STAGE" in rendered
