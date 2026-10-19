"""Origins follow actual decoding, implications, and surviving source precedence."""

import argparse

import pytest

from kinomlx.config.assembly import collect_invocation_layers, collect_invocation_sources
from kinomlx.config.provenance import PreparedConfig
from kinomlx.model_registry import config_registry


def test_prompt_decoder_retains_consumed_source_and_location(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("hello")
    options = argparse.Namespace(config=None, set_overrides=[f'generate.prompt_file="{prompt}"'])
    inputs = collect_invocation_layers(config_registry().model("flux2"), options)
    origin = inputs.provenance()[("generate", "prompt")]
    assert origin.explicit
    assert origin.kind == "normalized"
    assert origin.rule is not None
    assert origin.inputs[0].path == ("generate", "prompt_file")
    assert origin.inputs[0].location.kind == "override"
    assert origin.inputs[0].location.position == 1
    assert ("generate", "prompt_file") not in inputs.provenance()


def test_implication_has_trigger_origin_and_explicit_override_is_protected():
    options = argparse.Namespace(
        config=None,
        set_overrides=[
            "output.save_all_sidecars=true",
            "output.save_run_log=false",
        ],
    )
    inputs = collect_invocation_layers(config_registry().model("flux2"), options)
    origins = inputs.provenance()
    supplied = origins[("output", "save_run_log")]
    assert supplied.kind == "explicit"
    assert supplied.location.position == 2
    implied = origins[("output", "save_final_latents")]
    assert implied.kind == "implied"
    assert not implied.explicit
    assert implied.inputs[0].path == ("output", "save_all_sidecars")
    assert implied.inputs[0].location.position == 1


@pytest.mark.parametrize("model", ["gmnet", "flux2", "krea2", "ltx2"])
def test_precedence_keeps_winning_origins_and_group_siblings(tmp_path, monkeypatch, model):
    monkeypatch.setenv("KINO_VERBOSE", "true")
    config = tmp_path / "run.toml"
    config.write_text('[output]\nprefix="file"\n[settings]\nverbose=false\n')
    options = argparse.Namespace(
        config=config,
        set_overrides=[
            'output.prefix="discarded"',
            'output={prefix="winner"}',
        ],
        output_dir=tmp_path,
    )
    inputs = collect_invocation_sources(config_registry().model(model), options).prepare(
        PreparedConfig({"settings": {"verbose": True, "quiet": False}}, {})
    )
    origins = inputs.provenance()
    assert origins[("settings", "verbose")].location.identifier == str(config)
    assert origins[("settings", "quiet")].kind == "inherited"
    assert not origins[("settings", "quiet")].explicit
    assert origins[("output", "prefix")].location.position == 2
    assert origins[("output", "directory")].location.kind == "cli"
    assert ("settings", "verbose") in inputs.explicit_paths()
    with pytest.raises(TypeError):
        origins[("settings", "verbose")] = None


def test_deleted_prepared_field_does_not_steal_inherited_origin():
    options = argparse.Namespace(config=None, set_overrides=None, save_all_sidecars=True)
    inputs = collect_invocation_sources(config_registry().model("krea2"), options).prepare(
        PreparedConfig({"output": {"save_final_latents": False}}, {})
    )
    # Existing stage preparation may remove artifacts unavailable at that stage.
    inputs.cli.values["output"].pop("save_final_latents")
    origin = inputs.provenance()[("output", "save_final_latents")]
    assert origin.kind == "inherited"


@pytest.mark.parametrize("model", ["gmnet", "flux2", "krea2", "ltx2"])
def test_typed_invocation_retains_sparse_intent_separate_from_execution(model, monkeypatch):
    import importlib
    from dataclasses import replace

    monkeypatch.delenv("KINO_VERBOSE", raising=False)
    parser = importlib.import_module(f"kinomlx.models.{model}.arguments").build_parser()
    assemble = importlib.import_module(f"kinomlx.models.{model}.config").assemble
    invocation = assemble(
        parser.parse_args(
            [
                "--print-config",
                "--set",
                "settings.verbose=false",
                "--save-all-sidecars",
            ]
        )
    )
    intent = invocation.config_intent
    assert intent is not None
    assert intent.values["settings"]["verbose"] is False
    assert intent.origins[("settings", "verbose")].location.position == 1
    assert ("settings", "verbose") in invocation.explicit_config_paths
    assert "quiet" not in intent.values["settings"]
    assert invocation.resolved_config["settings"]["quiet"] is False
    assert intent.origins[("output", "save_run_log")].kind == "implied"
    assert replace(invocation, resolved_config={}).config_intent is intent


def test_internal_clear_marker_survives_in_intent_after_typed_normalization():
    from kinomlx.models.flux2.arguments import build_parser
    from kinomlx.models.flux2.config import assemble

    invocation = assemble(
        build_parser().parse_args(
            [
                "--print-config",
                "--set",
                "generate.seed=7",
            ]
        )
    )
    assert invocation.config_intent.values["generate"]["seeds"] is None
    cleared = invocation.config_intent.origins[("generate", "seeds")]
    assert not cleared.explicit
    assert cleared.rule is not None
    assert cleared.inputs[0].location.position == 1
    assert invocation.generate.seed == 7


@pytest.mark.parametrize("model", ["gmnet", "flux2", "krea2", "ltx2"])
def test_environment_default_and_fallback_have_distinct_origins(model, monkeypatch, tmp_path):
    schema = config_registry().model(model)
    monkeypatch.delenv("KINO_APP_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    options = argparse.Namespace(config=None, set_overrides=None)
    path = ("settings", "app_cache_dir")
    default = collect_invocation_layers(schema, options).state().origins[path]
    assert default.kind == "default"
    assert not default.explicit
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    fallback = collect_invocation_layers(schema, options).state().origins[path]
    assert fallback.kind == "explicit"
    assert fallback.location.identifier == "{{XDG_CACHE_HOME}}/KinoMLX"
    monkeypatch.setenv("KINO_APP_CACHE_DIR", str(tmp_path / "direct"))
    direct = collect_invocation_layers(schema, options).state().origins[path]
    assert direct.location.identifier == "{{KINO_APP_CACHE_DIR}}"


@pytest.mark.parametrize("model", ["gmnet", "flux2", "krea2", "ltx2"])
def test_model_environment_alias_origin_uses_actual_winner(model, monkeypatch, tmp_path):
    schema = config_registry().model(model)
    specific = f"KINO_{model.upper()}_OUTPUT_DIR"
    monkeypatch.setenv(specific, "")
    monkeypatch.setenv("KINO_OUTPUT_DIR", str(tmp_path))
    options = argparse.Namespace(config=None, set_overrides=None)
    origin = collect_invocation_layers(schema, options).state().origins[("output", "directory")]
    assert origin.location.identifier == "KINO_OUTPUT_DIR"
    monkeypatch.setenv(specific, str(tmp_path / "specific"))
    origin = collect_invocation_layers(schema, options).state().origins[("output", "directory")]
    assert origin.location.identifier == specific


def test_default_cache_path_does_not_become_explicit_sparse_output(monkeypatch, capsys):
    import tomllib

    from kinomlx.cli.main import main

    monkeypatch.delenv("KINO_APP_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert main(["--model", "gmnet", "--print-config", "--only-non-defaults"]) == 0
    assert "app_cache_dir" not in tomllib.loads(capsys.readouterr().out).get("settings", {})


def test_decoded_source_state_can_be_normalized_without_reopening_files(tmp_path):
    schema = config_registry().model("flux2")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("decoded once")
    raw = {
        "generate": {"prompt_file": str(prompt)},
        "output": {"save_all_sidecars": True},
    }
    decoded = schema.prepare_sources_with_provenance((("source", raw),))[0]
    assert decoded.values["generate"]["prompt"] == "decoded once"
    assert "prompt_file" not in decoded.values["generate"]
    assert "save_final_latents" not in decoded.values["output"]
    prompt.unlink()
    first = schema.normalize_prepared_layer(decoded)
    second = schema.normalize_prepared_layer(decoded)
    assert first == second
    assert first.values["output"]["save_final_latents"] is True
    assert first.origins[("output", "save_final_latents")].kind == "implied"
    assert not first.origins[("output", "save_final_latents")].explicit
    assert first.origins[("generate", "prompt")] == decoded.origins[("generate", "prompt")]
    assert "save_final_latents" not in decoded.values["output"]


def test_split_source_phase_admits_all_layers_before_any_read(tmp_path):
    schema = config_registry().model("flux2")
    # If the first layer were read early, this would fail with a missing-file error.
    layers = (
        ("first", {"generate": {"prompt_file": str(tmp_path / "absent.txt")}}),
        ("second", {"generate": {"width": True}}),
    )
    with pytest.raises(ValueError, match="width"):
        schema.prepare_sources_with_provenance(layers)


@pytest.mark.parametrize("model", ["gmnet", "flux2", "krea2", "ltx2"])
def test_invocations_require_intent_metadata(model):
    import dataclasses
    import importlib

    module = importlib.import_module(f"kinomlx.models.{model}.config")
    parser = importlib.import_module(f"kinomlx.models.{model}.arguments").build_parser()
    options = parser.parse_args(["--prompt", "a tree"] if model != "gmnet" else [])
    invocation = module.assemble(options)
    fields = {
        field.name: getattr(invocation, field.name)
        for field in dataclasses.fields(invocation)
        if field.init and field.name != "config_intent"
    }
    with pytest.raises(TypeError, match="config_intent"):
        type(invocation)(**fields)


def test_config_output_requires_retained_intent():
    from kinomlx.cli.config_output import handle_config_output

    with pytest.raises(TypeError, match="intent"):
        handle_config_output(argparse.Namespace(), model="flux2", json_output=False)
