"""Canonical saved input survives reload without execution observations or old-format recovery."""

import argparse
import copy
import json
from pathlib import Path

import pytest

from kinomlx.config import ConfigError
from kinomlx.config.assembly import collect_invocation_sources
from kinomlx.config.persistence import (
    RUN_SCHEMA_VERSION,
    configuration_payload,
    read_configuration,
)
from kinomlx.config.provenance import PreparedConfig
from kinomlx.debug.metadata import RunRecord
from kinomlx.model_registry import config_registry
from kinomlx.reporting import TimingReporter


def sources(*assignments, **cli):
    return collect_invocation_sources(
        config_registry().model("flux2"),
        argparse.Namespace(config=None, set_overrides=list(assignments), **cli),
    )


def saved_payload(intent):
    return json.loads(
        json.dumps(
            {
                "schema_version": RUN_SCHEMA_VERSION,
                "configuration": configuration_payload("flux2", intent),
                "effective_config": {"generate": {"steps": 999, "guidance": 999}},
            },
            default=str,
        )
    )


def test_saved_explicit_defaults_and_record_presence_survive_indexed_replay(monkeypatch):
    monkeypatch.delenv("KINO_VERBOSE", raising=False)
    original = (
        sources(
            "settings.verbose=false",
            'generate.adapters=[{path="first"},{path="second",strength=1.0}]',
        )
        .prepare()
        .state()
    )
    payload = saved_payload(original)
    restored = read_configuration(payload, model="flux2", source="saved.json")
    assert restored.explicit == original.explicit
    assert restored.replacements == original.replacements
    monkeypatch.setenv("KINO_VERBOSE", "true")
    result = sources("generate.adapters[0].exclude=[]").prepare(restored).state()
    assert result.values["settings"]["verbose"] is False
    assert ("settings", "verbose") in result.explicit
    assert ("generate", "adapters", 1, "strength") in result.explicit
    assert ("generate", "adapters", 0, "strength") not in result.explicit
    assert result.values["generate"]["adapters"][1] == {"path": "second", "strength": 1.0}
    assert "steps" not in result.values["generate"]
    assert "guidance" not in result.values["generate"]
    second = read_configuration(saved_payload(result), model="flux2", source="second.json")
    assert second.explicit == result.explicit
    assert second.replacements == result.replacements


def test_run_record_requires_intent_and_labels_effective_values_separately(tmp_path):
    path = tmp_path / "run.json"
    intent = sources("generate.seed=0").prepare().state()
    RunRecord(
        path,
        model="flux2",
        config_intent=intent,
        effective_config={"generate": {"seed": 0, "steps": 4}},
        argv=[],
        timings=TimingReporter(),
        planned_outputs={},
    )
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 2
    assert "invocation" not in payload
    assert payload["configuration"]["values"]["generate"] == {"seed": 0, "seeds": None}
    assert payload["effective_config"]["generate"]["steps"] == 4
    restored = read_configuration(payload, model="flux2", source=str(path))
    assert restored.explicit == intent.explicit
    assert ("generate", "seed") in restored.explicit


@pytest.mark.parametrize("version", [None, 1, 3, True, "2"])
def test_missing_or_unsupported_run_format_has_no_fallback(version):
    payload = saved_payload(PreparedConfig({}, {}))
    payload["schema_version"] = version
    payload["invocation"] = {"generate": {"seed": 0}}
    with pytest.raises(ConfigError, match="unsupported run schema"):
        read_configuration(payload, model="flux2", source="saved")


@pytest.mark.parametrize("field", ["format", "model", "values", "explicit", "replacements"])
def test_every_saved_configuration_member_is_required(field):
    payload = saved_payload(PreparedConfig({}, {}))
    del payload["configuration"][field]
    with pytest.raises(ConfigError, match="expected fields"):
        read_configuration(payload, model="flux2", source="saved")


@pytest.mark.parametrize("path", [[], [True], ["generate", "seed", -1], ["generate", "missing"]])
def test_saved_presence_cannot_name_invalid_or_absent_values(path):
    payload = saved_payload(sources("generate.seed=0").prepare().state())
    payload["configuration"]["explicit"] = [path]
    with pytest.raises(ConfigError, match="invalid field path|absent from values"):
        read_configuration(payload, model="flux2", source="saved")


def test_saved_unknown_fields_fail_before_current_source_file_io(tmp_path):
    prepared = sources(prompt_file=Path(tmp_path / "missing"))
    with pytest.raises(ConfigError, match="unknown field 'bogus'"):
        prepared.prepare(PreparedConfig({"generate": {"bogus": 1}}, {}))


def test_saved_collection_markers_do_not_survive_whole_collection_replacement():
    original = sources('generate.adapters=[{path="first"},{path="second"}]').prepare().state()
    result = sources("generate.adapters=[]").prepare(original).state()
    assert result.values["generate"]["adapters"] == []
    assert not any(
        len(path) > 2 and path[:2] == ("generate", "adapters") for path in result.origins
    )
    assert result.replacements == frozenset({("generate", "adapters")})
    snapshot = copy.deepcopy(saved_payload(result))
    restored = read_configuration(snapshot, model="flux2", source="saved")
    assert restored.replacements == result.replacements


@pytest.mark.parametrize("model", ["flux2", "krea2", "ltx2"])
@pytest.mark.parametrize("old_version", [True, False])
def test_public_restart_readers_require_current_configuration_before_artifact_io(
    model, old_version, tmp_path
):
    import importlib

    source = tmp_path / "run.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1 if old_version else 2,
                "model": model,
                "status": "completed",
                "invocation": {"generate": {"prompt": "old snapshot"}},
                "effective_config": {"generate": {"prompt": "observation"}},
                "outputs": {"final_latents": "does-not-exist.safetensors"},
            }
        )
    )
    module = importlib.import_module(f"kinomlx.models.{model}.restart")
    reader = module.RestartManifest.load if model == "flux2" else module.load_restart_manifest
    selector = module.Flux2RestartConfig if model == "flux2" else module.RestartConfig
    with pytest.raises(ValueError, match="unsupported run schema|configuration: expected"):
        reader(selector(source))


@pytest.mark.parametrize("field", ["format", "model"])
def test_saved_configuration_rejects_an_unsupported_identity(field):
    payload = saved_payload(PreparedConfig({}, {}))
    payload["configuration"][field] = "unsupported"
    with pytest.raises(ConfigError, match="unsupported format or model"):
        read_configuration(payload, model="flux2", source="saved")


def test_saved_presence_cannot_address_a_primitive_sequence_element():
    payload = saved_payload(sources("generate.seeds=[0,1]").prepare().state())
    payload["configuration"]["explicit"] = [["generate", "seeds", 0]]
    decoded = read_configuration(payload, model="flux2", source="saved")
    with pytest.raises(ConfigError, match="presence does not match its declared fields"):
        sources().prepare(decoded)


def test_api_settings_base_is_saved_with_its_env_mask_and_actual_presence(monkeypatch):
    from kinomlx.settings import Settings

    monkeypatch.setenv("KINO_VERBOSE", "true")
    monkeypatch.setenv("KINO_QUIET", "true")
    schema = config_registry().model("flux2")
    original = (
        collect_invocation_sources(
            schema,
            argparse.Namespace(config=None, set_overrides=[]),
            base_settings=Settings(verbose=False),
        )
        .prepare()
        .state()
    )
    assert original.values["settings"]["verbose"] is False
    assert original.values["settings"]["quiet"] is False
    assert ("settings", "verbose") in original.explicit
    assert ("settings", "quiet") not in original.explicit
    assert original.origins[("settings", "verbose")].location.kind == "api"
    restored = read_configuration(saved_payload(original), model="flux2", source="saved")
    replay = sources().prepare(restored).state()
    assert replay.values["settings"]["quiet"] is False
    assert replay.values["settings"]["verbose"] is False
    assert replay.explicit == original.explicit


@pytest.mark.parametrize("model", ["gmnet", "krea2", "ltx2"])
def test_public_api_model_settings_base_enters_canonical_intent(model):
    import importlib

    module = importlib.import_module(f"kinomlx.models.{model}.config")
    settings_module = importlib.import_module(f"kinomlx.models.{model}.settings")
    parser = importlib.import_module(f"kinomlx.models.{model}.arguments").build_parser()
    settings_type = {
        "gmnet": "GMNetSettings",
        "krea2": "Krea2Settings",
        "ltx2": "LTX2Settings",
    }[model]
    field, value = (
        ("transformer_dtype", "float16")
        if model != "gmnet"
        else ("weights_path", Path("model.safetensors"))
    )
    settings = getattr(settings_module, settings_type)(**{field: value})
    invocation = module.assemble(
        parser.parse_args(["--print-config"]), base_model_settings=settings
    )
    assert invocation.config_intent.values["model_settings"][field] == value
    assert ("model_settings", field) in invocation.config_intent.explicit
    assert invocation.config_intent.origins[("model_settings", field)].location.kind == "api"


def test_cli_overrides_api_settings_through_the_same_source_precedence():
    from kinomlx.settings import Settings

    state = (
        collect_invocation_sources(
            config_registry().model("flux2"),
            argparse.Namespace(config=None, set_overrides=["settings.verbose=false"]),
            base_settings=Settings(verbose=True),
        )
        .prepare()
        .state()
    )
    assert state.values["settings"]["verbose"] is False
    origin = state.origins[("settings", "verbose")]
    assert origin.explicit
    assert origin.location.kind == "override"


@pytest.mark.parametrize(
    ("model", "values"),
    [
        ("krea2", {"generate": {"prompt_file": "missing.txt"}}),
        ("flux2", {"generate": {"prompt_file": "missing.txt"}}),
        ("ltx2", {"generate": {"prompt_file": "missing.txt"}}),
        ("ltx2", {"generate": {"auto_duration": True}}),
    ],
)
def test_saved_boundaries_reject_unresolved_source_inputs(model, values):
    with pytest.raises(ConfigError, match="source-only input must be resolved"):
        configuration_payload(model, PreparedConfig(values, {}))
    configuration = configuration_payload(model, PreparedConfig({}, {}))
    configuration["values"] = values
    payload = {"schema_version": RUN_SCHEMA_VERSION, "configuration": configuration}
    with pytest.raises(ConfigError, match="source-only input must be resolved"):
        read_configuration(payload, model=model, source="saved")


@pytest.mark.parametrize(
    ("values", "error"),
    [
        ({"generate": {"unknown_input": 1}}, "unknown field"),
        ({"generate": {"steps": "invalid"}}, "expected int"),
        ({"generate": {"adapters": [{"path": "adapter", "strength": None}]}}, "expected float"),
    ],
)
def test_saved_boundaries_validate_registered_shape(values, error):
    with pytest.raises(ConfigError, match=error):
        configuration_payload("flux2", PreparedConfig(values, {}))
    payload = saved_payload(PreparedConfig({}, {}))
    payload["configuration"]["values"] = values
    with pytest.raises(ConfigError, match=error):
        read_configuration(payload, model="flux2", source="saved")


def test_every_model_field_declares_persistence():
    from kinomlx.model_registry import model_choices

    for model in model_choices():
        for _, declaration in config_registry().model(model).field_items():
            assert declaration.persistence in {"intent", "source"}
