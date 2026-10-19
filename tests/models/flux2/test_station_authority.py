"""Klein's complete saved-stage and direct API ownership declarations."""

from dataclasses import replace
from pathlib import Path

import pytest

from kinomlx.cli.resource_operation import Operation
from kinomlx.config import ConfigError
from kinomlx.config.assembly import ConfigLayer, InvocationLayers
from kinomlx.config.overrides import SetOverrides
from kinomlx.config.provenance import FieldOrigin, PreparedConfig, SourceLocation
from kinomlx.io.output_config import ImageOutputConfig
from kinomlx.models.flux2 import command
from kinomlx.models.flux2.arguments import build_parser
from kinomlx.models.flux2.config import assemble
from kinomlx.models.flux2.config_schema import CONFIG_SCHEMA
from kinomlx.models.flux2.pipelines import generate_flux2, restart_flux2
from kinomlx.models.flux2.resources import prepare_resources
from kinomlx.models.flux2.settings import Flux2Settings
from kinomlx.models.flux2.types import Flux2Request, Flux2RestartConfig
from kinomlx.settings import Settings
from tests.models.flux2.test_cli import _fake_generation


def _roots() -> dict[str, object]:
    return {
        "generate": Flux2Request(),
        "output": ImageOutputConfig(),
        "restart": Flux2RestartConfig(Path("recorded_run.json")),
        "settings": Settings(),
        "model_settings": Flux2Settings(),
        "operation": Operation(),
    }


def test_every_klein_table_and_field_declares_both_restart_stages():
    schema = CONFIG_SCHEMA
    assert set(schema.restart_stages or ()) == {"denoise", "decode"}
    assert all(set(table.stages) == {"denoise", "decode"} for table in schema.tables)
    assert all(
        {policy.stage for policy in field.stage_policies} == {"denoise", "decode"}
        and all(policy.reason for policy in field.stage_policies)
        for table in schema.tables
        for field in table.fields
        if not field.virtual
    )
    assert schema.environment_variables()[("restart", "run")][0] == ("KINO_FLUX2_RESTART_RUN")
    assert schema.environment_variables()[("restart", "index")][0] == ("KINO_FLUX2_RESTART_INDEX")


def test_klein_stage_admission_distinguishes_mutable_locked_and_unused_controls():
    schema = CONFIG_SCHEMA
    parent = _roots()
    current = {**parent, "generate": replace(parent["generate"], steps=6)}
    assert schema.stage_changes(current, parent, stage="denoise", active_explicit=frozenset()) == ()
    assert schema.stage_changes(current, parent, stage="decode", active_explicit=frozenset()) == (
        ("generate", "steps"),
    )

    resource = ("model_settings", "source_path")
    replacement = {**parent, "model_settings": Flux2Settings(source_path=Path("elsewhere"))}
    for stage in ("denoise", "decode"):
        policy = next(item for item in schema.field(resource).stage_policies if item.stage == stage)
        assert (policy.mode, policy.irrelevant_action) == ("irrelevant", "reject")
        assert schema.stage_changes(
            replacement,
            parent,
            stage=stage,
            active_explicit=frozenset({resource}),
        ) == (resource,)
        assert (
            schema.stage_changes(
                _roots(), parent, stage=stage, active_explicit=frozenset({resource})
            )
            == ()
        )
        assert (
            schema.stage_changes(replacement, parent, stage=stage, active_explicit=frozenset())
            == ()
        )
    output_policy = next(
        item
        for item in schema.field(("output", "save_text_conditioning")).stage_policies
        if item.stage == "decode"
    )
    assert (output_policy.mode, output_policy.irrelevant_action) == (
        "irrelevant",
        "ignore",
    )


def test_saved_resource_selector_masks_lower_priority_environment_input():
    path = ("model_settings", "source_path")
    environment = ConfigLayer(
        "environment",
        {"model_settings": {"source_path": Path("environment")}},
        {"model_settings": {"source_path": Path("environment")}},
        {path: FieldOrigin(path, "explicit", True, SourceLocation("environment", "source"))},
    )
    saved = PreparedConfig(
        {"model_settings": {"source_path": Path("recorded")}},
        {path: FieldOrigin(path, "inherited", True, SourceLocation("saved", "run"))},
    )
    empty = ConfigLayer("empty", {}, {}, {})
    inputs = InvocationLayers(
        environment,
        empty,
        empty,
        empty,
        SetOverrides.parse([]),
        CONFIG_SCHEMA,
        saved,
    )
    assert inputs.state().values["model_settings"] == {"source_path": Path("recorded")}
    assert path not in inputs.active_explicit_paths()
    parent = _roots()
    parent["model_settings"] = Flux2Settings(source_path=Path("recorded"))
    assert (
        CONFIG_SCHEMA.stage_changes(
            parent,
            parent,
            stage="denoise",
            active_explicit=inputs.active_explicit_paths(),
        )
        == ()
    )


def test_klein_direct_public_operations_have_declared_owners():
    operations = {boundary.operation for boundary in CONFIG_SCHEMA.api_boundaries}
    assert {prepare_resources, generate_flux2, restart_flux2} <= operations
    for boundary in CONFIG_SCHEMA.api_boundaries:
        boundary.validate_tracked_projections()


def test_replay_accepts_a_recorded_resource_value_and_refuses_a_different_one(
    tmp_path, monkeypatch
):
    _fake_generation(monkeypatch)
    source = tmp_path / "recorded-source"
    assert (
        command.run_flux2_command(
            [
                "--prompt",
                "a painted bowl",
                "--width",
                "64",
                "--height",
                "64",
                "--source-path",
                str(source),
                "--transformer-dtype",
                "bfloat16",
                "--save-all-sidecars",
                "--output",
                str(tmp_path / "first.png"),
                "--quiet",
            ]
        )
        == 0
    )
    restart = ["--restart", str(tmp_path / "first_run.json"), "--restart-from", "decode"]
    monkeypatch.setenv("KINO_FLUX2_SOURCE_PATH", str(tmp_path / "masked-environment"))
    replay = assemble(build_parser().parse_args(restart))
    assert replay.model_settings.source_path == source

    retyped = assemble(build_parser().parse_args([*restart, "--transformer-dtype", "bfloat16"]))
    assert retyped.model_settings.transformer_dtype == "bfloat16"

    shared = tmp_path / "shared.toml"
    shared.write_text(
        f'[model_settings]\nsource_path = "{source}"\n',
        encoding="utf-8",
    )
    from_file = assemble(build_parser().parse_args([*restart, "--config", str(shared)]))
    assert from_file.model_settings.source_path == source

    with pytest.raises(ConfigError) as refusal:
        assemble(build_parser().parse_args([*restart, "--transformer-dtype", "float16"]))
    assert str(refusal.value) == (
        "[restart]: decode restart: [model_settings].transformer_dtype is not used on "
        "replay: Replay reopens the recorded component dtype; use fresh generation to "
        "change it."
    )
