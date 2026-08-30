"""Model-specific starter configuration command contracts."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from kinomlx.cli._registry import (
    config_registry,
    model_choices,
    validate_model_parser,
)
from kinomlx.cli.args import build_parser
from kinomlx.cli.config_init import build_config_parser
from kinomlx.cli.config_templates import (
    _render_field,
    _render_table,
    config_template,
    template_models,
)
from kinomlx.cli.main import main
from kinomlx.config import (
    ConfigFieldSpec,
    ConfigGroup,
    ConfigImplication,
    ConfigTableSpec,
    ModelConfigSpec,
)


def _documented_fields(template: str) -> set[str]:
    table = ""
    documented: set[str] = set()
    multiline = False
    for line in template.splitlines():
        if multiline:
            if line in {"'''", "# '''"}:
                multiline = False
            continue
        content = line[2:] if line.startswith("# ") else line
        if content.startswith("[") and content.endswith("]"):
            table = content[1:-1]
            continue
        if " = " not in content:
            continue

        name, value = content.split(" = ", 1)
        if not name.replace("_", "").isalnum():
            continue
        documented.add(f"{table}.{name}" if table else name)
        multiline = value in {"'''", '"""'}

    return documented


def test_every_registered_model_has_one_distinct_template() -> None:
    assert template_models() == model_choices()
    assert config_template("ltx2") != config_template("gmnet")
    registry = config_registry()
    assert registry.model("ltx2").table(("settings",)) is registry.model("gmnet").table(
        ("settings",)
    )
    with pytest.raises(RuntimeError, match="registry is frozen"):
        registry.register(registry.model("ltx2"))


def test_ltx2_template_documents_the_complete_schema() -> None:
    schema = config_registry().model("ltx2")
    expected = {field.name for field in schema.root_fields}
    expected.update(
        ".".join((*table.path, field.name)) for table in schema.tables for field in table.fields
    )
    assert _documented_fields(config_template("ltx2")) == expected


def test_gmnet_template_documents_only_the_gmnet_schema() -> None:
    schema = config_registry().model("gmnet")
    expected = {field.name for field in schema.root_fields}
    expected.update(
        ".".join((*table.path, field.name)) for table in schema.tables for field in table.fields
    )
    documented = _documented_fields(config_template("gmnet"))
    assert documented == expected
    assert not any(name.startswith("generate.") for name in documented)


def test_registry_derives_every_environment_surface_from_record_metadata() -> None:
    registry = config_registry()
    assert "KINO_OUTPUT_DIR" in registry.environment_variables()
    assert "KINO_GMNET_VARIANT" in registry.environment_variables()
    assert "KINO_TRANSFORMER_DTYPE" in registry.environment_variables()
    for model in registry.models():
        template = config_template(model)
        for variables in registry.model(model).environment_variables().values():
            assert f"# Environment: {', '.join(variables)}." in template


def test_table_registration_rejects_an_undocumented_dataclass_field() -> None:
    @dataclass(frozen=True)
    class _Record:
        first: int = 1
        second: int = 2

    with pytest.raises(ValueError, match="schema drift: missing second"):
        ConfigTableSpec(
            path=("sample",),
            record=_Record,
            fields=(ConfigFieldSpec("first", "Set first", "an integer", 1, "first"),),
        )


def test_required_record_fields_are_limited_to_default_absent_tables() -> None:
    @dataclass(frozen=True)
    class _RequiredRecord:
        value: int

    field = ConfigFieldSpec("value", "Set value", "an integer", 1, "value")
    with pytest.raises(ValueError, match="default-present.*required field"):
        ConfigTableSpec(
            path=("sample",),
            record=_RequiredRecord,
            fields=(field,),
        )

    optional = ConfigTableSpec(
        path=("sample",),
        record=_RequiredRecord,
        fields=(field,),
        default_present=False,
    )
    assert optional.default_value(field) is None


def test_complete_parser_registration_rejects_a_missing_destination() -> None:
    parser = build_parser()
    action = next(item for item in parser._actions if item.dest == "seed")
    action.dest = "drifted_seed"
    with pytest.raises(ValueError, match="config/CLI drift"):
        validate_model_parser(parser, "ltx2")


def test_complete_parser_registration_rejects_a_cli_default() -> None:
    parser = build_parser()
    action = next(item for item in parser._actions if item.dest == "seed")
    action.default = 42
    with pytest.raises(ValueError, match="config/CLI default drift for seed"):
        validate_model_parser(parser, "ltx2")


def test_complete_parser_registration_rejects_stale_help() -> None:
    parser = build_parser()
    action = next(item for item in parser._actions if item.dest == "seed")
    action.help = "A stale hand-written description."
    with pytest.raises(ValueError, match="config/CLI help drift for seed"):
        validate_model_parser(parser, "ltx2")


def test_separate_negative_setting_flags_use_registry_negative_help() -> None:
    parser = build_parser()
    schema = config_registry().model("ltx2")
    positive = next(item for item in parser._actions if item.option_strings == ["--fast-mode"])
    negative = next(item for item in parser._actions if item.option_strings == ["--no-fast-mode"])

    assert positive.help == schema.cli_help("fast_mode")
    assert negative.help == schema.cli_help("fast_mode", negative=True)
    assert negative.help == "Set fast mode to false."

    negative.help = schema.cli_help("fast_mode")
    with pytest.raises(ValueError, match="config/CLI help drift for fast_mode"):
        validate_model_parser(parser, "ltx2")


def test_save_all_coverage_rejects_an_unclassified_sidecar() -> None:
    schema = config_registry().model("gmnet")
    implication = schema.implications[0]
    broken = replace(
        implication,
        implied=implication.implied[:-1],
    )
    with pytest.raises(ValueError, match="coverage drift.*save_all_candidate"):
        replace(schema, implications=(broken,))


def test_save_all_coverage_rejects_an_untagged_new_boolean() -> None:
    @dataclass(frozen=True)
    class _Output:
        save_known: bool = False
        save_new: bool = False
        save_all_sidecars: bool = False

    output = ConfigTableSpec(
        path=("output",),
        record=_Output,
        fields=(
            ConfigFieldSpec(
                "save_known",
                "Save the known sidecar",
                "true or false",
                True,
                "save_known",
                groups=frozenset({ConfigGroup.SAVE_ALL_CANDIDATE}),
            ),
            ConfigFieldSpec(
                "save_new",
                "Save the new sidecar",
                "true or false",
                True,
                "save_new",
            ),
            ConfigFieldSpec(
                "save_all_sidecars",
                "Save every sidecar",
                "true or false",
                True,
                "save_all_sidecars",
            ),
        ),
    )
    model = ConfigFieldSpec(
        "model",
        "Select the sample model",
        "",
        "sample",
        "model",
        virtual=True,
        default="sample",
        choices=("sample",),
    )
    implication = ConfigImplication(
        trigger=("output", "save_all_sidecars"),
        trigger_value=True,
        implied=((("output", "save_known"), True),),
        coverage_group=ConfigGroup.SAVE_ALL_CANDIDATE,
    )

    with pytest.raises(ValueError, match="unclassified save-all boolean fields.*save_new"):
        ModelConfigSpec(
            model="sample",
            title="Sample configuration",
            introduction=("Describe the sample configuration.",),
            root_fields=(model,),
            tables=(output,),
            implications=(implication,),
        )


def test_default_text_cannot_override_a_concrete_non_environment_default() -> None:
    @dataclass(frozen=True)
    class _Record:
        value: int = 1

    field = ConfigFieldSpec(
        "value",
        "Set the value",
        "an integer",
        1,
        "value",
        default_text="one",
    )
    with pytest.raises(ValueError, match="default_text is permitted only"):
        ConfigTableSpec(path=("sample",), record=_Record, fields=(field,))


def test_field_purpose_requires_the_consistent_imperative_style() -> None:
    with pytest.raises(ValueError, match="imperative convention"):
        ConfigFieldSpec("value", "Output value", "an integer", 1, "value")


def test_field_rejects_an_unregistered_semantic_group() -> None:
    with pytest.raises(ValueError, match="unknown semantic groups: typo"):
        ConfigFieldSpec(
            "value",
            "Set the value",
            "an integer",
            1,
            "value",
            groups=frozenset({"typo"}),
        )


def test_template_keeps_human_readable_blocks_and_line_width() -> None:
    template = config_template("ltx2")
    assert "prompt = '''\nDescribe your video here.\n'''" in template
    assert '# Valid values:\n#   "auto"' in template
    assert (
        '# [generate.image]\n# path = "input.png"\n# frame_index = 0\n# strength = 0.95' in template
    )
    assert "# width = 1024\n\n\n# Set the output height" in template
    assert max(map(len, template.splitlines())) <= 96


def test_every_template_field_keeps_complete_structural_documentation() -> None:
    for model in template_models():
        schema = config_registry().model(model)
        fields = [(None, field, True) for field in schema.root_fields]
        fields.extend(
            (table, field, table.starter_enabled)
            for table in schema.tables
            for field in table.fields
        )
        for table, field, include_assignment in fields:
            lines = _render_field(
                field,
                table=table,
                table_enabled=True,
                include_assignment=include_assignment,
            )
            assert lines[0].startswith(f"# {field.purpose.split()[0]}")
            assert (
                sum(line.startswith("# Valid:") or line == "# Valid values:" for line in lines) == 1
            )
            assert sum(line.startswith("# Default:") for line in lines) == 1
            assert "" not in lines
            assignments = [
                line
                for line in lines
                if line.startswith((f"{field.name} = ", f"# {field.name} = "))
            ]
            assert len(assignments) == int(include_assignment)
            assert all(" # " not in assignment for assignment in assignments)


def test_every_template_table_keeps_its_field_block_spacing() -> None:
    for model in template_models():
        for table in config_registry().model(model).tables:
            rendered = "\n".join(_render_table(table))
            if table.starter_enabled:
                assert rendered.count("\n\n\n") == len(table.fields)
                continue
            assert "\n\n" not in rendered
            for field in table.fields:
                assert f"# {field.name}\n" in rendered


def test_default_init_writes_a_round_trippable_ltx2_config(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["config", "init"]) == 0

    output = tmp_path / "kino-config.toml"
    generated = output.read_text(encoding="utf-8")
    assert "#   kinomlx --config kino-config.toml" in generated
    parsed = tomllib.loads(generated)
    assert parsed["model"] == "ltx2"
    assert parsed["generate"]["prompt"] == "Describe your video here.\n"
    assert "[expand]" not in generated

    capsys.readouterr()
    assert main(["--config", str(output), "--print-config"]) == 0
    resolved = tomllib.loads(capsys.readouterr().out)
    assert resolved["model"] == "ltx2"
    assert resolved["generate"]["prompt"] == "Describe your video here.\n"


def test_selected_output_writes_a_round_trippable_gmnet_config(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "gmnet.toml"

    assert (
        main(
            [
                "config",
                "init",
                "--model",
                "gmnet",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    generated = output.read_text(encoding="utf-8")
    assert f"#   kinomlx --config {output}" in generated
    parsed = tomllib.loads(generated)
    assert parsed["model"] == "gmnet"
    assert parsed["expand"]["image"] == "input.png"
    assert "[generate]" not in generated

    capsys.readouterr()
    assert main(["--config", str(output), "--print-config"]) == 0
    assert tomllib.loads(capsys.readouterr().out)["model"] == "gmnet"


def test_init_refuses_to_replace_an_existing_path(tmp_path: Path, caplog) -> None:
    output = tmp_path / "existing.toml"
    output.write_text("keep me\n", encoding="utf-8")

    assert main(["config", "init", "--output", str(output)]) == 2

    assert output.read_text(encoding="utf-8") == "keep me\n"
    assert f"output already exists: {output}" in caplog.text
    assert "choose another path or remove the existing file" in caplog.text


def test_config_init_help_names_model_selection(capsys) -> None:
    parser = build_config_parser()
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["init", "--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "--model {gmnet,ltx2}" in output
    assert "--output OUTPUT" in output


def test_config_without_a_subcommand_names_the_public_choice(capsys) -> None:
    parser = build_config_parser()
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args([])
    assert exit_info.value.code == 2
    assert "{init}" in capsys.readouterr().err
