"""Shared collection owns source precedence and admission ordering."""

import argparse

import pytest

from kinomlx.config import ConfigError
from kinomlx.config.assembly import collect_invocation_sources
from kinomlx.config.provenance import PreparedConfig
from kinomlx.model_registry import config_registry


def test_collection_keeps_named_raw_sources_and_replacement_semantics(tmp_path):
    config = tmp_path / "run.toml"
    config.write_text('[output]\nprefix="from-file"\n')
    options = argparse.Namespace(
        config=config,
        set_overrides=['output={prefix="from-set"}'],
        output_dir=tmp_path,
    )
    inputs = collect_invocation_sources(config_registry().model("gmnet"), options).prepare(
        PreparedConfig({"output": {"path": "saved.exr"}}, {})
    )
    assert inputs.file.source == str(config)
    assert inputs.file.raw == {"output": {"prefix": "from-file"}}
    assert inputs.cli.source == "command line"
    assert inputs.override.source == "--set"
    assert inputs.merge()["output"] == {
        "prefix": "from-set",
        "path": None,
        "directory": tmp_path,
    }
    assert inputs.file.raw == {"output": {"prefix": "from-file"}}


def test_all_sources_are_admitted_before_any_source_rule_reads_files(tmp_path):
    schema = config_registry().model("flux2")
    with pytest.raises(ConfigError, match="later source.*unknown field"):
        schema.prepare_layers(
            (
                ("earlier source", {"generate": {"prompt_file": str(tmp_path / "missing")}}),
                ("later source", {"output": {"typo": 1}}),
            )
        )


def test_invocation_sources_admit_their_layers_before_any_selector_is_read(tmp_path):
    config = tmp_path / "run.toml"
    config.write_text('[output]\ntypo="x"\n')
    options = argparse.Namespace(config=config, set_overrides=None, output_dir=tmp_path)

    with pytest.raises(ConfigError, match="unknown field"):
        collect_invocation_sources(config_registry().model("gmnet"), options)


def test_one_schema_index_and_one_composed_state_serve_every_reader(tmp_path):
    schema = config_registry().model("gmnet")
    options = argparse.Namespace(config=None, set_overrides=None, output_dir=tmp_path)
    inputs = collect_invocation_sources(schema, options).prepare()

    assert schema.field_items() is schema.field_items()
    assert inputs.state() is inputs.state()
    assert inputs.provenance() is inputs.state().origins
    assert inputs.explicit_paths() == inputs.state().explicit
