"""Supplied errors cannot disappear behind higher-precedence assignments."""

import importlib

import pytest

from kinomlx.config import ConfigError
from kinomlx.config_surfaces import SOURCE_SURFACES, STRUCTURED_SURFACES
from kinomlx.model_registry import config_registry


@pytest.mark.parametrize("model", ["gmnet", "flux2", "krea2", "ltx2"])
@pytest.mark.parametrize("invalid", ["prefix=9", 'prefxi="typo"'])
def test_invalid_file_field_is_not_hidden_by_replacing_output(tmp_path, model, invalid):
    parser = importlib.import_module(f"kinomlx.models.{model}.arguments").build_parser()
    assemble = importlib.import_module(f"kinomlx.models.{model}.config").assemble
    config = tmp_path / "invalid.toml"
    config.write_text(f'model="{model}"\n[output]\n{invalid}\n')
    with pytest.raises(ConfigError, match="prefix|prefxi"):
        assemble(
            parser.parse_args(["--print-config", "--config", str(config), "--set", "output={}"])
        )


@pytest.mark.parametrize("model", ["gmnet", "flux2", "krea2", "ltx2"])
def test_each_override_is_admitted_even_when_later_replaced(model):
    schema = config_registry().model(model)
    with pytest.raises(ConfigError, match=r"--set #1.*prefix"):
        schema.parse_overrides(["output.prefix=9", "output={}"])


def test_layer_admission_does_not_apply_cross_field_rules():
    schema = config_registry().model("ltx2")
    stream = schema.parse_overrides(
        ["generate={frames=17,auto_duration=true}", "generate.auto_duration=false"]
    )
    # A later scalar replacement removes this conflict. Group overlays alone
    # do not remove omitted siblings; rules run on the surviving intent.
    assert schema.prepare_layer(stream.config)["generate"] == {"frames": 17, "duration": None}


def test_group_overlay_keeps_duration_conflict_even_when_values_match():
    schema = config_registry().model("ltx2")
    stream = schema.parse_overrides(["generate={frames=17,duration=5}", "generate={frames=121}"])
    with pytest.raises(ConfigError, match="cannot be combined"):
        schema.prepare_layer(stream.config)


def test_unknown_field_fails_before_source_file_is_opened(tmp_path):
    schema = config_registry().model("flux2")
    with pytest.raises(ConfigError, match="unknown field"):
        schema.prepare_layer({"generate": {"prompt_file": str(tmp_path / "missing"), "typo": 1}})


def test_quoted_large_seeds_remain_valid_source_input():
    schema = config_registry().model("flux2")
    stream = schema.parse_overrides(['generate.seed="18446744073709551615"'])
    assert (
        schema.normalize_config(schema.prepare_layer(stream.config))["generate"]["seed"]
        == 2**64 - 1
    )


def test_nullable_virtual_fields_require_an_explicit_type():
    from kinomlx.config import ConfigFieldSpec

    with pytest.raises(ValueError, match="need an input annotation"):
        ConfigFieldSpec(
            "virtual_input",
            "Read virtual input",
            "a string",
            "value",
            "virtual_input",
            persistence="source",
            virtual=True,
            default=None,
            surfaces=SOURCE_SURFACES,
        )


def test_record_field_type_cannot_be_overridden_by_schema_metadata():
    from kinomlx.config import ConfigFieldSpec

    with pytest.raises(ValueError, match="cannot be overridden"):
        ConfigFieldSpec(
            "steps",
            "Set steps",
            "integer",
            20,
            "steps",
            persistence="intent",
            input_annotation=str,
            surfaces=STRUCTURED_SURFACES,
        )
