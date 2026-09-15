"""Declared layer behavior, extension, and registration failures."""

from dataclasses import replace

import pytest

from kinomlx.config import ConfigError
from kinomlx.config.transforms import LayerTransform
from kinomlx.model_registry import config_registry
from kinomlx.models.flux2.config_schema import CONFIG_SCHEMA

_WIDTH = ("generate", "width")
_HEIGHT = ("generate", "height")


def _copy_width(values):
    return {_HEIGHT: values[_WIDTH]} if _WIDTH in values else {}


def _square_rule(**kwargs):
    return LayerTransform(
        "square_geometry", frozenset({_WIDTH}), frozenset({_HEIGHT}), _copy_width, **kwargs
    )


def test_new_rule_runs_from_declaration_without_assembler_edit():
    schema = replace(
        CONFIG_SCHEMA, layer_transforms=(*CONFIG_SCHEMA.layer_transforms, _square_rule())
    )
    original = {"generate": {"width": 1024}, "output": {"prefix": "example"}}
    normalized = schema.normalize_layer(original)
    assert normalized == {
        "generate": {"width": 1024, "height": 1024},
        "output": {"prefix": "example", "path": None},
    }
    assert original == {"generate": {"width": 1024}, "output": {"prefix": "example"}}


def test_rule_receives_only_declared_present_inputs():
    def run(values):
        assert dict(values) == {_WIDTH: 1024}
        with pytest.raises(TypeError):
            values[_WIDTH] = 1
        return {_HEIGHT: values[_WIDTH]}

    rule = replace(_square_rule(), run=run)
    schema = replace(CONFIG_SCHEMA, layer_transforms=(rule,))
    normalized = schema.normalize_layer({"generate": {"width": 1024, "height": 768}})
    assert normalized["generate"]["height"] == 1024


def test_undeclared_writes_are_rejected():
    rule = replace(_square_rule(), run=lambda values: {_WIDTH: 1})
    schema = replace(CONFIG_SCHEMA, layer_transforms=(rule,))
    with pytest.raises(ValueError, match="undeclared writes"):
        schema.normalize_layer({"generate": {"width": 1024}})


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"reads": frozenset({("generate", "typo")})}, "unknown field paths"),
        ({"writes": frozenset()}, "reads and writes must be declared"),
        ({"after": ("missing",)}, "must precede"),
        ({"phase": "unknown"}, "invalid layer phase"),
    ],
)
def test_invalid_rule_declarations_fail_at_schema_construction(changes, error):
    with pytest.raises(ValueError, match=error):
        replace(CONFIG_SCHEMA, layer_transforms=(replace(_square_rule(), **changes),))


def test_conflicting_rule_identity_is_rejected():
    with pytest.raises(ValueError, match="conflicting transform"):
        replace(
            CONFIG_SCHEMA,
            layer_transforms=(_square_rule(), replace(_square_rule(), reads=frozenset({_HEIGHT}))),
        )


def test_overlapping_rules_require_declared_order():
    second = replace(_square_rule(), name="second")
    with pytest.raises(ValueError, match="requires an explicit dependency"):
        replace(CONFIG_SCHEMA, layer_transforms=(_square_rule(), second))
    schema = replace(
        CONFIG_SCHEMA,
        layer_transforms=(_square_rule(), replace(second, after=("square_geometry",))),
    )
    assert schema.normalize_layer({"generate": {"width": 8}})["generate"]["height"] == 8


def test_dependency_cannot_cross_phases_backward():
    second = replace(
        _square_rule(), name="second", phase="before_implications", after=("square_geometry",)
    )
    with pytest.raises(ValueError, match="crosses phases backward"):
        replace(CONFIG_SCHEMA, layer_transforms=(_square_rule(), second))


@pytest.mark.parametrize("model", ["gmnet", "flux2", "krea2", "ltx2"])
def test_shared_fields_install_selection_rules_automatically(model):
    original = config_registry().model(model)
    # No separate model-level registration is necessary when composing fields.
    schema = replace(original, layer_transforms=())
    expected = {"output_selection"} if model == "gmnet" else {"seed_selection", "output_selection"}
    if model != "gmnet":
        expected.add("prompt_file")
    if model == "ltx2":
        expected.update(
            {
                "ltx_duration_selection",
                "ltx_duration_serialization",
                "ltx_vae_tiling_input",
                "ltx_vae_tiling_selection",
                "ltx_vae_tiling_effective",
            }
        )
    assert {rule.name for rule in schema.layer_transforms} == expected
    assert schema.normalize_layer({"output": {"directory": "renders"}})["output"]["path"] is None
    if model != "gmnet":
        assert schema.normalize_layer({"generate": {"seed": 7}})["generate"] == {
            "seed": 7,
            "seeds": None,
        }
        with pytest.raises(ConfigError, match="mutually exclusive"):
            schema.normalize_layer({"generate": {"seed": 7, "seeds": [8]}})


def test_declared_deletion_consumes_a_field_without_a_null_or_input_mutation():
    from kinomlx.config.transforms import DELETE_FIELD

    rule = replace(_square_rule(), run=lambda values: {_HEIGHT: DELETE_FIELD})
    schema = replace(CONFIG_SCHEMA, layer_transforms=(rule,))
    source = {"generate": {"height": 768}}
    assert schema.normalize_layer(source) == {"generate": {}}
    assert source == {"generate": {"height": 768}}


@pytest.mark.parametrize(
    ("generation", "expected"),
    [
        ({"auto_duration": True}, {"frames": None, "duration": None}),
        ({"auto_duration": False}, {}),
        ({"frames": 121}, {"frames": 121, "duration": None}),
        ({"duration": 5.0}, {"duration": 5.0}),
        ({"auto_duration": False, "duration": 5.0}, {"duration": 5.0}),
    ],
)
def test_ltx_duration_behavior_comes_from_schema(generation, expected):
    schema = config_registry().model("ltx2")
    assert schema.normalize_layer({"generate": generation}) == {"generate": expected}


@pytest.mark.parametrize(
    ("generation", "message"),
    [
        ({"auto_duration": "true"}, "must be a boolean"),
        ({"auto_duration": True, "frames": None}, "cannot be combined"),
        ({"auto_duration": True, "duration": 5.0}, "cannot be combined"),
        ({"frames": 121, "duration": 5.0, "fps": 24.0}, "cannot be combined"),
        ({"frames": 17, "duration": 5.0}, "cannot be combined"),
        ({"frames": 121, "duration": float("nan")}, "cannot be combined"),
    ],
)
def test_ltx_duration_conflicts_fail_in_shared_runner(generation, message):
    with pytest.raises(ConfigError, match=message):
        config_registry().model("ltx2").normalize_layer({"generate": generation})


@pytest.mark.parametrize("mode", ["auto", "single"])
def test_effective_tiling_policy_removes_inherited_geometry(mode):
    from kinomlx.config import merge_configs

    schema = config_registry().model("ltx2")
    lower = schema.normalize_layer({"generate": {"vae_tiling": {"spatial_tile_pixels": 256}}})
    assert lower["generate"]["vae_tiling"]["mode"] == "custom"
    upper = schema.normalize_layer({"generate": {"vae_tiling": {"mode": mode}}})
    merged = merge_configs(lower, upper)
    expected = {"generate": {"vae_tiling": {"mode": mode}}}
    assert schema.normalize_effective(merged) == expected
    assert schema.normalize_config(merged) == expected
    assert merged["generate"]["vae_tiling"]["spatial_tile_pixels"] == 256


def test_custom_tiling_geometry_survives_effective_normalization():
    schema = config_registry().model("ltx2")
    config = {"generate": {"vae_tiling": {"mode": "custom", "spatial_tile_pixels": 256}}}
    assert schema.normalize_effective(config) == config


@pytest.mark.parametrize(
    ("values", "cleared"),
    [
        ({"weights_path": "base.safetensors"}, "transformer_path"),
        ({"transformer_path": "transformer"}, "weights_path"),
        ({"gemma_path": "gemma.safetensors"}, "text_encoder_path"),
        ({"text_encoder_path": "text-encoder"}, "gemma_path"),
        ({"video_vae": "conv"}, "video_vae_path"),
    ],
)
def test_ltx_source_alternatives_are_schema_owned(values, cleared):
    result = config_registry().model("ltx2").normalize_layer({"model_settings": values})
    assert result["model_settings"] == {**values, cleared: None}


def test_ltx_same_layer_explicit_sources_remain_authoritative():
    schema = config_registry().model("ltx2")
    values = {
        "weights_path": "base",
        "transformer_path": "override",
        "video_vae": "conv",
        "video_vae_path": "decoder",
    }
    assert schema.normalize_layer({"model_settings": values}) == {"model_settings": values}


def test_compiled_source_group_drift_fails_schema_construction():
    from kinomlx.models.ltx2.config_groups import LTX2ConfigGroup

    schema = config_registry().model("ltx2")
    tables = tuple(
        replace(
            table,
            fields=tuple(
                replace(field, groups=field.groups | {LTX2ConfigGroup.MODEL_SOURCE})
                if field.name == "fast_mode"
                else field
                for field in table.fields
            ),
        )
        if table.path == ("model_settings",)
        else table
        for table in schema.tables
    )
    with pytest.raises(ValueError, match="compiled group .* has drifted"):
        replace(schema, tables=tables)


@pytest.mark.parametrize("model", ["flux2", "krea2", "ltx2"])
def test_source_preparation_reads_prompt_once_and_pure_normalization_does_not(tmp_path, model):
    path = tmp_path / "prompt.txt"
    path.write_text("literal {{text}}")
    schema = config_registry().model(model)
    raw = {"generate": {"prompt_file": str(path)}}
    assert schema.normalize_layer(raw) == raw
    prepared = schema.prepare_layer(raw)
    assert prepared == {"generate": {"prompt": "literal {{text}}"}}
    path.unlink()
    assert schema.prepare_layer(prepared) == prepared
    with pytest.raises(ConfigError, match="not a file"):
        schema.prepare_layer(raw)


def test_file_effects_cannot_run_in_a_pure_phase():
    with pytest.raises(ValueError, match="filesystem effects require the source phase"):
        replace(
            CONFIG_SCHEMA,
            layer_transforms=(replace(_square_rule(), effects=frozenset({"read_files"})),),
        )
