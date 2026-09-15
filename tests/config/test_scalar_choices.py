"""Scalar choices are owned once by typed fields and bound by all surfaces."""

from dataclasses import dataclass, field, replace

import pytest

from kinomlx.config import ConfigFieldSpec, ConfigRecordSpec, ConfigTableSpec
from kinomlx.config_constraints import ChoiceConstraint, field_choices
from kinomlx.config_records import config_record
from kinomlx.config_surfaces import STRUCTURED_SURFACES
from kinomlx.config_values import dataclass_from_mapping
from kinomlx.model_registry import config_registry
from kinomlx.models.flux2.settings import Flux2Settings
from kinomlx.models.gmnet.settings import GMNetSettings
from kinomlx.models.krea2.settings import Krea2Settings
from kinomlx.models.ltx2.settings import LTX2Settings


@config_record
@dataclass(frozen=True)
class Render:
    mode: str = field(
        default="draft", metadata={"config_constraints": (ChoiceConstraint(("draft", "final")),)}
    )


def mode_field():
    choices = field_choices(Render, "mode")
    return ConfigFieldSpec(
        "mode",
        "Select render quality",
        "",
        "draft",
        "mode",
        persistence="intent",
        choices=choices,
        choice_help=(("draft", "Render a preview"), ("final", "Render the final image")),
        surfaces=STRUCTURED_SURFACES,
    )


@pytest.mark.parametrize("value", ["unknown", 1, False, [], None])
def test_api_constructor_and_structured_record_reject_invalid_choice(value):
    with pytest.raises(ValueError, match="mode"):
        Render(mode=value)
    with pytest.raises(ValueError, match="mode"):
        dataclass_from_mapping(Render, {"mode": value}, table="render")


@pytest.mark.parametrize("container", ["table", "record"])
def test_owned_choices_extend_both_table_and_record_schemas(container):
    declaration = mode_field()
    schema = (
        ConfigTableSpec(("render",), Render, (declaration,))
        if container == "table"
        else ConfigRecordSpec(Render, (declaration,))
    )
    assert schema.fields[0].choices == ("draft", "final")
    assert Render(mode="final").mode == "final"
    drift = replace(
        declaration,
        choices=("draft", "other"),
        choice_help=(("draft", "Render a preview"), ("other", "Another mode")),
    )
    with pytest.raises(ValueError, match="choice binding drift"):
        replace(schema, fields=(drift,))


@pytest.mark.parametrize("name", ["variant", "transformer_dtype", "vae_dtype", "vae_decoder"])
def test_krea_surfaces_read_the_typed_owner(name):
    schema = config_registry().model("krea2")
    declaration = dict(schema.field_items())[("model_settings", name)]
    assert declaration.choices == field_choices(Krea2Settings, name)
    for value in declaration.choices:
        assert getattr(Krea2Settings(**{name: value}), name) == value


@pytest.mark.parametrize("choices", [(), ("a", "a"), ("",), (1,), ["a"]])
def test_invalid_choice_declarations_fail(choices):
    with pytest.raises(ValueError, match="unique nonempty strings"):
        ChoiceConstraint(choices)


@pytest.mark.parametrize(
    ("model", "record", "name"),
    [
        ("flux2", Flux2Settings, "variant"),
        ("flux2", Flux2Settings, "transformer_dtype"),
        ("gmnet", GMNetSettings, "variant"),
        ("ltx2", LTX2Settings, "gemma_variant"),
        ("ltx2", LTX2Settings, "model_generation"),
        ("ltx2", LTX2Settings, "video_vae"),
        ("ltx2", LTX2Settings, "transformer_dtype"),
        ("ltx2", LTX2Settings, "adapter_accumulation"),
        ("ltx2", LTX2Settings, "transformer_quantize"),
    ],
)
def test_other_model_settings_use_owned_choices_without_validate_calls(model, record, name):
    field = dict(config_registry().model(model).field_items())[("model_settings", name)]
    assert field.choices == field_choices(record, name)
    for choice in field.choices:
        assert getattr(record(**{name: choice}), name) == choice
    with pytest.raises(ValueError, match=name):
        record(**{name: "unsupported"})
