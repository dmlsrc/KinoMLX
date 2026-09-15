"""Generation inputs and capability observations have distinct owned phases."""

import re
from dataclasses import dataclass, field
from importlib import import_module

import pytest

from kinomlx.config import ConfigError
from kinomlx.config.inspection import configuration_inventory
from kinomlx.config_records import config_record, replace_config
from kinomlx.config_rules import RecordRule, RuleInputs, validate_capabilities, validate_record
from kinomlx.model_registry import config_registry
from kinomlx.models.flux2.types import Flux2Request
from kinomlx.models.krea2.types import Krea2GenerateConfig, Krea2Request
from kinomlx.models.ltx2.types import (
    DistilledRequest,
    HDRReferenceConditioningConfig,
    ImageConditioningConfig,
)


@pytest.mark.parametrize("name", ["width", "height", "steps", "guidance"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), True])
def test_klein_invalid_local_numeric_values_fail_at_admission(name, value):
    with pytest.raises(ValueError, match=name):
        Flux2Request(**{name: value})


@pytest.mark.parametrize(
    ("record", "name"),
    [
        (Flux2Request, "sigmas"),
        (Krea2Request, "sigmas"),
        (Krea2GenerateConfig, "sigmas"),
        (DistilledRequest, "stage_1_sigmas"),
        (DistilledRequest, "stage_2_sigmas"),
    ],
)
@pytest.mark.parametrize("sigmas", [(0.5, 1.0, 0.0), (1.0,), (1.0, float("nan")), (1.1, 0.0)])
def test_all_explicit_schedule_fields_share_local_admission(record, name, sigmas):
    with pytest.raises(ValueError, match=name):
        record(**{name: sigmas})


@pytest.mark.parametrize("model", ["krea2", "flux2", "ltx2"])
def test_bad_lower_schedule_is_not_hidden_by_valid_override(model, tmp_path):
    name = "stage_1_sigmas" if model == "ltx2" else "sigmas"
    path = tmp_path / "config.toml"
    path.write_text(f"[generate]\n{name} = [0.5, 1.0, 0.0]\n")
    parser = import_module(f"kinomlx.models.{model}.arguments").build_parser()
    assemble = import_module(f"kinomlx.models.{model}.config").assemble
    options = parser.parse_args(
        ["--config", str(path), "--set", f"generate.{name}=[1.0, 0.0]", "--print-config"]
    )
    with pytest.raises(ConfigError, match=name):
        assemble(options)


@pytest.mark.parametrize(
    ("values", "override", "message"),
    [
        (
            {"image": ImageConditioningConfig(path="missing.exr")},
            {"hdr": "ACESCG"},
            "EXR conditioning requires an explicit --hdr signal interpretation",
        ),
        (
            {"hdr_reference": HDRReferenceConditioningConfig(path="missing.mp4")},
            {"hdr": "ACESCG"},
            "hdr_reference requires an explicit HDR output mode",
        ),
        (
            {"frames": 1, "generated_keyframes": 1},
            {"frames": 9},
            "1 frames provide only 0 interior keyframe slots",
        ),
    ],
)
def test_ltx_input_relationships_resolve_before_source_observation(values, override, message):
    request = DistilledRequest(**values)
    with pytest.raises(ValueError, match=rf"\[generate\]: {re.escape(message)}"):
        validate_record(request, location="[generate]")
    validate_record(replace_config(request, **override), location="[generate]")


def _encoder_support(values: RuleInputs) -> None:
    if values["alpha"] and not values.context["supports_alpha"]:
        raise ValueError("selected encoder does not support alpha")
    with pytest.raises(KeyError):
        _ = values.context["undeclared"]
    with pytest.raises(TypeError):
        values.context["supports_alpha"] = True


ENCODER_SUPPORT = RecordRule(
    "encoder_support",
    frozenset({"alpha"}),
    _encoder_support,
    phase="capability",
    context_reads=frozenset({"supports_alpha"}),
)


@config_record
@dataclass(frozen=True)
class ExportImage:
    alpha: bool = field(default=False, metadata={"config_rules": (ENCODER_SUPPORT,)})


@config_record
@dataclass(frozen=True)
class ExportBatch:
    images: tuple[ExportImage, ...] = ()


def test_capability_phase_requires_declared_observations_and_keeps_them_readonly():
    request = ExportImage(alpha=True)
    validate_record(request, location="[export]")
    with pytest.raises(ValueError, match="missing required observations: supports_alpha"):
        validate_capabilities(request, {}, location="[export]")
    with pytest.raises(ValueError, match="selected encoder"):
        validate_capabilities(request, {"supports_alpha": False}, location="[export]")
    validate_capabilities(
        request, {"supports_alpha": True, "undeclared": "not exposed"}, location="[export]"
    )


def test_capability_validation_descends_with_concrete_indexes():
    request = ExportBatch(images=[{}, {"alpha": True}])
    with pytest.raises(ValueError, match=r"\[export\].images\[1\]: selected encoder"):
        validate_capabilities(request, {"supports_alpha": False}, location="[export]")


@pytest.mark.parametrize(
    "options",
    [{"phase": "unknown"}, {"phase": "capability"}, {"context_reads": frozenset({"source"})}],
)
def test_invalid_phase_or_observation_declarations_fail_registration(options):
    with pytest.raises(ValueError, match="phase|context reads"):
        RecordRule("invalid", frozenset({"alpha"}), _encoder_support, **options)


def test_inspection_reports_capability_owner_and_exact_observation_dependencies():
    inventory = configuration_inventory(
        config_registry().model("ltx2"), field="generate.generate_audio"
    )
    rules = inventory["fields"][0]["validation_rules"]
    owned = next(rule for rule in rules if rule["rule"] == "distilled_capabilities")
    assert owned["phase"] == "capability"
    assert "generates_audio" in owned["context_reads"]
    assert owned["effects"] == []
