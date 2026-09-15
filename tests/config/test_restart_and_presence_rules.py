"""Shared restart owners and declared presence-sensitive semantic rules."""

from dataclasses import dataclass, field, fields

import pytest

from kinomlx.config_constraints import field_choices
from kinomlx.config_records import config_record, record_intent, replace_config
from kinomlx.config_rules import RecordRule, RuleInputs, validate_record
from kinomlx.models.flux2.adapters.inputs import Flux2AdapterConfig
from kinomlx.models.flux2.restart import Flux2Restart
from kinomlx.models.flux2.restart import RestartManifest as FluxManifest
from kinomlx.models.flux2.types import Flux2Request, Flux2RestartConfig
from kinomlx.models.krea2.adapters.inputs import Krea2AdapterConfig
from kinomlx.models.krea2.restart import Krea2Restart
from kinomlx.models.krea2.restart import RestartConfig as KreaRestartConfig
from kinomlx.models.krea2.restart import load_restart_manifest as load_krea
from kinomlx.models.krea2.types import Krea2GenerateConfig
from kinomlx.models.ltx2.restart import DistilledRestart
from kinomlx.models.ltx2.restart import RestartConfig as LTXRestartConfig
from kinomlx.models.ltx2.restart import load_restart_manifest as load_ltx


def test_still_restart_surfaces_share_field_objects():
    for first, second in (
        (Flux2RestartConfig, KreaRestartConfig),
        (Flux2Restart, Krea2Restart),
    ):
        second_fields = {item.name: item for item in fields(second)}
        for item in fields(first):
            assert item is second_fields[item.name]
        assert field_choices(first, "phase") == ("decode", "denoise")


@pytest.mark.parametrize("factory", [Flux2RestartConfig, KreaRestartConfig])
def test_still_selector_conflict_is_final_and_an_override_can_resolve_it(factory):
    selection = factory(run="absent.json", phase="denoise", latents="latents.safetensors")
    with pytest.raises(ValueError, match=r"\[restart\]: restart latents apply only"):
        validate_record(selection, location="[restart]")
    validate_record(replace_config(selection, phase="decode"), location="[restart]")


@pytest.mark.parametrize("factory", [Flux2Restart, Krea2Restart])
@pytest.mark.parametrize(
    ("values", "override"),
    [
        ({}, {"latents": "latents.safetensors"}),
        ({"phase": "denoise"}, {"text_conditioning": "text.safetensors"}),
        (
            {"latents": "latents.safetensors", "text_conditioning": "text.safetensors"},
            {"text_conditioning": None},
        ),
    ],
)
def test_public_still_artifacts_use_the_same_final_rule(factory, values, override):
    selection = factory(**values)
    with pytest.raises(ValueError, match=r"\[restart\]: (?:decode|denoise) restart"):
        validate_record(selection, location="[restart]")
    validate_record(replace_config(selection, **override), location="[restart]")


@pytest.mark.parametrize(
    ("factory", "load", "values"),
    [
        (
            Flux2RestartConfig,
            FluxManifest.load,
            {"phase": "denoise", "latents": "missing.safetensors"},
        ),
        (KreaRestartConfig, load_krea, {"phase": "denoise", "latents": "missing.safetensors"}),
        (LTXRestartConfig, load_ltx, {"phase": "stage-2", "latent_stage": "final"}),
    ],
)
def test_manifest_policy_precedes_missing_source_read(factory, load, values, tmp_path):
    selection = factory(run=tmp_path / "absent.json", **values)
    with pytest.raises(
        ValueError, match=r"\[restart\]: restart latent(?:s apply|_stage applies) only"
    ):
        load(selection)


def test_ltx_input_default_is_not_written_into_retained_selection():
    selection = LTXRestartConfig(run="run.json")
    assert selection.latent_stage is None
    assert "latent_stage" not in record_intent(selection)
    stage_two = replace_config(selection, phase="stage-2")
    validate_record(stage_two, location="[restart]")
    assert stage_two.latent_stage is None
    public = DistilledRestart(latents="latents.safetensors")
    assert public.latent_stage is None
    assert public.selected_latent_stage == "final"
    public = replace_config(public, phase="stage-2", text_conditioning="text.safetensors")
    validate_record(public, location="[restart]")
    assert public.selected_latent_stage == "stage-1"


@pytest.mark.parametrize("record", [Krea2GenerateConfig, Flux2Request])
def test_adapter_member_conflict_reports_concrete_index(record):
    request = record(
        adapters=[
            {"path": "first.safetensors", "step_strengths": [1.0]},
            {"path": "second.safetensors", "strength": 1.0, "step_strengths": [1.0]},
        ]
    )
    with pytest.raises(
        ValueError, match=r"\[generate\].adapters\[1\]: strength and step_strengths"
    ):
        validate_record(request, location="[generate]")


@pytest.mark.parametrize("record", [Krea2AdapterConfig, Flux2AdapterConfig])
def test_direct_adapter_selection_uses_the_owned_presence_policy(record):
    implicit = record(path="adapter.safetensors", step_strengths=[1.0])
    implicit.selection()
    explicit = replace_config(implicit, strength=1.0)
    with pytest.raises(ValueError, match="strength and step_strengths"):
        explicit.selection()


def _layout(values: RuleInputs) -> None:
    if values["width_mm"] is not None and values.presence["paper"]:
        raise ValueError("explicit paper size conflicts with custom width")


LAYOUT = RecordRule(
    "layout",
    frozenset({"paper", "width_mm"}),
    _layout,
    presence_reads=frozenset({"paper"}),
)


@config_record
@dataclass(frozen=True)
class ExportLayout:
    paper: str = field(default="A4", metadata={"config_rules": (LAYOUT,)})
    width_mm: float | None = None


def test_non_adapter_rule_distinguishes_omitted_from_explicit_default():
    implicit = ExportLayout(width_mm=100)
    validate_record(implicit, location="[export]")
    explicit = replace_config(implicit, paper="A4")
    with pytest.raises(ValueError, match="explicit paper size"):
        validate_record(explicit, location="[export]")
    validate_record(replace_config(explicit, width_mm=None), location="[export]")
    assert LAYOUT.describe()["presence_reads"] == ["paper"]


def test_rule_can_only_read_declared_presence_and_cannot_mutate_it():
    def check(values: RuleInputs):
        assert values.presence["paper"] is False
        with pytest.raises(KeyError):
            _ = values.presence["width_mm"]
        with pytest.raises(TypeError):
            values.presence["paper"] = True

    @config_record
    @dataclass(frozen=True)
    class Layout:
        paper: str = field(
            default="A4",
            metadata={
                "config_rules": (
                    RecordRule(
                        "layout",
                        frozenset({"paper", "width_mm"}),
                        check,
                        presence_reads=frozenset({"paper"}),
                    ),
                )
            },
        )
        width_mm: float = 100

    validate_record(Layout(), location="[export]")


def test_presence_rule_refuses_untracked_records():
    @dataclass(frozen=True)
    class UntrackedLayout:
        paper: str = field(default="A4", metadata={"config_rules": (LAYOUT,)})
        width_mm: float | None = None

    with pytest.raises(TypeError, match="presence reads require a tracked record"):
        validate_record(UntrackedLayout(), location="[export]")


def test_presence_reads_cannot_escape_the_declared_read_set():
    with pytest.raises(ValueError, match="presence reads"):
        RecordRule("layout", frozenset({"paper"}), _layout, presence_reads=frozenset({"width_mm"}))
