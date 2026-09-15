"""Artifact availability controls inherited defaults, implications and explicit requests."""

import argparse
from dataclasses import replace

import pytest

from kinomlx.config.assembly import collect_invocation_sources
from kinomlx.config.provenance import PreparedConfig
from kinomlx.config.stations import OutputPolicy, StationPolicy, station_output_defaults
from kinomlx.model_registry import config_registry
from tests.config.test_bindings import SensorConfig, sensor_schema


def prepare(model, station, *, assignments=(), **cli):
    schema = config_registry().model(model)
    options = argparse.Namespace(config=None, set_overrides=list(assignments), **cli)
    return collect_invocation_sources(schema, options).prepare(station=station).state()


@pytest.mark.parametrize("model", ["krea2", "flux2"])
@pytest.mark.parametrize("surface", ["cli", "override", "environment"])
def test_decode_save_all_selects_only_available_artifacts(model, surface, monkeypatch):
    kwargs = {}
    if surface == "environment":
        monkeypatch.setenv(f"KINO_{model.upper()}_SAVE_ALL_SIDECARS", "true")
    elif surface == "override":
        kwargs["assignments"] = ("output.save_all_sidecars=true",)
    else:
        kwargs["save_all_sidecars"] = True
    state = prepare(model, "decode", **kwargs)
    output = state.values["output"]
    assert output["save_all_sidecars"] is True
    assert "save_initial_noise" not in output
    assert "save_text_conditioning" not in output
    assert output.get("save_final_latents") is True
    assert ("output", "save_initial_noise") not in state.origins


@pytest.mark.parametrize("model", ["krea2", "flux2"])
def test_decode_restart_may_request_the_latent_it_re_emits(model):
    state = prepare(model, "decode", assignments=("output.save_final_latents=true",))
    assert state.values["output"]["save_final_latents"] is True
    config_registry().model(model).validate_station_outputs(state.values, station="decode")


def test_ltx_decode_restart_may_request_the_latent_it_re_emits():
    state = prepare("ltx2", "decode", assignments=("model_artifacts.save_latents=true",))
    assert state.values["model_artifacts"]["save_latents"] is True
    config_registry().model("ltx2").validate_station_outputs(state.values, station="decode")


@pytest.mark.parametrize("model", ["krea2", "flux2"])
@pytest.mark.parametrize("reverse", [False, True])
def test_explicit_unavailable_request_is_never_hidden_by_save_all(model, reverse):
    assignments = ["output.save_all_sidecars=true", "output.save_initial_noise=true"]
    if reverse:
        assignments.reverse()
    state = prepare(model, "decode", assignments=assignments)
    assert state.values["output"]["save_initial_noise"] is True
    with pytest.raises(ValueError, match="save_initial_noise is not produced"):
        config_registry().model(model).validate_station_outputs(state.values, station="decode")


@pytest.mark.parametrize("model", ["krea2", "flux2"])
def test_explicit_false_survives_station_implication_filter(model):
    state = prepare(
        model,
        "decode",
        assignments=("output.save_initial_noise=false", "output.save_all_sidecars=true"),
    )
    assert state.values["output"]["save_initial_noise"] is False
    assert ("output", "save_initial_noise") in state.explicit
    config_registry().model(model).validate_station_outputs(state.values, station="decode")


def test_neutral_sensor_station_uses_declared_output_policy_without_model_branch():
    schema = sensor_schema()
    table = schema.tables[0]
    policies = (
        StationPolicy("decode", "mutable", reason="The decode station selects this capture input."),
    )
    # An artifact field's mode follows its availability, so a field produced
    # only at generate is irrelevant at decode rather than mutable there.
    unavailable = (
        StationPolicy(
            "decode",
            "irrelevant",
            reason="The decode station does not produce this capture output.",
            irrelevant_action="ignore",
        ),
    )
    table = replace(
        table,
        stations=("decode",),
        station_record=SensorConfig,
        fields=tuple(
            replace(
                field,
                station_policies=unavailable if field.name == "enabled" else policies,
                output_policy=OutputPolicy(frozenset({"generate"}))
                if field.name == "enabled"
                else None,
            )
            for field in table.fields
        ),
    )
    schema = replace(schema, tables=(table,), restart_stations=("decode",), no_restart_reason=None)
    with pytest.raises(ValueError, match="capture.enabled is not produced"):
        schema.validate_station_outputs({"capture": {"enabled": True}}, station="decode")
    assert station_output_defaults(
        {"enabled": True}, {"enabled": OutputPolicy(frozenset({"generate"}))}, station="decode"
    ) == {"enabled": False}


def test_ltx_artifact_station_modes_are_read_from_availability():
    from kinomlx.models.ltx2.artifacts import OUTPUT_POLICIES

    table = config_registry().model("ltx2").table(("model_artifacts",))
    declared = {
        (field.name, policy.station): (policy.mode, policy.irrelevant_action)
        for field in table.fields
        for policy in field.station_policies
    }
    assert declared == {
        ("save_latents", "stage-2"): ("mutable", None),
        ("save_latents", "decode"): ("mutable", None),
        ("save_media_conditioning", "stage-2"): ("mutable", None),
        ("save_media_conditioning", "decode"): ("irrelevant", "ignore"),
        ("save_text_conditioning", "stage-2"): ("mutable", None),
        ("save_text_conditioning", "decode"): ("irrelevant", "ignore"),
    }
    # Every entry above is exactly what the artifact availability table says.
    assert declared == {
        (field, station): ("mutable", None)
        if station in OUTPUT_POLICIES[field].produced_at
        else ("irrelevant", "ignore")
        for field in OUTPUT_POLICIES
        for station in ("stage-2", "decode")
    }
    # Only the modes are derived; each station still explains itself.
    reasons = {policy.reason for field in table.fields for policy in field.station_policies}
    assert len(reasons) == 6
    assert all(reason.strip() for reason in reasons)


@pytest.mark.parametrize("model", ["krea2", "flux2", "ltx2"])
def test_no_model_writes_a_second_answer_to_artifact_availability(model):
    schema = config_registry().model(model)
    checked = {
        (path, policy.station): policy.mode
        for path, declaration in schema.field_items()
        if declaration.output_policy is not None
        for policy in declaration.station_policies
    }
    expected = {
        (path, policy.station): (
            "mutable" if policy.station in declaration.output_policy.produced_at else "irrelevant"
        )
        for path, declaration in schema.field_items()
        if declaration.output_policy is not None
        for policy in declaration.station_policies
    }
    assert checked
    assert checked == expected


def test_output_policy_rejects_non_boolean_owner():
    table = sensor_schema().tables[0]
    fields = tuple(
        replace(field, output_policy=OutputPolicy(frozenset({"generate"})))
        if field.name == "count"
        else field
        for field in table.fields
    )
    with pytest.raises(ValueError, match="output policy requires a boolean field"):
        replace(table, fields=fields)


def test_output_policy_rejects_unknown_station_at_registration():
    schema = sensor_schema()
    table = schema.tables[0]
    table = replace(
        table,
        fields=tuple(
            replace(field, output_policy=OutputPolicy(frozenset({"typo"})))
            if field.name == "enabled"
            else field
            for field in table.fields
        ),
    )
    with pytest.raises(ValueError, match="unknown output stations"):
        replace(schema, tables=(table,))


def test_direct_output_policy_rejects_explicit_unavailable_output_even_with_save_all():
    policy = OutputPolicy(frozenset({"generate"}))
    with pytest.raises(ValueError, match="not produced"):
        policy.selected(True, save_all=True, station="decode", field="save_preview")
    assert not policy.selected(None, save_all=True, station="decode", field="save_preview")


def test_station_filter_does_not_mutate_saved_input():
    state = PreparedConfig({"output": {"save_initial_noise": False}}, {})
    before = state.values.copy()
    config_registry().model("krea2").prepare_station_layer(state, station="decode")
    assert state.values == before


def test_unknown_station_fails_before_preparation():
    with pytest.raises(ValueError, match="unsupported station"):
        prepare("krea2", "deocde", save_all_sidecars=True)


def test_klein_restart_protects_canonical_parent_source_images(tmp_path):
    from kinomlx.config.persistence import configuration_payload
    from kinomlx.io.output_config import ImageOutputConfig
    from kinomlx.models.flux2.command import _source_paths
    from kinomlx.models.flux2.config import Flux2Invocation
    from kinomlx.models.flux2.config_schema import Flux2Operation
    from kinomlx.models.flux2.restart import RestartManifest
    from kinomlx.models.flux2.settings import Flux2Settings
    from kinomlx.models.flux2.types import Flux2Request, Flux2RestartConfig
    from kinomlx.settings import Settings

    image = tmp_path / "original.png"
    retired = tmp_path / "retired.png"
    payload = {
        "schema_version": 2,
        "configuration": configuration_payload(
            "flux2", PreparedConfig({"generate": {"source_images": [str(image)]}}, {})
        ),
        "invocation": {"generate": {"source_images": [str(retired)]}},
    }
    invocation = Flux2Invocation(
        Settings(),
        Flux2Settings(),
        Flux2Operation(),
        Flux2Request(prompt="scene"),
        ImageOutputConfig(),
        {},
        PreparedConfig({}, {}),
        restart=RestartManifest(Flux2RestartConfig(run=tmp_path / "run.json"), payload),
    )
    protected = _source_paths(invocation, argparse.Namespace())
    assert image in protected
    assert retired not in protected


@pytest.mark.parametrize(
    ("station", "field"),
    [
        ("decode", "save_text_conditioning"),
        ("decode", "save_media_conditioning"),
    ],
)
@pytest.mark.parametrize("save_all", [False, True])
def test_ltx_explicit_unavailable_artifacts_fail_with_or_without_save_all(station, field, save_all):
    from kinomlx.models.ltx2.artifacts import LTX2ArtifactConfig, requested_artifacts

    options = LTX2ArtifactConfig(**{field: True})
    with pytest.raises(ValueError, match=f"{field} is not produced"):
        requested_artifacts(
            options, save_all=save_all, has_media_conditioning=True, station=station
        )


def test_ltx_stage_two_explicit_clear_and_media_capability():
    from kinomlx.models.ltx2.artifacts import (
        FINAL_LATENTS,
        STAGE_1_LATENTS,
        TEXT_CONDITIONING,
        LTX2ArtifactConfig,
        requested_artifacts,
    )

    replayable = {STAGE_1_LATENTS, FINAL_LATENTS, TEXT_CONDITIONING}
    assert (
        requested_artifacts(
            LTX2ArtifactConfig(save_media_conditioning=False),
            save_all=True,
            has_media_conditioning=True,
            station="stage-2",
        )
        == replayable
    )
    assert requested_artifacts(LTX2ArtifactConfig(), save_all=True, station="stage-2") == replayable
    with pytest.raises(ValueError, match="save_media_conditioning requires"):
        requested_artifacts(LTX2ArtifactConfig(save_media_conditioning=True), station="stage-2")


def test_ltx_decode_re_emits_only_the_latent_product_it_consumed():
    from kinomlx.models.ltx2.artifacts import (
        FINAL_LATENTS,
        STAGE_1_LATENTS,
        LTX2ArtifactConfig,
        requested_artifacts,
    )

    options = LTX2ArtifactConfig(save_latents=True)
    assert requested_artifacts(options, station="decode") == {FINAL_LATENTS}
    assert requested_artifacts(options, station="decode", source_latent_stage="final") == {
        FINAL_LATENTS
    }
    assert requested_artifacts(options, station="decode", source_latent_stage="stage-1") == {
        STAGE_1_LATENTS
    }


@pytest.mark.parametrize(
    ("station", "field"),
    [
        ("decode", "save_text_conditioning"),
        ("decode", "save_media_conditioning"),
    ],
)
def test_ltx_override_station_admission_uses_artifact_policy(station, field):
    state = prepare("ltx2", station, assignments=(f"model_artifacts.{field}=true",))
    with pytest.raises(ValueError, match=f"{field} is not produced"):
        config_registry().model("ltx2").validate_station_outputs(state.values, station=station)
