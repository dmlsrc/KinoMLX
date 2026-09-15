"""Station permissions belong to fields and fail registration when incomplete."""

from dataclasses import dataclass, replace

import pytest

from kinomlx.config import ConfigFieldSpec, ConfigTableSpec
from kinomlx.config.stations import OutputPolicy, StationPolicy
from kinomlx.config_surfaces import MODEL_SURFACES, STRUCTURED_SURFACES
from kinomlx.invocation_controls import MODEL_CONTROL
from kinomlx.model_registry import config_registry
from kinomlx.models.krea2.types import Krea2Request


@dataclass(frozen=True)
class Capture:
    count: int = 1
    gain: float = 1.0


def capture_table():
    return ConfigTableSpec(
        path=("capture",),
        record=Capture,
        stations=("process",),
        station_record=Capture,
        fields=(
            ConfigFieldSpec(
                "count",
                "Number of samples",
                "integer",
                1,
                "count",
                persistence="intent",
                station_policies=(
                    StationPolicy("process", "locked", reason="Processing already used the count."),
                ),
                surfaces=STRUCTURED_SURFACES,
            ),
            ConfigFieldSpec(
                "gain",
                "Processing gain",
                "number",
                1.0,
                "gain",
                persistence="intent",
                station_policies=(
                    StationPolicy("process", "mutable", reason="Processing runs with a new gain."),
                ),
                surfaces=STRUCTURED_SURFACES,
            ),
        ),
    )


@pytest.mark.parametrize(
    ("arguments", "keywords", "message"),
    [
        (("", "mutable"), {}, "requires a station"),
        (("denoise", "sideways"), {}, "requires a station"),
        (("denoise", "mutable"), {"reason": "   "}, "nonempty prose"),
        (("denoise", "locked", "not an attribute"), {}, "typed record attribute"),
        (("denoise", "mutable", "prompts"), {}, "may select a projection"),
        (("denoise", "irrelevant"), {}, "ignore or reject action"),
        (("denoise", "mutable"), {"irrelevant_action": "ignore"}, "ignore or reject action"),
    ],
)
def test_malformed_station_policy_declarations_are_rejected(arguments, keywords, message):
    """Each guard rejects its own malformed declaration, not just one of them."""
    with pytest.raises(ValueError, match=message):
        StationPolicy(*arguments, **keywords)


@pytest.mark.parametrize(
    "produced_at",
    [
        {"generate"},
        frozenset({"not an identifier"}),
        frozenset({"generate", 7}),
    ],
)
def test_output_policy_requires_frozen_station_names(produced_at):
    with pytest.raises(ValueError, match="frozen station names"):
        OutputPolicy(produced_at)


def test_station_declarations_are_immutable():
    policy = StationPolicy("denoise", "mutable", reason="A denoise restart reruns this.")
    with pytest.raises(AttributeError):
        policy.mode = "locked"
    with pytest.raises(AttributeError):
        OutputPolicy(frozenset({"generate"})).produced_at = frozenset()


def test_generic_station_comparison_uses_owned_fields():
    table = capture_table()
    assert table.locked_changes(Capture(gain=2), Capture(), station="process") == ()
    assert table.locked_changes(Capture(count=2), Capture(), station="process") == (
        ("capture", "count"),
    )
    with pytest.raises(ValueError, match="unsupported station"):
        table.locked_changes(Capture(), Capture(), station="missing")
    with pytest.raises(TypeError, match="typed records"):
        table.locked_changes({}, Capture(), station="process")


def test_missing_duplicate_and_stale_station_classification_fail_registration():
    table = capture_table()
    count, gain = table.fields
    for policies in ((), (StationPolicy("wrong", "locked"),), count.station_policies * 2):
        with pytest.raises(ValueError, match="station policy|duplicate station"):
            replace(table, fields=(replace(count, station_policies=policies), gain))
    with pytest.raises(ValueError, match="unknown station projection"):
        replace(
            table,
            fields=(
                replace(count, station_policies=(StationPolicy("process", "locked", "missing"),)),
                gain,
            ),
        )


def test_new_request_field_cannot_silently_escape_station_coverage():
    @dataclass(frozen=True)
    class ExtendedCapture(Capture):
        exposure: float = 0.5

    table = capture_table()
    with pytest.raises(ValueError, match="station record field coverage drift"):
        replace(table, station_record=ExtendedCapture)
    with pytest.raises(ValueError, match="exposure: station policy coverage drift"):
        replace(
            table,
            record=ExtendedCapture,
            station_record=ExtendedCapture,
            fields=(
                *table.fields,
                ConfigFieldSpec(
                    "exposure",
                    "Exposure",
                    "number",
                    0.5,
                    "exposure",
                    persistence="intent",
                    surfaces=STRUCTURED_SURFACES,
                ),
            ),
        )


def krea_table():
    return next(
        table for table in config_registry().model("krea2").tables if table.path == ("generate",)
    )


def test_krea_policy_preserves_existing_station_permissions():
    table = krea_table()
    mutable = {
        station: {
            field.name
            for field in table.fields
            for policy in field.station_policies
            if policy.station == station and policy.mode == "mutable"
        }
        for station in table.stations
    }
    assert mutable["decode"] == {"vae_tiling", "vae_tile_size", "vae_tile_stride"}
    assert mutable["denoise"] == mutable["decode"] | {
        "width",
        "height",
        "seed",
        "steps",
        "guidance",
        "mu",
        "sigmas",
        "noise_backend",
        "adapters",
        "turbo_4step",
        "turbo_4step_strength",
        "turbo_4step_exclusions",
        "turbo_4step_step_strengths",
        "turbo_4step_sigma_window",
        "turbo_sda",
        "turbo_sda_strength",
        "turbo_sda_exclusions",
        "turbo_sda_step_strengths",
        "turbo_sda_sigma_window",
        "text_conditioning",
    }


def test_krea_compares_prompt_meaning_and_applies_station_specific_permissions():
    table = krea_table()
    parent = Krea2Request(prompt="hello")
    assert table.locked_changes(replace(parent, prompt=("hello",)), parent, station="decode") == ()
    assert table.locked_changes(replace(parent, prompt="changed"), parent, station="denoise") == (
        ("generate", "prompt"),
    )
    resized = replace(parent, width=1152)
    assert table.locked_changes(resized, parent, station="denoise") == ()
    assert table.locked_changes(resized, parent, station="decode") == (("generate", "width"),)
    assert (
        table.locked_changes(replace(parent, vae_tiling="single"), parent, station="decode") == ()
    )


def test_flux_policy_preserves_existing_denoise_allowances_and_decode_locks():
    from kinomlx.models.flux2.types import Flux2Request

    table = next(
        table for table in config_registry().model("flux2").tables if table.path == ("generate",)
    )
    mutable = {
        station: {
            field.name
            for field in table.fields
            for policy in field.station_policies
            if policy.station == station and policy.mode == "mutable"
        }
        for station in table.stations
    }
    assert mutable["decode"] == set()
    assert mutable["denoise"] == {
        "adapters",
        "steps",
        "sigmas",
        "guidance",
        "cfg_mode",
        "recompute_reference_kv",
        "seed",
        "noise_backend",
        "text_conditioning",
    }
    parent = Flux2Request(prompt="hello", steps=4)
    changed = replace(parent, steps=6)
    assert table.locked_changes(changed, parent, station="denoise") == ()
    assert table.locked_changes(changed, parent, station="decode") == (("generate", "steps"),)
    changed = replace(parent, width=1024)
    assert table.locked_changes(changed, parent, station="denoise") == (("generate", "width"),)


def test_nested_optional_station_records_preserve_presence_and_child_coverage():
    from kinomlx.config import ModelConfigSpec

    @dataclass(frozen=True)
    class Envelope:
        capture: Capture | None = None

    child = replace(capture_table(), path=("session", "capture"))
    root = ConfigTableSpec(
        path=("session",),
        record=Envelope,
        fields=(),
        nested=("capture",),
        stations=("process",),
        station_record=Envelope,
    )
    schema = ModelConfigSpec(
        model="probe",
        title="Probe",
        introduction=("Probe nested policy.",),
        restart_stations=("process",),
        root_fields=(
            ConfigFieldSpec(
                "model",
                "Select model",
                "",
                "probe",
                "model",
                persistence="intent",
                virtual=True,
                default="probe",
                choices=("probe",),
                surfaces=MODEL_SURFACES,
                cli=MODEL_CONTROL.binding,
            ),
        ),
        tables=(root, child),
    )
    assert schema.locked_changes(("session",), Envelope(), Envelope(), station="process") == ()
    assert schema.locked_changes(
        ("session",), Envelope(Capture()), Envelope(), station="process"
    ) == (("session", "capture", "count"),)
    assert schema.locked_changes(
        ("session",), Envelope(), Envelope(Capture()), station="process"
    ) == (("session", "capture", "count"),)
    assert (
        schema.locked_changes(
            ("session",), Envelope(Capture(gain=2)), Envelope(Capture()), station="process"
        )
        == ()
    )
    unclassified = replace(
        child,
        stations=(),
        station_record=None,
        fields=tuple(replace(field, station_policies=()) for field in child.fields),
    )
    with pytest.raises(ValueError, match="probe.session.capture: station coverage drift"):
        replace(schema, tables=(root, unclassified))


def test_ltx_station_decisions_match_prior_locks_including_whole_conditioning_records():
    schema = config_registry().model("ltx2")
    locked = {
        station: {
            ".".join((*table.path[1:], field.name))
            for table in schema.tables
            if table.path[0] == "generate"
            for field in table.fields
            for policy in field.station_policies
            if policy.station == station and policy.mode == "locked"
        }
        for station in ("decode", "stage-2")
    }
    both = {
        "adapters",
        "duration",
        "fps",
        "frames",
        "generated_keyframes",
        "height",
        "pad_prompt_to_max",
        "prompt",
        "prompt_unconditional",
        "reference_aligned_audio",
        "sampler",
        "stage_1_sigmas",
        "width",
    }
    assert locked["stage-2"] == both
    assert locked["decode"] == both | {
        "stage_2_sigmas",
        "text_conditioning",
        "image.path",
        "image.frame_index",
        "image.strength",
        "hdr_reference.path",
        "hdr_reference.strength",
    }


def test_ltx_nested_changes_report_canonical_fields_and_preserve_station_behavior():
    from pathlib import Path

    from kinomlx.models.ltx2.types import (
        DistilledRequest,
        ImageConditioningConfig,
        VideoVAETilingConfig,
    )

    schema = config_registry().model("ltx2")
    parent = DistilledRequest(image=ImageConditioningConfig(Path("image.png")))
    changed = replace(parent, image=replace(parent.image, strength=0.5))
    assert schema.locked_changes(("generate",), changed, parent, station="decode") == (
        ("generate", "image", "strength"),
    )
    assert schema.locked_changes(("generate",), changed, parent, station="stage-2") == ()
    changed = replace(parent, vae_tiling=VideoVAETilingConfig(mode="single"))
    assert schema.locked_changes(("generate",), changed, parent, station="decode") == ()


@pytest.mark.parametrize("station", ["denoise", "decode"])
def test_flux_model_setting_locks_are_declared_per_field(station):
    from kinomlx.config_records import replace_config
    from kinomlx.models.flux2.settings import Flux2Settings

    baseline = Flux2Settings()
    changed = replace_config(baseline, variant="base")
    schema = config_registry().model("flux2")
    assert schema.locked_changes(("model_settings",), changed, baseline, station=station) == (
        ("model_settings", "variant"),
    )
    assert schema.locked_changes(("model_settings",), baseline, baseline, station=station) == ()


@pytest.mark.parametrize(
    "field",
    [
        "transformer_dtype",
        "source_path",
        "tokenizer_path",
        "text_encoder_path",
        "transformer_path",
        "vae_path",
    ],
)
@pytest.mark.parametrize("station", ["denoise", "decode"])
def test_flux_resource_selection_declares_irrelevance_instead_of_a_lock(field, station):
    """Replay reopens recorded components, so a new resource choice is not a lock."""
    schema = config_registry().model("flux2")
    policy = next(
        policy
        for policy in schema.field(("model_settings", field)).station_policies
        if policy.station == station
    )
    assert (policy.mode, policy.irrelevant_action) == ("irrelevant", "reject")
    assert policy.reason


def test_flux_model_settings_cannot_omit_one_field_station_policy():
    schema = config_registry().model("flux2")
    table = next(table for table in schema.tables if table.path == ("model_settings",))
    field, *others = table.fields
    with pytest.raises(ValueError, match="station policy coverage drift"):
        replace(table, fields=(replace(field, station_policies=()), *others))
