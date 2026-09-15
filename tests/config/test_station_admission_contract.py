"""Neutral fixtures for complete typed restart admission and current presence."""

from dataclasses import dataclass, replace

import pytest

from kinomlx.cli.config_schema import settings_table
from kinomlx.cli.image_schema import image_output_table
from kinomlx.config import ConfigFieldSpec, ConfigRecordSpec, ConfigTableSpec, ModelConfigSpec
from kinomlx.config.assembly import ConfigLayer, InvocationLayers
from kinomlx.config.overrides import SetOverrides
from kinomlx.config.provenance import FieldOrigin, PreparedConfig, SourceLocation
from kinomlx.config.stations import OutputPolicy, StationPolicy
from kinomlx.config_records import config_record
from kinomlx.config_surfaces import MODEL_SURFACES, STRUCTURED_SURFACES
from kinomlx.invocation_controls import MODEL_CONTROL


@dataclass(frozen=True)
class Child:
    count: int = 1


@config_record
@dataclass(frozen=True)
class Entry:
    token: str


@dataclass(frozen=True)
class Capture:
    count: int = 1
    gain: int = 1
    ignored: bool = False
    prohibited: bool = False
    child: Child | None = None
    entries: tuple[Entry, ...] = ()


@dataclass(frozen=True)
class Output:
    path: str = "new"


def _field(name: str, policy: StationPolicy, *, destination: str | None = None) -> ConfigFieldSpec:
    return ConfigFieldSpec(
        name=name,
        purpose=f"Configure {name}",
        valid="a typed value",
        example=False if name in {"ignored", "prohibited"} else 1,
        cli_dest=name if destination is None else destination,
        environment_key=name if destination is None else destination,
        persistence="intent",
        surfaces=STRUCTURED_SURFACES,
        station_policies=(policy,),
    )


def _schema() -> ModelConfigSpec:
    entry = ConfigRecordSpec(
        Entry,
        (
            ConfigFieldSpec(
                name="token",
                purpose="Name an entry",
                valid="text",
                example="first",
                cli_dest="entry_token",
                persistence="intent",
                surfaces=STRUCTURED_SURFACES,
            ),
        ),
    )
    child = ConfigTableSpec(
        path=("capture", "child"),
        record=Child,
        fields=(
            _field(
                "count",
                StationPolicy("process", "locked", reason="Child count is precomputed."),
                destination="child_count",
            ),
        ),
        stations=("process",),
        station_record=Child,
    )
    capture = ConfigTableSpec(
        path=("capture",),
        record=Capture,
        nested=("child",),
        fields=(
            _field("count", StationPolicy("process", "locked", reason="Count is precomputed.")),
            _field("gain", StationPolicy("process", "mutable", reason="Gain runs again.")),
            _field(
                "ignored",
                StationPolicy(
                    "process",
                    "irrelevant",
                    reason="The repeated station does not read this value.",
                    irrelevant_action="ignore",
                ),
            ),
            _field(
                "prohibited",
                StationPolicy(
                    "process",
                    "irrelevant",
                    reason="This source-only selection cannot be supplied on replay.",
                    irrelevant_action="reject",
                ),
            ),
            ConfigFieldSpec(
                name="entries",
                purpose="Configure ordered entries",
                valid="a list of entry records",
                example=(),
                cli_dest="entries",
                persistence="intent",
                surfaces=STRUCTURED_SURFACES,
                item_schema=entry,
                station_policies=(
                    StationPolicy("process", "mutable", reason="Entries run again."),
                ),
            ),
        ),
        stations=("process",),
        station_record=Capture,
    )
    output = ConfigTableSpec(
        path=("output",),
        record=Output,
        fields=(
            _field(
                "path",
                StationPolicy("process", "mutable", reason="New output path."),
                destination="output_path",
            ),
        ),
        stations=("process",),
        station_record=Output,
    )
    return ModelConfigSpec(
        model="probe",
        title="Probe",
        introduction=("Probe station ownership.",),
        root_fields=(
            ConfigFieldSpec(
                name="model",
                purpose="Choose a model",
                valid="",
                example="probe",
                cli_dest="model",
                persistence="intent",
                surfaces=MODEL_SURFACES,
                virtual=True,
                default="probe",
                choices=("probe",),
                cli=MODEL_CONTROL.binding,
            ),
        ),
        tables=(capture, child, output),
        restart_stations=("process",),
    )


def test_station_admission_visits_all_roots_and_nested_records():
    schema = _schema()
    parent = {"capture": Capture(child=Child()), "output": Output()}
    current = {"capture": Capture(gain=2, ignored=True, child=Child()), "output": Output("next")}
    assert (
        schema.station_changes(current, parent, station="process", active_explicit=frozenset())
        == ()
    )
    current["capture"] = replace(current["capture"], count=2, child=Child(2))
    assert schema.station_changes(
        current, parent, station="process", active_explicit=frozenset()
    ) == (("capture", "child", "count"), ("capture", "count"))


def test_rejected_irrelevant_input_compares_the_recorded_value():
    schema = _schema()
    parent = {"capture": Capture(), "output": Output()}
    equal = {"capture": Capture(), "output": Output()}
    supplied = frozenset({("capture", "prohibited")})
    assert schema.station_changes(equal, parent, station="process", active_explicit=supplied) == ()
    changed = {**equal, "capture": replace(equal["capture"], prohibited=True)}
    assert schema.station_changes(changed, parent, station="process", active_explicit=supplied) == (
        ("capture", "prohibited"),
    )
    assert (
        schema.station_changes(changed, parent, station="process", active_explicit=frozenset())
        == ()
    )


def test_rejected_irrelevant_input_on_a_shared_record_stays_presence_only():
    """A root both sides share records nothing, so supplying it is the change."""
    schema = _schema()
    parent = {"capture": Capture(), "output": Output()}
    assert schema.station_changes(
        parent,
        parent,
        station="process",
        active_explicit=frozenset({("capture", "prohibited")}),
    ) == (("capture", "prohibited"),)
    assert (
        schema.station_changes(parent, parent, station="process", active_explicit=frozenset()) == ()
    )


def test_station_admission_rejects_missing_or_extra_root_records():
    schema = _schema()
    full = {"capture": Capture(), "output": Output()}
    with pytest.raises(ValueError, match="missing=\\['output'\\]"):
        schema.station_changes(
            {"capture": Capture()}, full, station="process", active_explicit=frozenset()
        )
    with pytest.raises(ValueError, match="stale=\\['other'\\]"):
        schema.station_changes(
            full,
            {**full, "other": object()},
            station="process",
            active_explicit=frozenset(),
        )
    with pytest.raises(TypeError, match="current station record must be Capture"):
        schema.station_changes(
            {**full, "capture": None},
            full,
            station="process",
            active_explicit=frozenset(),
        )
    with pytest.raises(TypeError, match="parent station record must be Output"):
        schema.station_changes(
            full,
            {**full, "output": None},
            station="process",
            active_explicit=frozenset(),
        )


def test_declared_restart_applicability_rejects_incomplete_tables_and_reasons():
    schema = _schema()
    output = schema.tables[-1]
    incomplete = replace(
        output,
        stations=(),
        station_record=None,
        fields=(replace(output.fields[0], station_policies=()),),
    )
    with pytest.raises(ValueError, match="output: station coverage drift"):
        replace(schema, tables=(*schema.tables[:-1], incomplete))
    without_reason = replace(
        output,
        fields=(
            replace(output.fields[0], station_policies=(StationPolicy("process", "mutable"),)),
        ),
    )
    with pytest.raises(ValueError, match="policy needs a reason"):
        replace(schema, tables=(*schema.tables[:-1], without_reason))
    with pytest.raises(ValueError, match="no-restart model requires a reason"):
        replace(
            schema,
            restart_stations=(),
            tables=tuple(
                replace(
                    table,
                    stations=(),
                    station_record=None,
                    fields=tuple(replace(field, station_policies=()) for field in table.fields),
                )
                for table in schema.tables
            ),
        )


def test_shared_settings_and_image_outputs_bind_real_station_decisions():
    settings = settings_table(
        stations=("denoise", "decode"),
        adapter_weight_check_stations=frozenset({"denoise"}),
    )
    by_name = {field.name: field for field in settings.fields}
    assert set(by_name) == {
        "app_cache_dir",
        "check_adapter_weights",
        "hf_home",
        "verbose",
        "quiet",
        "json_output",
        "profile_signposts",
        "profile_signpost_log",
        "mlx_cache_limit_gb",
        "mlx_wired_limit_gb",
    }
    assert all(policy.reason for field in settings.fields for policy in field.station_policies)
    assert [policy.mode for policy in by_name["check_adapter_weights"].station_policies] == [
        "mutable",
        "irrelevant",
    ]
    assert by_name["check_adapter_weights"].station_policies[1].irrelevant_action == "ignore"

    output = image_output_table(
        "probe",
        noise_help="Save noise",
        latents_help="Save latents",
        conditioning_help="Save text",
        output_policies={
            "save_initial_noise": OutputPolicy(frozenset({"generate", "denoise"})),
            "save_final_latents": OutputPolicy(frozenset({"generate", "denoise", "decode"})),
            "save_text_conditioning": OutputPolicy(frozenset({"generate"})),
        },
        stations=("denoise", "decode"),
    )
    by_name = {field.name: field for field in output.fields}
    assert [policy.mode for policy in by_name["save_final_latents"].station_policies] == [
        "mutable",
        "mutable",
    ]
    assert [policy.mode for policy in by_name["save_text_conditioning"].station_policies] == [
        "irrelevant",
        "irrelevant",
    ]
    assert all(policy.reason for field in output.fields for policy in field.station_policies)


def test_current_presence_excludes_inherited_and_masked_values():
    schema = _schema()

    def origin(path, kind, source):
        return FieldOrigin(path, kind, True, SourceLocation(source, source))

    environment = ConfigLayer(
        "environment",
        {"capture": {"count": 3}},
        {"capture": {"count": 3}},
        {("capture", "count"): origin(("capture", "count"), "explicit", "environment")},
    )
    saved = PreparedConfig(
        {"capture": {"count": 1}},
        {("capture", "count"): origin(("capture", "count"), "inherited", "saved")},
    )
    file = ConfigLayer(
        "file",
        {
            "capture": {
                "gain": 1,
                "prohibited": False,
                "child": {"count": 2},
                "entries": [{"token": "old"}, {"token": "removed"}],
            }
        },
        {
            "capture": {
                "gain": 1,
                "prohibited": False,
                "child": {"count": 2},
                "entries": [{"token": "old"}, {"token": "removed"}],
            }
        },
        {
            ("capture", "gain"): origin(("capture", "gain"), "normalized", "file"),
            ("capture", "prohibited"): origin(("capture", "prohibited"), "normalized", "file"),
            ("capture", "child", "count"): origin(
                ("capture", "child", "count"), "explicit", "file"
            ),
            ("capture", "entries", 0, "token"): origin(
                ("capture", "entries", 0, "token"), "explicit", "file"
            ),
            ("capture", "entries", 1, "token"): origin(
                ("capture", "entries", 1, "token"), "explicit", "file"
            ),
        },
    )
    cli = ConfigLayer("cli", {}, {}, {})
    override = ConfigLayer("override", {}, {}, {})
    inputs = InvocationLayers(
        environment, file, cli, override, SetOverrides.parse([]), schema, saved
    )
    assert inputs.state().values["capture"] == {
        "count": 1,
        "gain": 1,
        "prohibited": False,
        "child": {"count": 2},
        "entries": [{"token": "old"}, {"token": "removed"}],
    }
    assert inputs.active_explicit_paths() == frozenset(
        {
            ("capture", "gain"),
            ("capture", "prohibited"),
            ("capture", "child", "count"),
            ("capture", "entries", 0, "token"),
            ("capture", "entries", 1, "token"),
        }
    )
    parent = {"capture": Capture(), "output": Output()}
    assert schema.station_changes(
        parent,
        parent,
        station="process",
        active_explicit=inputs.active_explicit_paths(),
    ) == (("capture", "prohibited"),)

    override = ConfigLayer(
        "override",
        {"capture": {"gain": 2, "entries": [{"token": "new"}]}},
        {"capture": {"gain": 2, "entries": [{"token": "new"}]}},
        {
            ("capture", "gain"): origin(("capture", "gain"), "explicit", "override"),
            ("capture", "entries", 0, "token"): origin(
                ("capture", "entries", 0, "token"), "explicit", "override"
            ),
        },
    )
    inputs = replace(inputs, override=override)
    assert inputs.active_explicit_paths() == frozenset(
        {
            ("capture", "gain"),
            ("capture", "prohibited"),
            ("capture", "child", "count"),
            ("capture", "entries", 0, "token"),
        }
    )
    assert inputs.provenance()[("capture", "gain")].location == SourceLocation(
        "override", "override"
    )
