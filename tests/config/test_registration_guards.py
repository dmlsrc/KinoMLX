"""Every registration guard on a model contribution refuses what it names.

These guards run once, when a model contribution is registered, and each is the
only thing standing between a contribution mistake and a surface that would carry
it: a shadowed CLI destination silently captures another field's flag, and a
settings record whose environment source drifts from the registry's makes
``from_env()`` read a variable the CLI never writes. The fixtures below build one
minimal contribution and perturb exactly one declaration at a time.
"""

from dataclasses import dataclass
from typing import ClassVar

import pytest

from kinomlx.config import ConfigFieldSpec, ConfigTableSpec, ModelConfigSpec
from kinomlx.config.stages import OutputPolicy, StagePolicy
from kinomlx.config_surfaces import MODEL_SURFACES, STRUCTURED_SURFACES, SurfacePolicy
from kinomlx.invocation_controls import MODEL_CONTROL
from kinomlx.settings import EnvironmentSettings

_ROOT_SURFACES = SurfacePolicy(
    (
        ("api", "A root selector is not a record member."),
        ("cli", "A root selector uses a structured input surface."),
    )
)


@dataclass(frozen=True)
class Capture:
    steps: int = 1
    save_trace: bool = False


@dataclass(frozen=True)
class ProbeSettings(EnvironmentSettings):
    environment_prefix: ClassVar[str] = "KINO_PROBE_"
    config_table: ClassVar[str] = "[model_settings]"

    alpha: int = 1


def _field(
    name: str,
    *,
    destination: str | None = None,
    environment_key: str | None = None,
    example: object = 1,
    mode: str = "mutable",
    output_policy: OutputPolicy | None = None,
) -> ConfigFieldSpec:
    return ConfigFieldSpec(
        name=name,
        purpose=f"Configure {name}",
        valid="a typed value",
        example=example,
        cli_dest=name if destination is None else destination,
        environment_key=name if environment_key is None else environment_key,
        persistence="intent",
        surfaces=STRUCTURED_SURFACES,
        output_policy=output_policy,
        stage_policies=(
            StagePolicy(
                "process",
                mode,
                reason="The stage runs again.",
                irrelevant_action="ignore" if mode == "irrelevant" else None,
            ),
        ),
    )


def _model_field() -> ConfigFieldSpec:
    return ConfigFieldSpec(
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
    )


def _capture_table(
    *,
    trace_produced_at: frozenset[str] = frozenset({"acquire", "process"}),
    trace_mode: str = "mutable",
) -> ConfigTableSpec:
    return ConfigTableSpec(
        path=("capture",),
        record=Capture,
        fields=(
            _field("steps"),
            _field(
                "save_trace",
                example=False,
                mode=trace_mode,
                output_policy=OutputPolicy(trace_produced_at),
            ),
        ),
        stages=("process",),
        stage_record=Capture,
    )


def _settings_table(
    *,
    destination: str = "alpha",
    environment_key: str | None = None,
) -> ConfigTableSpec:
    return ConfigTableSpec(
        path=("model_settings",),
        record=ProbeSettings,
        fields=(_field("alpha", destination=destination, environment_key=environment_key),),
        stages=("process",),
        stage_record=ProbeSettings,
    )


def _spec(**overrides: object) -> ModelConfigSpec:
    declared: dict[str, object] = {
        "model": "probe",
        "title": "Probe",
        "introduction": ("Probe the registration guards.",),
        "root_fields": (_model_field(),),
        "tables": (_capture_table(), _settings_table()),
        "stages": ("acquire", "process"),
        "restart_stages": ("process",),
    }
    declared.update(overrides)
    return ModelConfigSpec(**declared)  # type: ignore[arg-type]


def test_the_unperturbed_contribution_registers() -> None:
    """Each refusal below is one changed declaration away from this baseline."""
    spec = _spec()
    assert spec.model == "probe"
    assert [table.path for table in spec.tables] == [("capture",), ("model_settings",)]


def test_duplicate_root_fields_are_refused() -> None:
    with pytest.raises(ValueError, match="probe: duplicate root configuration fields"):
        _spec(root_fields=(_model_field(), _model_field()))


def test_duplicate_table_paths_are_refused() -> None:
    with pytest.raises(ValueError, match="probe: duplicate configuration table paths"):
        _spec(tables=(_capture_table(), _capture_table(), _settings_table()))


@pytest.mark.parametrize("stages", [("process", "process"), ("process", "")])
def test_repeated_or_unnamed_restart_stages_are_refused(stages: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="probe: restart stages must be unique nonempty names"):
        _spec(restart_stages=stages)


def test_a_restart_enabled_model_may_not_also_claim_no_restart() -> None:
    with pytest.raises(ValueError, match="probe: restart-enabled model cannot claim no restart"):
        _spec(no_restart_reason="This model has no saved restart operation.")


def test_a_root_field_may_not_shadow_a_table_root() -> None:
    shadow = ConfigFieldSpec(
        name="capture",
        purpose="Shadow a registered table root",
        valid="a typed value",
        example="probe",
        cli_dest="capture_selector",
        persistence="intent",
        surfaces=_ROOT_SURFACES,
        virtual=True,
        default="probe",
    )
    with pytest.raises(ValueError, match="probe: root fields collide with tables: capture"):
        _spec(root_fields=(_model_field(), shadow))


def test_a_second_field_may_not_shadow_a_registered_cli_destination() -> None:
    """A new contribution reusing the registered --steps destination is refused."""
    with pytest.raises(ValueError, match="probe: duplicate CLI destinations in config schema"):
        _spec(tables=(_capture_table(), _settings_table(destination="steps")))


def test_a_settings_record_may_not_read_a_different_variable_than_the_registry() -> None:
    """The record declares KINO_PROBE_ALPHA while the registry publishes _BETA."""
    with pytest.raises(ValueError, match=r"probe\.alpha: settings/registry environment drift"):
        _spec(tables=(_capture_table(), _settings_table(environment_key="beta")))


def test_an_artifact_may_not_be_mutable_where_it_is_not_produced() -> None:
    """Availability is the authority; a hand-written mode cannot outvote it."""
    table = _capture_table(trace_produced_at=frozenset({"acquire"}))
    with pytest.raises(
        ValueError,
        match=(
            r"capture\.save_trace: stage process declares 'mutable' "
            r"but this artifact is not produced there"
        ),
    ):
        _spec(tables=(table, _settings_table()))


def test_an_artifact_may_not_be_irrelevant_where_it_is_produced() -> None:
    table = _capture_table(trace_mode="irrelevant")
    with pytest.raises(
        ValueError,
        match=(
            r"capture\.save_trace: stage process declares 'irrelevant' "
            r"but this artifact is produced there"
        ),
    ):
        _spec(tables=(table, _settings_table()))
