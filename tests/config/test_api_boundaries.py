"""Public runner parameters have complete typed configuration or runtime ownership."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from types import FunctionType
from typing import TypeVar, get_args, get_type_hints

import pytest

from kinomlx.api_boundaries import (
    APIBoundary,
    ConfigurationOwner,
    RuntimeOwner,
    configuration,
    runtime,
)
from kinomlx.config_constraints import NumberConstraint
from kinomlx.config_records import config_record
from kinomlx.model_registry import config_registry
from kinomlx.models.flux2 import runner as flux2_runner
from kinomlx.models.gmnet import runner as gmnet_runner
from kinomlx.models.gmnet.types import GMNetRequest
from kinomlx.models.krea2 import runner as krea2_runner
from kinomlx.models.krea2.types import Krea2Request
from kinomlx.models.ltx2 import runner as ltx2_runner
from kinomlx.settings import Settings
from tests.config.test_record_schema import Generate, fixture_schema

BOUNDARIES = (
    *flux2_runner.API_BOUNDARIES,
    *gmnet_runner.API_BOUNDARIES,
    *krea2_runner.API_BOUNDARIES,
    *ltx2_runner.API_BOUNDARIES,
)
RUNNERS = (
    flux2_runner.Flux2Runner,
    gmnet_runner.GMNetRunner,
    krea2_runner.Krea2Runner,
    ltx2_runner.LTX2Runner,
)


def test_runner_inventory_includes_every_public_method_and_constructor():
    actual = {
        value
        for runner in RUNNERS
        for name, value in vars(runner).items()
        if isinstance(value, FunctionType) and (name == "__init__" or not name.startswith("_"))
    }
    assert {boundary.operation for boundary in BOUNDARIES} == actual


def test_every_runner_parameter_has_a_real_typed_owner_and_configuration_table():
    registry = config_registry()
    assert len(BOUNDARIES) == 15
    for boundary in BOUNDARIES:
        boundary.validate()
        schema = registry.model(boundary.model)
        tables = {table.path: table for table in schema.tables}
        hints = get_type_hints(boundary.operation)
        for parameter in boundary.parameters:
            assert parameter.name in hints
            owner = parameter.owner
            if isinstance(owner, ConfigurationOwner):
                assert tables[owner.table].record is owner.record
            else:
                assert isinstance(owner, RuntimeOwner)
                assert owner.role
                entry = next(
                    item
                    for item in boundary.describe()["parameters"]
                    if item["name"] == parameter.name
                )
                assert entry["api_only"] is True
                assert entry["serializable"] is False


def test_projected_request_types_link_to_the_registered_generate_or_expand_record():
    projections = {
        "gmnet": GMNetRequest,
        "krea2": Krea2Request,
    }
    for boundary in BOUNDARIES:
        if boundary.model not in projections or _is_run(boundary):
            continue
        for parameter in boundary.parameters:
            if parameter.name != "request":
                continue
            owner = parameter.owner
            assert isinstance(owner, ConfigurationOwner)
            assert owner.projection
            assert get_type_hints(boundary.operation)["request"] is projections[boundary.model]


def _is_run(boundary):
    return getattr(boundary.operation, "__qualname__", "").endswith(".run")


def test_every_run_port_is_the_same_generic_shape_on_every_model():
    """run() hosts any recipe, so its request is an unbounded type parameter."""
    boundaries = [boundary for boundary in BOUNDARIES if _is_run(boundary)]
    assert len(boundaries) == len(RUNNERS)
    for boundary in boundaries:
        declared = {
            parameter.name: parameter.owner
            for parameter in boundary.parameters
            if parameter.name in {"recipe", "request"}
        }
        assert declared.keys() == {"recipe", "request"}
        assert all(isinstance(owner, RuntimeOwner) for owner in declared.values())
        assert declared["request"].role == "recipe-selected request port"
        hints = get_type_hints(boundary.operation)
        assert isinstance(hints["request"], TypeVar)
        assert get_args(hints["recipe"]) == (hints["request"],)


def test_a_registered_table_record_cannot_wear_a_runtime_label():
    def misclassified(request: Generate) -> None:
        pass

    def generic[RequestT](request: RequestT) -> None:
        pass

    schema = fixture_schema()
    port = runtime("request", "recipe-selected request port")
    with pytest.raises(ValueError, match="names the registered table record for generate"):
        replace(schema, api_boundaries=(APIBoundary("fixture", misclassified, (port,)),))
    # An unbounded type parameter names no record, so it stays a port.
    replace(schema, api_boundaries=(APIBoundary("fixture", generic, (port,)),))


def test_new_parameter_and_stale_declaration_fail_before_api_inspection():
    def operation(value: int, callback: Callable[[], None] | None = None) -> None:
        pass

    with pytest.raises(ValueError, match="API parameter ownership drift"):
        APIBoundary("fixture", operation, (runtime("value", "one invocation value"),))
    with pytest.raises(ValueError, match="API parameter ownership drift"):
        APIBoundary(
            "fixture",
            operation,
            (
                runtime("value", "one invocation value"),
                runtime("callback", "completion callback"),
                runtime("obsolete", "removed callback"),
            ),
        )


def test_untyped_and_unowned_runtime_inputs_fail_declaration():
    def untyped(callback):
        return callback

    with pytest.raises(ValueError, match="missing API annotation"):
        APIBoundary("fixture", untyped, (runtime("callback", "completion callback"),))
    with pytest.raises(ValueError, match="requires a role"):
        runtime("callback", " ")


def test_configuration_record_cannot_hide_inside_a_collection_annotation():
    def wrong(settings: list[Settings]) -> None:
        pass

    with pytest.raises(ValueError, match="annotation does not name its configuration owner"):
        APIBoundary("fixture", wrong, (configuration("settings", Settings, "settings"),))


def test_projection_rejects_member_type_and_local_constraint_drift():
    @dataclass(frozen=True)
    class WrongImage:
        image: int

    def wrong_image(request: WrongImage) -> None:
        pass

    gmnet_owner = next(
        parameter
        for parameter in gmnet_runner.API_BOUNDARIES[-1].parameters
        if parameter.name == "request"
    )
    with pytest.raises(ValueError, match="projection semantic drift"):
        APIBoundary("gmnet", wrong_image, (gmnet_owner,))

    @config_record
    @dataclass(frozen=True)
    class Owned:
        gain: float = field(
            default=1.0, metadata={"config_constraints": (NumberConstraint(minimum=0),)}
        )

    @dataclass(frozen=True)
    class Unconstrained:
        gain: float = 1.0

    def wrong_constraint(request: Unconstrained) -> None:
        pass

    with pytest.raises(ValueError, match="projection semantic drift"):
        APIBoundary(
            "fixture",
            wrong_constraint,
            (configuration("request", Owned, "generate", projection=True),),
        )


def test_final_projection_gate_requires_tracked_constructor_presence():
    @config_record
    @dataclass(frozen=True)
    class Owned:
        value: int = 1

    @dataclass(frozen=True)
    class Untracked:
        value: int = 1

    def operation(request: Untracked) -> None:
        pass

    boundary = APIBoundary(
        "fixture", operation, (configuration("request", Owned, "generate", projection=True),)
    )
    with pytest.raises(ValueError, match="does not track constructor presence"):
        boundary.validate_tracked_projections()
    for krea_boundary in krea2_runner.API_BOUNDARIES:
        krea_boundary.validate_tracked_projections()


def test_runtime_callback_uses_public_runner_without_becoming_intent(monkeypatch):
    from kinomlx.config_records import record_intent

    request = Krea2Request("a still")

    def callback(_index, _image):
        pass

    received = []
    result = object()

    def receive(_self, recipe, incoming):
        received.append((recipe, incoming))
        return result

    monkeypatch.setattr(krea2_runner.Krea2Runner, "run", receive)
    runner = object.__new__(krea2_runner.Krea2Runner)
    assert runner.generate(request, on_image=callback) is result
    assert received[0][1] is request
    assert received[0][0].keywords["on_image"] is callback
    assert "on_image" not in record_intent(request)
    entry = next(
        item
        for boundary in krea2_runner.API_BOUNDARIES
        if boundary.operation is krea2_runner.Krea2Runner.generate
        for item in boundary.describe()["parameters"]
        if item["name"] == "on_image"
    )
    assert entry["classification"] == "runtime"
    assert entry["serializable"] is False


def test_model_contribution_validates_api_table_model_and_operation_ownership():
    schema = config_registry().model("flux2")
    boundary = flux2_runner.API_BOUNDARIES[0]
    assert replace(schema, api_boundaries=(boundary,)).api_boundaries == (boundary,)
    with pytest.raises(ValueError, match="duplicate API operation ownership"):
        replace(schema, api_boundaries=(boundary, boundary))
    with pytest.raises(ValueError, match="API boundary belongs to"):
        replace(schema, api_boundaries=(replace(boundary, model="fixture"),))
    parameters = tuple(
        replace(parameter, owner=replace(parameter.owner, table=("missing",)))
        if isinstance(parameter.owner, ConfigurationOwner)
        else parameter
        for parameter in boundary.parameters
    )
    with pytest.raises(ValueError, match="does not match registered table missing"):
        replace(schema, api_boundaries=(replace(boundary, parameters=parameters),))
