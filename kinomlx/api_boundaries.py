"""Typed ownership of public API parameters, without an execution layer."""

import dataclasses
import inspect
from dataclasses import dataclass
from types import FunctionType, UnionType
from typing import Union, cast, get_args, get_origin, get_type_hints

from kinomlx._config_record_state import tracks_presence
from kinomlx.config_constraints import field_constraints


@dataclass(frozen=True)
class ConfigurationOwner:
    """A parameter supplied through an existing configuration table or its projection."""

    table: tuple[str, ...]
    record: type[object]
    projection: bool = False

    def __post_init__(self) -> None:
        if not self.table or not all(part.isidentifier() for part in self.table):
            raise ValueError("API configuration owner requires a table path")
        if not tracks_presence(self.record):
            raise ValueError("API configuration owner requires a tracked record")


@dataclass(frozen=True)
class RuntimeOwner:
    """An API-only argument that never becomes saved configuration intent."""

    role: str

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise ValueError("API runtime owner requires a role")


@dataclass(frozen=True)
class APIParameter:
    name: str
    owner: ConfigurationOwner | RuntimeOwner

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError("API parameter requires a Python identifier")


def configuration(
    name: str, record: type[object], table: str, *, projection: bool = False
) -> APIParameter:
    """Refer to a registered table without copying its fields or defaults."""
    return APIParameter(name, ConfigurationOwner(tuple(table.split(".")), record, projection))


def runtime(name: str, role: str) -> APIParameter:
    """Classify a typed, nonserializable invocation port."""
    return APIParameter(name, RuntimeOwner(role))


def _optional_member(annotation: object) -> object | None:
    if get_origin(annotation) not in {UnionType, Union}:
        return None
    members = get_args(annotation)
    if len(members) != 2 or type(None) not in members:
        return None
    return cast(object, members[0] if members[1] is type(None) else members[1])


def _owns_record(annotation: object, record: type[object]) -> bool:
    return annotation is record or _optional_member(annotation) is record


def _projection_annotation_matches(owner: object, projection: object) -> bool:
    return owner == projection or _optional_member(owner) == projection


def _tracked_projection(annotation: object) -> bool:
    return isinstance(annotation, type) and tracks_presence(annotation)


@dataclass(frozen=True)
class APIBoundary:
    """Check that an actual callable has one owner for every public parameter."""

    model: str
    operation: object
    parameters: tuple[APIParameter, ...]

    def __post_init__(self) -> None:
        if not self.model.isidentifier():
            raise ValueError("API boundary requires a model")
        self.validate()

    def _function(self) -> FunctionType:
        if not isinstance(self.operation, FunctionType):
            raise TypeError("API boundary requires a function or unbound method")
        return self.operation

    def validate(self) -> None:
        """Fail on missing, stale, untyped or incompatible parameter ownership."""
        operation = self._function()
        signature = inspect.signature(operation)
        actual = tuple(name for name in signature.parameters if name not in {"self", "cls"})
        declared = tuple(parameter.name for parameter in self.parameters)
        if actual != declared:
            raise ValueError(
                f"{operation.__qualname__}: API parameter ownership drift: "
                f"signature {actual}, declaration {declared}"
            )
        annotations = get_type_hints(operation)
        for parameter in self.parameters:
            annotation = annotations.get(parameter.name)
            if annotation is None:
                raise ValueError(
                    f"{operation.__qualname__}.{parameter.name}: missing API annotation"
                )
            owner = parameter.owner
            if isinstance(owner, ConfigurationOwner):
                if owner.projection:
                    if not dataclasses.is_dataclass(annotation) or not dataclasses.is_dataclass(
                        owner.record
                    ):
                        raise ValueError(
                            f"{operation.__qualname__}.{parameter.name}: "
                            "configuration projection must be a dataclass"
                        )
                    owned_fields = {
                        field.name: field
                        for field in dataclasses.fields(owner.record)
                        if field.init
                    }
                    projected_fields = {
                        field.name: field for field in dataclasses.fields(annotation) if field.init
                    }
                    if projected_fields.keys() != owned_fields.keys():
                        raise ValueError(
                            f"{operation.__qualname__}.{parameter.name}: "
                            "configuration projection field drift"
                        )
                    owner_hints = get_type_hints(owner.record)
                    projected_hints = get_type_hints(annotation)
                    for name, field in projected_fields.items():
                        if not _projection_annotation_matches(
                            owner_hints[name], projected_hints[name]
                        ) or field_constraints(field) != field_constraints(owned_fields[name]):
                            raise ValueError(
                                f"{operation.__qualname__}.{parameter.name}.{name}: "
                                "configuration projection semantic drift"
                            )
                elif not _owns_record(annotation, owner.record):
                    raise ValueError(
                        f"{operation.__qualname__}.{parameter.name}: "
                        "annotation does not name its configuration owner"
                    )

    def runtime_annotations(self) -> dict[str, object]:
        """Resolve the annotation behind every runtime-declared parameter.

        A runtime port is nonserializable by declaration. The registry compares
        these annotations with its own table records, which is where a record
        that really is configuration can be caught wearing a runtime label.
        """
        self.validate()
        annotations = get_type_hints(self._function())
        return {
            parameter.name: annotations[parameter.name]
            for parameter in self.parameters
            if isinstance(parameter.owner, RuntimeOwner)
        }

    def validate_tracked_projections(self) -> None:
        """Require that public projections preserve constructor presence."""
        self.validate()
        annotations = get_type_hints(self._function())
        for parameter in self.parameters:
            owner = parameter.owner
            if isinstance(owner, ConfigurationOwner) and owner.projection:
                projected = annotations[parameter.name]
                if not _tracked_projection(projected):
                    raise ValueError(
                        f"{self._function().__qualname__}.{parameter.name}: "
                        "configuration projection does not track constructor presence"
                    )

    def describe(self) -> dict[str, object]:
        """Expose actual annotations and explicit ownership for generated inspection."""
        self.validate()
        operation = self._function()
        annotations = get_type_hints(operation)
        entries: list[dict[str, object]] = []
        for parameter in self.parameters:
            owner = parameter.owner
            entry: dict[str, object] = {
                "name": parameter.name,
                "type": str(annotations[parameter.name]),
            }
            if isinstance(owner, ConfigurationOwner):
                projection_presence = (
                    ("tracked" if _tracked_projection(annotations[parameter.name]) else "untracked")
                    if owner.projection
                    else None
                )
                entry.update(
                    classification="configuration",
                    table=".".join(owner.table),
                    record=f"{owner.record.__module__}.{owner.record.__qualname__}",
                    projection=owner.projection,
                    projection_presence=projection_presence,
                )
            else:
                entry.update(
                    classification="runtime",
                    role=owner.role,
                    api_only=True,
                    serializable=False,
                )
            entries.append(entry)
        return {
            "model": self.model,
            "operation": f"{operation.__module__}.{operation.__qualname__}",
            "parameters": entries,
        }


__all__ = [
    "APIBoundary",
    "APIParameter",
    "ConfigurationOwner",
    "RuntimeOwner",
    "configuration",
    "runtime",
]
