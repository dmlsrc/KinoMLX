"""Field-owned pure validation of complete configuration records."""

import copy
import dataclasses
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from kinomlx.config_record_state import supplied_fields
from kinomlx.config_values import ConfigError


@dataclass(frozen=True)
class RuleInputs(Mapping[str, object]):
    """Read-only declared values and explicitly declared supplied-field observations."""

    values_by_name: Mapping[str, object]
    presence: Mapping[str, bool]
    context: Mapping[str, object]

    def __getitem__(self, name: str) -> object:
        return self.values_by_name[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values_by_name)

    def __len__(self) -> int:
        return len(self.values_by_name)


@dataclass(frozen=True)
class RecordRule:
    """A named semantic check with an explicit read set and no writes or source effects."""

    name: str
    reads: frozenset[str]
    run: Callable[[RuleInputs], None]
    presence_reads: frozenset[str] = frozenset()
    phase: Literal["semantic", "capability"] = "semantic"
    context_reads: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.name.isidentifier() or not isinstance(self.reads, frozenset) or not self.reads:
            raise ValueError("record rule requires a name and nonempty frozen read set")
        if any(not isinstance(name, str) or not name.isidentifier() for name in self.reads):
            raise ValueError("record rule reads must name fields")
        if not isinstance(self.presence_reads, frozenset) or not self.presence_reads <= self.reads:
            raise ValueError("presence reads must be a frozen subset of the declared value reads")
        if self.phase not in {"semantic", "capability"}:
            raise ValueError("record rule phase must be semantic or capability")
        if not isinstance(self.context_reads, frozenset) or any(
            not isinstance(name, str) or not name.isidentifier() for name in self.context_reads
        ):
            raise ValueError("context reads must be frozen observation names")
        if bool(self.context_reads) != (self.phase == "capability"):
            raise ValueError(
                "capability rules require context reads; semantic rules read no context"
            )
        if not callable(self.run):
            raise TypeError("record rule requires a callable validator")

    def describe(self) -> dict[str, object]:
        result: dict[str, object] = {
            "rule": self.name,
            "phase": self.phase,
            "reads": sorted(self.reads),
            "effects": [],
        }
        if self.presence_reads:
            result["presence_reads"] = sorted(self.presence_reads)
        if self.context_reads:
            result["context_reads"] = sorted(self.context_reads)
        return result


def record_rules(record: type[object]) -> tuple[RecordRule, ...]:
    """Compile field-owned rules, rejecting unknown reads and conflicting identities."""
    if not dataclasses.is_dataclass(record):
        raise TypeError("record rules require a dataclass owner")
    fields = {field.name: field for field in dataclasses.fields(record) if field.init}
    rules: dict[str, RecordRule] = {}
    for field in fields.values():
        declared: object = field.metadata.get("config_rules", ())
        if not isinstance(declared, tuple) or any(
            not isinstance(rule, RecordRule) for rule in declared
        ):
            raise TypeError(
                f"{field.name}: config_rules must be a tuple of RecordRule declarations"
            )
        for rule in declared:
            if field.name not in rule.reads or not rule.reads <= fields.keys():
                raise ValueError(
                    f"{field.name}: {rule.name} must read its owner and only declared fields"
                )
            if rule.name in rules and rules[rule.name] != rule:
                raise ValueError(f"{record.__name__}: conflicting semantic rule {rule.name}")
            rules[rule.name] = rule
    return tuple(rules.values())


def _validate_record(
    value: object, location: str, *, phase: str, context: Mapping[str, object]
) -> None:
    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        raise TypeError("record validation requires a dataclass instance")
    for rule in record_rules(type(value)):
        if rule.phase != phase:
            continue
        missing = rule.context_reads - context.keys()
        if missing:
            raise ValueError(
                f"{location} rule {rule.name}: "
                f"missing required observations: {', '.join(sorted(missing))}"
            )
        observations = {name: copy.deepcopy(context[name]) for name in rule.context_reads}
        inputs = {name: copy.deepcopy(getattr(value, name)) for name in rule.reads}
        presence: dict[str, bool] = {}
        if rule.presence_reads:
            supplied = supplied_fields(value)
            if supplied is None:
                raise TypeError(
                    f"{location} rule {rule.name}: declared presence reads require a tracked record"
                )
            presence = {name: name in supplied for name in rule.presence_reads}
        try:
            rule.run(
                RuleInputs(
                    MappingProxyType(inputs),
                    MappingProxyType(presence),
                    MappingProxyType(observations),
                )
            )
        except ConfigError:
            # A rule that validated a nested record raises with that member's own
            # location already attached; prefixing it again would repeat a table.
            raise
        except ValueError as exc:
            # The rule name is an internal identity; a reader gets the TOML
            # location the violated value was written at instead.
            raise ConfigError(f"{location}: {exc}") from exc
    for field in dataclasses.fields(value):
        if field.init:
            _validate_nested(
                getattr(value, field.name), f"{location}.{field.name}", phase=phase, context=context
            )


def _validate_nested(
    value: object, location: str, *, phase: str, context: Mapping[str, object]
) -> None:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        _validate_record(value, location, phase=phase, context=context)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _validate_nested(item, f"{location}[{index}]", phase=phase, context=context)


def validate_record(value: object, *, location: str) -> None:
    """Validate a surviving record and its nested members, never partial source layers.

    ``location`` is the TOML table the record was written at, such as
    ``[generate]``; nested members render as ``[generate].image`` and collection
    items as ``[generate].adapters[0]``. It is required so that no violation can
    fall back to a Python class name the reader has no way to act on.
    """
    _validate_record(value, location, phase="semantic", context={})


def validate_capabilities(value: object, context: Mapping[str, object], *, location: str) -> None:
    """Validate final inputs and their declared prepared-resource observations."""
    validate_record(value, location=location)
    _validate_record(value, location, phase="capability", context=context)
