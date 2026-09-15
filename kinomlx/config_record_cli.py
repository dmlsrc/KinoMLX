"""Argparse actions that build sparse records from local item and member tokens."""

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import overload

_RECORD_SCOPE_PREFIX = "_kinomlx_cli_records_"

type RecordCoercer = Callable[[object, str], dict[str, object]]


class RecordArgument(argparse.Action):
    """Apply one declared token to a CLI-local ordered collection."""

    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        *,
        member: str,
        starts_record: bool,
        coerce: RecordCoercer,
        location: str,
        item_flag: str,
        type: Callable[[str], object] | None = None,
        choices: tuple[str, ...] | None = None,
        metavar: str | None = None,
        help: str | None = None,
        default: object = None,
    ) -> None:
        super().__init__(
            option_strings,
            dest,
            type=type,
            choices=choices,
            metavar=metavar,
            help=help,
            default=default,
        )
        self.member = member
        self.starts_record = starts_record
        self.coerce = coerce
        self.location = location
        self.item_flag = item_flag

    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,  # noqa: ARG002
    ) -> None:
        marker = _RECORD_SCOPE_PREFIX + self.dest
        started = getattr(namespace, marker, False)
        existing = getattr(namespace, self.dest, None)
        records = list(existing) if started and isinstance(existing, list) else []
        if not self.starts_record and not records:
            raise argparse.ArgumentError(self, f"requires a preceding {self.item_flag} item")
        index = len(records) if self.starts_record else len(records) - 1
        raw = (
            values if self.starts_record and isinstance(values, Mapping) else {self.member: values}
        )
        try:
            members = self.coerce(raw, f"{self.location}[{index}]")
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentError(self, str(exc)) from exc
        if self.starts_record:
            records.append(members)
        else:
            current = records[-1]
            if not isinstance(current, dict):
                raise argparse.ArgumentError(self, "current CLI item is not a record")
            duplicates = members.keys() & current.keys()
            if duplicates:
                raise argparse.ArgumentError(
                    self, "duplicate member assignment: " + ", ".join(sorted(duplicates))
                )
            records[-1] = {**current, **members}
        setattr(namespace, self.dest, records)
        setattr(namespace, marker, True)


class ConfigArgumentParser(argparse.ArgumentParser):
    """Give each parse a fresh local-item scope, including reused namespaces."""

    @overload
    def parse_known_args(
        self, args: Iterable[str] | None = None, namespace: None = None
    ) -> tuple[argparse.Namespace, list[str]]: ...

    @overload
    def parse_known_args[NamespaceT](
        self, args: Iterable[str] | None, namespace: NamespaceT
    ) -> tuple[NamespaceT, list[str]]: ...

    @overload
    def parse_known_args[NamespaceT](
        self, *, namespace: NamespaceT
    ) -> tuple[NamespaceT, list[str]]: ...

    def parse_known_args(
        self, args: Iterable[str] | None = None, namespace: object = None
    ) -> tuple[object, list[str]]:
        if namespace is not None:
            for name in tuple(vars(namespace)):
                if name.startswith(_RECORD_SCOPE_PREFIX):
                    delattr(namespace, name)
        return super().parse_known_args(args, namespace)
