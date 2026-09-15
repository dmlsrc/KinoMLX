"""Typed CLI bindings owned by configuration field declarations."""

import argparse
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Literal, TypeAliasType, Union, get_args, get_origin


def argument_value[T](parse: Callable[[str], T], name: str) -> Callable[[str], T]:
    """Preserve a domain codec's useful error at the argparse boundary."""

    def convert(text: str) -> T:
        try:
            return parse(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    convert.__name__ = name
    return convert


def _scalar_types(annotation: object, *, repeated: bool) -> set[type[object]]:
    if isinstance(annotation, TypeAliasType):
        return _scalar_types(annotation.__value__, repeated=repeated)
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Annotated:
        return _scalar_types(arguments[0], repeated=repeated)
    if origin is Literal:
        return {type(value) for value in arguments}
    if origin in (types.UnionType, Union):
        return set().union(
            *(
                _scalar_types(choice, repeated=repeated)
                for choice in arguments
                if choice is not type(None)
            )
        )
    if origin in (tuple, list, Sequence) and repeated:
        if origin is tuple and (len(arguments) != 2 or arguments[1] is not Ellipsis):
            raise TypeError(f"CLI binding needs a codec for {annotation!r}")
        return _scalar_types(arguments[0], repeated=False)
    if annotation in (str, int, float, Path) and isinstance(annotation, type):
        return {annotation}
    raise TypeError(f"CLI binding needs a codec for {annotation!r}")


@dataclass(frozen=True)
class CliBinding:
    """One field's public spellings and action; value types come from its record.

    ``parse`` is an explicit transport codec for non-scalar syntax or virtual
    fields. It is not a second declaration of the field's semantic type.
    """

    group: str
    flags: tuple[str, ...] = ()
    action: (
        Literal["store", "append", "boolean", "store_true", "store_false", "paired_boolean"]
        | type[argparse.Action]
    ) = "store"
    parse: Callable[[str], object] | None = None
    nargs: int | Literal["?", "+", "*"] | None = None
    const: object = None
    negative_flags: tuple[str, ...] = ()
    metavar: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.action, str):
            if self.action not in (
                "store",
                "append",
                "boolean",
                "store_true",
                "store_false",
                "paired_boolean",
            ):
                raise ValueError(f"unsupported CLI action {self.action!r}")
        elif not isinstance(self.action, type) or not issubclass(self.action, argparse.Action):
            raise TypeError("CLI action must be a supported action or argparse.Action class")
        if (
            self.nargs is not None
            and self.nargs not in ("?", "+", "*")
            and (type(self.nargs) is not int or self.nargs < 1)
        ):
            raise ValueError("CLI arity must be positive or ?, +, *")
        if self.negative_flags and self.action != "paired_boolean":
            raise ValueError("negative CLI flags require a paired boolean action")
        if self.negative_flags:
            all_flags = self.flags + self.negative_flags
            if len(all_flags) != len(set(all_flags)) or any(
                not flag.startswith("--")
                or len(flag) <= 2
                or "=" in flag
                or any(c.isspace() for c in flag)
                for flag in self.negative_flags
            ):
                raise ValueError("negative CLI flags must be unique long option spellings")
        if not self.group.strip():
            raise ValueError("CLI group cannot be empty")
        if len(self.flags) != len(set(self.flags)) or any(
            not flag.startswith("--")
            or len(flag) <= 2
            or "=" in flag
            or any(c.isspace() for c in flag)
            for flag in self.flags
        ):
            raise ValueError("CLI flags must be unique long option spellings")
        if self.action in ("boolean", "store_true", "store_false", "paired_boolean") and (
            self.parse is not None or self.nargs is not None or self.const is not None
        ):
            raise ValueError("boolean CLI actions cannot declare a value codec, arity, or const")
        if self.const is not None and self.nargs != "?":
            raise ValueError("CLI const requires optional value arity")

    def option_strings(self, destination: str) -> tuple[str, ...]:
        """Mechanical spelling unless the declaration owns an explicit alias."""
        return self.flags or ("--" + destination.replace("_", "-"),)

    def value_parser(self, annotation: object) -> Callable[[str], object] | None:
        """Infer primitive input from typed records; complex syntax needs a codec."""
        if self.parse is not None:
            return self.parse
        if self.action not in ("store", "append"):
            return None
        scalars = _scalar_types(annotation, repeated=self.action == "append")
        if len(scalars) != 1:
            raise TypeError(f"CLI binding needs a codec for {annotation!r}")
        scalar = scalars.pop()
        if scalar is str:
            return str
        if scalar is int:
            return int
        if scalar is float:
            return float
        if scalar is Path:
            return Path
        raise TypeError(f"unsupported CLI scalar type {scalar!r}")

    def add_argument(
        self,
        parser: argparse._ActionsContainer,
        *,
        destination: str,
        annotation: object,
        help: str | None,
        choices: tuple[str, ...] = (),
    ) -> argparse.Action:
        """Construct a sparse argparse input, never applying a record default."""
        if self.action == "paired_boolean":
            raise TypeError("paired boolean bindings require add_arguments()")
        action = argparse.BooleanOptionalAction if self.action == "boolean" else self.action
        if self.action in ("boolean", "store_true", "store_false", "paired_boolean"):
            return parser.add_argument(
                *self.option_strings(destination),
                dest=destination,
                action=action,
                default=None,
                help=help,
            )
        value_parser = self.value_parser(annotation)
        if value_parser is None:
            return parser.add_argument(
                *self.option_strings(destination),
                dest=destination,
                action=action,
                nargs=self.nargs,
                const=self.const,
                default=None,
                choices=choices or None,
                metavar=self.metavar,
                help=help,
            )
        return parser.add_argument(
            *self.option_strings(destination),
            dest=destination,
            action=action,
            type=value_parser,
            nargs=self.nargs,
            const=self.const,
            default=None,
            choices=choices or None,
            metavar=self.metavar,
            help=help,
        )

    def validate_action(self, action: argparse.Action, *, annotation: object) -> None:
        """Detect downstream changes to an already generated parser action."""
        expected_type = (
            {
                "store": argparse._StoreAction,
                "append": argparse._AppendAction,
                "boolean": argparse.BooleanOptionalAction,
                "store_true": argparse._StoreTrueAction,
                "store_false": argparse._StoreFalseAction,
            }[self.action]
            if isinstance(self.action, str)
            else self.action
        )
        flags = self.option_strings(action.dest)
        if self.action == "boolean":
            flags = tuple(option for flag in flags for option in (flag, "--no-" + flag[2:]))
        expected_nargs = (
            0
            if self.action in ("boolean", "store_true", "store_false", "paired_boolean")
            else self.nargs
        )
        expected_const = (
            (self.action == "store_true")
            if self.action in ("store_true", "store_false")
            else self.const
        )
        if (
            type(action) is not expected_type
            or tuple(action.option_strings) != flags
            or action.type is not self.value_parser(annotation)
            or action.nargs != expected_nargs
            or action.const != expected_const
            or action.metavar != self.metavar
        ):
            raise ValueError(f"CLI binding drift for {action.dest}")

    def _paired_bindings(self, destination: str) -> tuple[CliBinding, CliBinding]:
        negative = self.negative_flags or tuple(
            "--no-" + flag.removeprefix("--") for flag in self.option_strings(destination)
        )
        return (
            replace(self, action="store_true", negative_flags=()),
            replace(self, action="store_false", flags=negative, negative_flags=()),
        )

    def add_arguments(
        self,
        parser: argparse._ActionsContainer,
        *,
        destination: str,
        annotation: object,
        help: str | None,
        choices: tuple[str, ...] = (),
        negative_help: str | None = None,
    ) -> tuple[argparse.Action, ...]:
        """Construct a declared value action or paired boolean actions with distinct help."""
        if self.action != "paired_boolean":
            return (
                self.add_argument(
                    parser,
                    destination=destination,
                    annotation=annotation,
                    help=help,
                    choices=choices,
                ),
            )
        positive, negative = self._paired_bindings(destination)
        return (
            positive.add_argument(
                parser, destination=destination, annotation=annotation, help=help
            ),
            negative.add_argument(
                parser,
                destination=destination,
                annotation=annotation,
                help=negative_help if negative_help is not None else help,
            ),
        )

    def validate_actions(
        self, actions: list[argparse.Action], *, annotation: object, destination: str
    ) -> None:
        """Verify action cardinality as well as every field-owned binding property."""
        bindings = (
            self._paired_bindings(destination) if self.action == "paired_boolean" else (self,)
        )
        if len(actions) != len(bindings):
            raise ValueError(f"CLI binding drift for {destination}: wrong action count")
        for binding, action in zip(bindings, actions, strict=True):
            binding.validate_action(action, annotation=annotation)
