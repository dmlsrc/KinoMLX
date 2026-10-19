"""Invocation control declarations shared by bootstrap, model parsers and inspection."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from kinomlx.config_bindings import CliAction, CliBinding
from kinomlx.config_surfaces import MODEL_SURFACES, UTILITY_SURFACES, SurfacePolicy


@dataclass(frozen=True)
class InvocationControl:
    """A command input with explicit transport scope and no saved-value projection."""

    name: str
    annotation: object
    binding: CliBinding
    purpose: str
    phase: Literal["selection", "utility"]
    default: object = None
    requires_any: tuple[str, ...] = ()
    configuration_target: tuple[str, ...] | None = None
    surfaces: SurfacePolicy = UTILITY_SURFACES

    def __post_init__(self) -> None:
        if not self.name.isidentifier() or not self.purpose.strip():
            raise ValueError("invocation control requires a name and purpose")
        if self.phase not in {"selection", "utility"}:
            raise ValueError("invocation control requires a selection or utility phase")
        if not isinstance(self.surfaces, SurfacePolicy) or not self.surfaces.supports("cli"):
            raise ValueError("invocation controls require a CLI surface policy")
        self.binding.value_parser(self.annotation)

    def add(
        self,
        parser: argparse.ArgumentParser,
        *,
        choices: tuple[str, ...] = (),
        help: str | None = None,
    ) -> None:
        for action in self.binding.add_arguments(
            parser,
            destination=self.name,
            annotation=self.annotation,
            help=self.purpose if help is None else help,
            choices=choices,
        ):
            action.default = self.default

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "owner": f"{__name__}.INVOCATION_CONTROLS",
            "flags": list(self.binding.option_strings(self.name)),
            "purpose": self.purpose,
            "phase": self.phase,
            "requires_any": list(self.requires_any),
            "configuration_target": None
            if self.configuration_target is None
            else ".".join(self.configuration_target),
            "surface_classification": self.surfaces.describe(),
        }


MODEL_CONTROL = InvocationControl(
    "model",
    str,
    CliBinding("Invocation", flags=("--model",)),
    "Model family.",
    "selection",
    configuration_target=("model",),
    surfaces=MODEL_SURFACES,
)
INVOCATION_CONTROLS = (
    MODEL_CONTROL,
    InvocationControl(
        "config", Path, CliBinding("Invocation"), "TOML invocation file.", "selection"
    ),
    InvocationControl(
        "set_overrides",
        tuple[str, ...],
        CliBinding("Invocation", flags=("--set",), action=CliAction.APPEND, metavar="KEY=VALUE"),
        "Apply a TOML-typed key override after config and ordinary CLI flags.",
        "selection",
    ),
    InvocationControl(
        "print_config",
        bool,
        CliBinding("Invocation", action=CliAction.STORE_TRUE),
        "Print canonical input TOML with commented reference defaults and exit.",
        "utility",
        False,
    ),
    InvocationControl(
        "save_config",
        Path,
        CliBinding("Invocation", metavar="PATH"),
        "Save canonical input TOML with commented reference defaults and exit; existing paths are refused.",
        "utility",
    ),
    InvocationControl(
        "only_non_defaults",
        bool,
        CliBinding("Invocation", action=CliAction.STORE_TRUE),
        "With --print-config or --save-config, omit the commented reference defaults. "
        "Executable inputs retain explicit values, environment selections and clearing.",
        "utility",
        False,
        requires_any=("print_config", "save_config"),
    ),
)


def validate_invocation_controls(parser: argparse.ArgumentParser) -> None:
    """Utility destinations, actions, help and defaults must match their declarations."""
    for control in INVOCATION_CONTROLS:
        if control is MODEL_CONTROL:
            continue  # Model choices and model-specific help are checked by the model contribution.
        actions = [action for action in parser._actions if action.dest == control.name]
        if len(actions) != 1:
            raise ValueError(f"invocation control drift for {control.name}: expected one action")
        action = actions[0]
        control.binding.validate_action(action, annotation=control.annotation)
        if action.help != control.purpose or action.default != control.default or action.required:
            raise ValueError(f"invocation control drift for {control.name}: help/default/required")


def validate_control_declarations(controls: tuple[InvocationControl, ...]) -> None:
    """Reject duplicate controls, duplicate flags and unresolved dependencies before exposure."""
    names = {control.name for control in controls}
    if len(names) != len(controls):
        raise ValueError("duplicate invocation control names")
    flags = [*control.binding.option_strings(control.name) for control in controls]
    if len(flags) != len(set(flags)):
        raise ValueError("duplicate invocation control flags")
    for control in controls:
        unknown = set(control.requires_any) - names
        if unknown:
            raise ValueError(f"{control.name}: unknown invocation dependencies {sorted(unknown)}")


def validate_control_inputs(options: object) -> None:
    """Apply declared invocation relationships at the shared command utility boundary."""
    controls = {control.name: control for control in INVOCATION_CONTROLS}
    for control in INVOCATION_CONTROLS:
        if (
            control.requires_any
            and getattr(options, control.name)
            and not any(getattr(options, name) for name in control.requires_any)
        ):
            flag = control.binding.option_strings(control.name)[0]
            alternatives = " or ".join(
                controls[name].binding.option_strings(name)[0] for name in control.requires_any
            )
            raise ValueError(f"{flag} requires {alternatives}")


validate_control_declarations(INVOCATION_CONTROLS)
