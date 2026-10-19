"""The resource-only operation every model family offers, declared once.

A dry run resolves and validates every source a generation would need, reports
it, and exits without loading tensors. That is one behavior with four
implementations of the same wiring: a field on an ``[operation]`` table, a
refusal when the same invocation also asks for work, and a command branch that
prepares, names a preparation fault, and picks the console or the machine
surface. Only the report itself is per-family, because each family's sources
differ; the wiring around it does not.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from kinomlx._json import JsonObject
from kinomlx.cli.presentation import render_error, render_model_error
from kinomlx.config import CliBinding, ConfigError, ConfigFieldSpec, ConfigTableSpec
from kinomlx.config.stages import StageMode, StagePolicy
from kinomlx.config_bindings import CliAction
from kinomlx.config_records import config_record
from kinomlx.config_surfaces import CONFIG_SURFACES
from kinomlx.errors import KinoMLXError
from kinomlx.ui import configure_machine_output

RESOURCE_ONLY_STAGE_REASON = (
    "A saved-stage restart executes a generation stage, so resource-only dry-run stays false."
)


@config_record
@dataclass(frozen=True)
class Operation:
    """What one invocation does besides generate."""

    dry_run: bool = False


class ResourceReport(Protocol):
    """A prepared inventory that can describe itself without loading tensors."""

    def describe(self) -> JsonObject: ...


def operation_table(*, cli_group: str, stages: tuple[str, ...]) -> ConfigTableSpec:
    """Declare the shared operation table, locked at each of the model's stages."""
    return ConfigTableSpec(
        path=("operation",),
        cli_generated=True,
        cli_group=cli_group,
        record=Operation,
        stages=stages,
        stage_record=Operation if stages else None,
        fields=(
            ConfigFieldSpec(
                persistence="intent",
                name="dry_run",
                purpose="Describe validated local resources",
                valid="true or false",
                example=True,
                cli_dest="dry_run",
                cli=CliBinding(cli_group, action=CliAction.STORE_TRUE),
                surfaces=CONFIG_SURFACES,
                stage_policies=tuple(
                    StagePolicy(stage, StageMode.LOCKED, reason=RESOURCE_ONLY_STAGE_REASON)
                    for stage in stages
                ),
            ),
        ),
    )


def refuse_resource_only_work(*, dry_run: bool, requested: bool, work: str) -> None:
    """Refuse a resource-only operation that the same invocation also asks to act on."""
    if dry_run and requested:
        raise ConfigError(f"Resource-only operations cannot be combined with {work}")


def report_resources[ResourcesT: ResourceReport](
    model: str,
    prepare: Callable[[], ResourcesT],
    *,
    log: Callable[[ResourcesT], None],
    json_output: bool,
) -> int:
    """Resolve every source the run would need, report it, and load none of it.

    Preparation reaches the user's own configuration, so a contradiction it
    only discovers against the checkpoint stays a configuration error rather
    than reading like a fault in the model.
    """
    try:
        resources = prepare()
    except ConfigError as exc:
        return render_error(f"config error: {exc}", json_output=json_output)
    except (
        ArithmeticError,
        KinoMLXError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        return render_model_error(model, "preparation", exc, json_output=json_output)
    if not json_output:
        log(resources)
        return 0
    configure_machine_output(f"kinomlx.models.{model}.stdout").info(
        "%s", json.dumps(resources.describe(), indent=2)
    )
    return 0


__all__ = [
    "RESOURCE_ONLY_STAGE_REASON",
    "Operation",
    "ResourceReport",
    "operation_table",
    "refuse_resource_only_work",
    "report_resources",
]
