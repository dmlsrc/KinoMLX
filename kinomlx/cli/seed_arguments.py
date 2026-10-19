"""One CLI spelling for scalar and ordered seed selections."""

import argparse

from kinomlx.config import CliBinding, ConfigFieldSpec
from kinomlx.config.layer_rules import SEED_SELECTION
from kinomlx.config_surfaces import CONFIG_SURFACES
from kinomlx.seeds import SEED_MODES, resolve_seeds


class SeedAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,  # noqa: ARG002
    ) -> None:
        if not isinstance(values, list):
            raise argparse.ArgumentError(self, "expected one or more seeds")
        if (
            getattr(namespace, "seed", None) is not None
            or getattr(namespace, "seeds", None) is not None
        ):
            raise argparse.ArgumentError(self, "provide --seed once with all selected seeds")
        if any(not isinstance(value, str) for value in values):
            raise argparse.ArgumentError(self, "expected integer seed values")
        try:
            seeds = tuple(int(part) for value in values for part in value.split(","))
            resolve_seeds(0, seeds)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentError(self, str(exc)) from exc
        if len(seeds) == 1:
            namespace.seed = seeds[0]
        else:
            namespace.seeds = seeds


def seed_fields(*, image: bool) -> tuple[ConfigFieldSpec, ...]:
    """Declare the mutually exclusive scalar and ordered seed spellings together.

    Both carry ``drop_when_none`` so that a higher-precedence layer selecting
    one spelling removes the other instead of merging a null into the request.
    """
    fields: tuple[ConfigFieldSpec, ...] = (
        ConfigFieldSpec(
            persistence="intent",
            name="seed",
            layer_transforms=(SEED_SELECTION,),
            purpose=(
                "Set one seed, or the first of consecutive per-image seeds"
                if image
                else "Set the generation seed"
            ),
            valid="an unsigned 64-bit integer; exclusive with seeds",
            example=42,
            cli_dest="seed",
            cli=CliBinding("Generation", action=SeedAction, nargs="+"),
            stringify_large_int=True,
            drop_when_none=True,
            surfaces=CONFIG_SURFACES,
        ),
        ConfigFieldSpec(
            persistence="intent",
            name="seeds",
            purpose="Select an ordered seed list, preserving duplicates",
            valid="a nonempty array of unsigned 64-bit integers; exclusive with seed",
            example=(42, 123, 999),
            cli_dest="seeds",
            cli=CliBinding("Generation", action=SeedAction, nargs="+"),
            stringify_large_int=True,
            drop_when_none=True,
            surfaces=CONFIG_SURFACES,
        ),
    )
    if image:
        fields += (
            ConfigFieldSpec(
                persistence="intent",
                name="seed_mode",
                purpose="Control model lifetimes across seeds",
                valid="",
                example="sequential",
                cli_dest="seed_mode",
                cli=CliBinding("Generation"),
                choices=SEED_MODES,
                choice_help=(
                    ("sequential", "Complete each seed with normal stage lifetimes"),
                    ("stage", "Process seeds sequentially inside each loaded stage"),
                    ("resident", "Complete each seed while retaining transformer and VAE"),
                ),
                surfaces=CONFIG_SURFACES,
            ),
        )
    return fields
