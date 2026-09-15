"""Data-only GMNet settings record.

The environment-shaped facts of ``kinomlx --model gmnet`` travel through
the shared declarative bridge with the standard precedence: dataclass
defaults < environment < TOML ``[model_settings]`` table < generated CLI
flags < ``--set model_settings.<key>=<value>``.

Per-invocation request arguments (the input image, output selection,
``--force``) are deliberately not settings; they follow the same split
the generation CLI uses.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from kinomlx.config_constraints import ChoiceConstraint
from kinomlx.config_records import config_record
from kinomlx.models.gmnet.catalog import GMNetVariant
from kinomlx.settings import EnvironmentSettings


@config_record
@dataclass(frozen=True)
class GMNetSettings(EnvironmentSettings):
    """Which published GMNet behavior to run and where its weights live."""

    environment_prefix: ClassVar[str] = "KINO_GMNET_"
    config_table: ClassVar[str] = "[model_settings]"

    variant: str = field(
        default=GMNetVariant.REALWORLD.value,
        metadata={
            "config_constraints": (ChoiceConstraint(tuple(item.value for item in GMNetVariant)),)
        },
    )
    weights_path: Path | None = None


__all__ = ["GMNetSettings"]
