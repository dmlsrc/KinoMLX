"""Data-only GMNet settings record; importing this module does not load MLX.

The environment-shaped facts of ``kinomlx --model gmnet`` travel through
the shared declarative bridge with the standard precedence: dataclass
defaults < environment < TOML ``[model_settings]`` table < generated CLI
flags < ``--set model_settings.<key>=<value>``.

Per-invocation request arguments (the input image, output selection,
``--force``) are deliberately not settings; they follow the same split
the generation CLI uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kinomlx.models.gmnet.catalog import GMNetVariant
from kinomlx.settings import EnvironmentSettings


@dataclass(frozen=True)
class GMNetSettings(EnvironmentSettings):
    """Which published GMNet behavior to run and where its weights live."""

    variant: str = field(
        default=GMNetVariant.REALWORLD.value,
        metadata={"env": "{{KINO_GMNET_VARIANT}}"},
    )
    weights_path: Path | None = field(
        default=None,
        metadata={"env": "{{KINO_GMNET_WEIGHTS}}"},
    )

    def validate(self) -> None:
        """Validate the record after full precedence resolution."""
        allowed = tuple(item.value for item in GMNetVariant)
        if self.variant not in allowed:
            raise ValueError(f"variant must be one of {', '.join(allowed)}, got {self.variant!r}")


__all__ = ["GMNetSettings"]
