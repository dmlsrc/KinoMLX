"""Local component selection for the Klein resource graph."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from kinomlx.config_constraints import ChoiceConstraint
from kinomlx.config_records import config_record
from kinomlx.settings import EnvironmentSettings

from .catalog import Flux2Variant


@config_record
@dataclass(frozen=True)
class Flux2Settings(EnvironmentSettings):
    environment_prefix: ClassVar[str] = "KINO_FLUX2_"
    config_table: ClassVar[str] = "[model_settings]"

    variant: str = field(
        default="distilled",
        metadata={
            "config_constraints": (ChoiceConstraint(tuple(item.value for item in Flux2Variant)),)
        },
    )
    transformer_dtype: str = field(
        default="bfloat16",
        metadata={"config_constraints": (ChoiceConstraint(("bfloat16", "float16")),)},
    )
    source_path: Path | None = None
    tokenizer_path: Path | None = None
    text_encoder_path: Path | None = None
    transformer_path: Path | None = None
    vae_path: Path | None = None

    @property
    def component_overrides(self) -> dict[str, Path]:
        return {
            name: path
            for name, path in (
                ("tokenizer", self.tokenizer_path),
                ("text_encoder", self.text_encoder_path),
                ("transformer", self.transformer_path),
                ("vae", self.vae_path),
            )
            if path is not None
        }
