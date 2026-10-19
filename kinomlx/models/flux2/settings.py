"""Local component selection for the Klein resource graph."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from kinomlx.config_constraints import ChoiceConstraint
from kinomlx.config_records import config_record
from kinomlx.modeling.dtypes import FloatDType
from kinomlx.settings import EnvironmentSettings

from .catalog import Flux2Variant

# Klein binds a subset of the shared vocabulary; float32 transformer weights
# are not supported, so the narrowing is a constraint rather than a new type.
TRANSFORMER_DTYPE_CHOICES = (FloatDType.BFLOAT16.value, FloatDType.FLOAT16.value)
VAE_DTYPE_CHOICES = (FloatDType.BFLOAT16.value, FloatDType.FLOAT32.value)


@config_record
@dataclass(frozen=True)
class Flux2Settings(EnvironmentSettings):
    environment_prefix: ClassVar[str] = "KINO_FLUX2_"
    config_table: ClassVar[str] = "[model_settings]"

    variant: str = field(
        default=Flux2Variant.DISTILLED,
        metadata={
            "config_constraints": (ChoiceConstraint(tuple(item.value for item in Flux2Variant)),)
        },
    )
    transformer_dtype: FloatDType = field(
        default=FloatDType.BFLOAT16,
        metadata={"config_constraints": (ChoiceConstraint(TRANSFORMER_DTYPE_CHOICES),)},
    )
    # The decoder binds BF16 or FP32; FP16 is not a supported Klein VAE path.
    vae_dtype: FloatDType = field(
        default=FloatDType.BFLOAT16,
        metadata={"config_constraints": (ChoiceConstraint(VAE_DTYPE_CHOICES),)},
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
