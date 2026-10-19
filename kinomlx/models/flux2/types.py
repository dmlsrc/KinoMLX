"""Typed image invocation controls using the shared image-model vocabulary."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from kinomlx.adapters.selection import AdapterSelection
from kinomlx.config_constraints import ChoiceConstraint, IntegerConstraint, NumberSequenceConstraint
from kinomlx.config_records import config_record
from kinomlx.config_rules import validate_capabilities
from kinomlx.image_restart_config import ImageRestartSelectionFields
from kinomlx.io.geometry import MATCH_SOURCE_ASPECT_CHOICES, REFERENCE_FIT_CHOICES
from kinomlx.prompts import PromptInputConstraint, normalize_prompts
from kinomlx.request_rules import IMAGE_PROMPT, IMAGE_SEEDS, SAVED_IMAGE_CONDITIONING
from kinomlx.samplers.config import SigmaScheduleConstraint
from kinomlx.seeds import SEED_MODES, SeedMode, resolve_seeds
from kinomlx.types import DEFAULT_NOISE_BACKEND, NOISE_BACKEND_CHOICES, NoiseBackend

from .adapters.inputs import Flux2AdapterConfig
from .catalog import Flux2Variant
from .geometry_rules import REFERENCE_GEOMETRY
from .request_constraints import CFG_MODE, DENOISE_STEPS, GUIDANCE, OUTPUT_DIMENSION, Flux2CFGMode
from .request_rules import NEGATIVE_ROWS, RECIPE_CAPABILITIES, negative_rows
from .sampling import SamplingRecipe


@config_record
@dataclass(frozen=True)
class Flux2Request:
    config_table: ClassVar[str] = "[generate]"

    prompt: str | tuple[str, ...] | None = field(
        default=None, metadata={"config_constraints": (PromptInputConstraint(),)}
    )
    prompt_unconditional: bool = field(default=False, metadata={"config_rules": (IMAGE_PROMPT,)})
    negative_prompt: str | tuple[str, ...] | None = field(
        default=None,
        metadata={
            "config_constraints": (PromptInputConstraint(),),
            "config_rules": (NEGATIVE_ROWS,),
        },
    )
    width: int | None = field(default=None, metadata={"config_constraints": (OUTPUT_DIMENSION,)})
    height: int | None = field(default=None, metadata={"config_constraints": (OUTPUT_DIMENSION,)})
    seed: int = field(
        default=0,
        metadata={"config_constraints": (IntegerConstraint(minimum=0, maximum=2**64 - 1),)},
    )
    seeds: tuple[int, ...] | None = field(
        default=None,
        metadata={
            "config_constraints": (
                NumberSequenceConstraint(
                    minimum_length=1, item=IntegerConstraint(minimum=0, maximum=2**64 - 1)
                ),
            ),
            "config_rules": (IMAGE_SEEDS,),
        },
    )
    seed_mode: SeedMode = field(
        default=SeedMode.SEQUENTIAL,
        metadata={"config_constraints": (ChoiceConstraint(SEED_MODES),)},
    )
    steps: int | None = field(default=None, metadata={"config_constraints": (DENOISE_STEPS,)})
    guidance: float | None = field(default=None, metadata={"config_constraints": (GUIDANCE,)})
    sigmas: tuple[float, ...] = field(
        default=(), metadata={"config_constraints": (SigmaScheduleConstraint(),)}
    )
    # Klein KV editing extracts the reference K/V once and reuses it; a per-step
    # adapter schedule would read a stale cache, so this recomputes it every step
    # at the cost of one extra reference forward per step. KV variant only.
    recompute_reference_kv: bool = False
    images_per_prompt: int = field(
        default=1, metadata={"config_constraints": (IntegerConstraint(minimum=1),)}
    )
    noise_backend: NoiseBackend = field(
        default=DEFAULT_NOISE_BACKEND,
        metadata={"config_constraints": (ChoiceConstraint(NOISE_BACKEND_CHOICES),)},
    )
    adapters: tuple[Flux2AdapterConfig, ...] = ()
    text_conditioning: Path | None = field(
        default=None, metadata={"config_rules": (SAVED_IMAGE_CONDITIONING,)}
    )
    source_images: tuple[Path, ...] = field(
        default=(), metadata={"config_rules": (REFERENCE_GEOMETRY,)}
    )
    reference_fit: str | None = field(
        default=None, metadata={"config_constraints": (ChoiceConstraint(REFERENCE_FIT_CHOICES),)}
    )
    match_source_aspect: str | None = field(
        default=None,
        metadata={"config_constraints": (ChoiceConstraint(MATCH_SOURCE_ASPECT_CHOICES),)},
    )
    cfg_mode: Flux2CFGMode = field(
        default=Flux2CFGMode.SEQUENTIAL,
        metadata={"config_constraints": (CFG_MODE,), "config_rules": (RECIPE_CAPABILITIES,)},
    )

    def selected_seeds(self) -> tuple[int, ...]:
        """Ordered seeds for each prompt."""
        return resolve_seeds(self.seed, self.seeds, self.images_per_prompt)

    @property
    def image_count(self) -> int:
        return len(self.prompts()) * len(self.selected_seeds())

    def prompts(self) -> tuple[str, ...]:
        if self.prompt_unconditional:
            return self.prompt if isinstance(self.prompt, tuple) else ("",)
        if self.prompt is None:
            return ()
        return normalize_prompts(self.prompt, name="prompt")

    def negatives(self) -> tuple[str, ...]:
        return negative_rows(len(self.prompts()), self.negative_prompt)

    def resolved_adapters(self) -> tuple[AdapterSelection, ...]:
        return tuple(
            adapter.selection(location=f"[generate].adapters[{index}]")
            for index, adapter in enumerate(self.adapters)
        )

    def recipe(self, variant: str) -> SamplingRecipe:
        recipe = SamplingRecipe(
            Flux2Variant(variant),
            self.steps,
            self.guidance,
            self.cfg_mode,
            sigmas=self.sigmas,
            recompute_reference_kv=self.recompute_reference_kv,
        )
        recipe.resolved()
        return recipe

    def validate(self, variant: str) -> None:
        validate_capabilities(self, {"variant": variant}, location="[generate]")


@config_record
@dataclass(frozen=True)
class Flux2RestartConfig(ImageRestartSelectionFields):
    """Select a prior Klein run and its first repeated stage."""
