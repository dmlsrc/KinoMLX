"""Typed Krea 2 text-to-image requests and invocation records."""

from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import ClassVar

from kinomlx.adapters.selection import AdapterSelection
from kinomlx.config_constraints import (
    ChoiceConstraint,
    IntegerConstraint,
    LengthConstraint,
    NumberConstraint,
    NumberSequenceConstraint,
)
from kinomlx.config_records import config_record, construct_record, supplied_fields
from kinomlx.io.geometry import MATCH_SOURCE_ASPECT_CHOICES, REFERENCE_FIT_CHOICES
from kinomlx.prompts import PromptInputConstraint, normalize_prompts
from kinomlx.request_rules import IMAGE_PROMPT, IMAGE_SEEDS
from kinomlx.samplers.config import SigmaScheduleConstraint
from kinomlx.seeds import SEED_MODES, SeedMode, resolve_seeds
from kinomlx.types import (
    DEFAULT_NOISE_BACKEND,
    NOISE_BACKEND_CHOICES,
    VAE_TILING_MODE_CHOICES,
    NoiseBackend,
    VAETilingMode,
)

from .adapters.inputs import Krea2AdapterConfig
from .adapters.sda_profile import SDA_DEFAULT_SIGMA_WINDOW, SDAProfile
from .adapters.turbo_profile import TURBO_4STEP_RECIPE, TurboFourStepProfile
from .request_constraints import (
    NonzeroStrengthConstraint,
    ProfileExclusionsConstraint,
    ProfileWindowConstraint,
    ReferenceRegionsConstraint,
)
from .request_rules import NEGATIVE_CFG, PROFILES, RECIPE_MODE, REFERENCES, VAE_TILING

KREA2_DEFAULT_NOISE_BACKEND: NoiseBackend = DEFAULT_NOISE_BACKEND

type Krea2PromptInput = str | Sequence[str]


@dataclass(frozen=True)
class _Krea2InputFields:
    """One field owner for configuration admission and resolved generation requests."""

    prompt: Krea2PromptInput | None = field(
        default=None,
        metadata={
            "config_constraints": (PromptInputConstraint(),),
            "config_rules": (IMAGE_PROMPT,),
        },
    )
    prompt_unconditional: bool = False
    negative_prompt: Krea2PromptInput | None = field(
        default=None,
        metadata={
            "config_constraints": (PromptInputConstraint(),),
            "config_rules": (NEGATIVE_CFG,),
        },
    )
    system_prompt: str | None = field(
        default=None, metadata={"config_constraints": (LengthConstraint(maximum=65_536),)}
    )
    width: int = field(
        default=1024, metadata={"config_constraints": (IntegerConstraint(minimum=1),)}
    )
    height: int = field(
        default=1024, metadata={"config_constraints": (IntegerConstraint(minimum=1),)}
    )
    seed: int = field(
        default=0,
        metadata={
            "config_constraints": (IntegerConstraint(minimum=0, maximum=2**64 - 1),),
            "config_rules": (IMAGE_SEEDS,),
        },
    )
    seeds: tuple[int, ...] | None = field(
        default=None,
        metadata={
            "config_constraints": (
                NumberSequenceConstraint(
                    minimum_length=1, item=IntegerConstraint(minimum=0, maximum=2**64 - 1)
                ),
            )
        },
    )
    seed_mode: SeedMode = field(
        default=SeedMode.SEQUENTIAL,
        metadata={"config_constraints": (ChoiceConstraint(SEED_MODES),)},
    )
    turbo_4step: bool = field(default=False, metadata={"config_rules": (PROFILES,)})
    turbo_4step_exclusions: str = field(
        default="", metadata={"config_constraints": (ProfileExclusionsConstraint(),)}
    )
    turbo_4step_strength: float = field(
        default=1.0,
        metadata={"config_constraints": (NumberConstraint(minimum=0, exclusive_minimum=True),)},
    )
    turbo_4step_step_strengths: tuple[float, ...] = field(
        default=(), metadata={"config_constraints": (NumberSequenceConstraint(minimum_length=0),)}
    )
    turbo_4step_sigma_window: str = field(
        default="", metadata={"config_constraints": (ProfileWindowConstraint(),)}
    )
    turbo_sda: bool = False
    turbo_sda_exclusions: str = field(
        default="", metadata={"config_constraints": (ProfileExclusionsConstraint(),)}
    )
    turbo_sda_strength: float = field(
        default=1.0, metadata={"config_constraints": (NumberConstraint(),)}
    )
    turbo_sda_step_strengths: tuple[float, ...] = field(
        default=(), metadata={"config_constraints": (NumberSequenceConstraint(minimum_length=0),)}
    )
    turbo_sda_sigma_window: str = field(
        default="", metadata={"config_constraints": (ProfileWindowConstraint(),)}
    )
    steps: int | None = field(
        default=None, metadata={"config_constraints": (IntegerConstraint(minimum=1),)}
    )
    guidance: float | None = field(
        default=None, metadata={"config_constraints": (NumberConstraint(minimum=0),)}
    )
    mu: float | None = field(default=None, metadata={"config_constraints": (NumberConstraint(),)})
    sigmas: tuple[float, ...] = field(
        default=(), metadata={"config_constraints": (SigmaScheduleConstraint(),)}
    )
    images_per_prompt: int = field(
        default=1, metadata={"config_constraints": (IntegerConstraint(minimum=1),)}
    )
    noise_backend: NoiseBackend = field(
        default=KREA2_DEFAULT_NOISE_BACKEND,
        metadata={"config_constraints": (ChoiceConstraint(NOISE_BACKEND_CHOICES),)},
    )
    vae_tiling: VAETilingMode = field(
        default=VAETilingMode.AUTO,
        metadata={"config_constraints": (ChoiceConstraint(VAE_TILING_MODE_CHOICES),)},
    )
    vae_tile_size: int = field(
        default=256, metadata={"config_constraints": (IntegerConstraint(minimum=1),)}
    )
    vae_tile_stride: int = field(
        default=192,
        metadata={
            "config_constraints": (IntegerConstraint(minimum=1),),
            "config_rules": (VAE_TILING,),
        },
    )
    adapters: tuple[Krea2AdapterConfig, ...] = ()
    text_conditioning: Path | None = None
    source_images: tuple[Path, ...] = field(
        default=(), metadata={"config_rules": (REFERENCES, RECIPE_MODE)}
    )
    edit_adapter_path: Path | None = None
    edit_adapter_strength: float = field(
        default=1.0, metadata={"config_constraints": (NonzeroStrengthConstraint(),)}
    )
    edit_adapter_exclusions: str = field(
        default="", metadata={"config_constraints": (ProfileExclusionsConstraint(),)}
    )
    reference_fit: str = field(
        default="fit", metadata={"config_constraints": (ChoiceConstraint(REFERENCE_FIT_CHOICES),)}
    )
    grounding_system_prompt: str = field(
        default="", metadata={"config_constraints": (LengthConstraint(maximum=65_536),)}
    )
    reference_boost_mask: Path | None = None
    reference_boost_regions: tuple[str, ...] = field(
        default=(), metadata={"config_constraints": (ReferenceRegionsConstraint(),)}
    )
    grounding_px: int = field(
        default=768, metadata={"config_constraints": (IntegerConstraint(minimum=0),)}
    )
    reference_strength: float = field(
        default=1.0, metadata={"config_constraints": (NumberConstraint(minimum=0),)}
    )
    scene_reference_strength: float = field(
        default=1.0, metadata={"config_constraints": (NumberConstraint(minimum=0),)}
    )
    match_source_aspect: str | None = field(
        default=None,
        metadata={"config_constraints": (ChoiceConstraint(MATCH_SOURCE_ASPECT_CHOICES),)},
    )


@config_record
@dataclass(frozen=True)
class Krea2Request(_Krea2InputFields):
    """Generate one or more stills with one Krea Raw or Turbo resource set."""

    config_table: ClassVar[str] = "[generate]"

    _resolved_prompts: tuple[str, ...] = field(init=False, repr=False)
    _resolved_negative_prompts: tuple[str, ...] | None = field(init=False, repr=False)

    def selected_seeds(self) -> tuple[int, ...]:
        """Ordered seeds for each prompt."""
        return resolve_seeds(self.seed, self.seeds, self.images_per_prompt)

    def __post_init__(self) -> None:
        prompts = (
            ()
            if self.prompt is None
            else normalize_prompts(self.prompt, name="prompt", allow_empty=True)
        )
        if self.prompt_unconditional and self.prompt is None:
            prompts = ("",)
        negative = (
            None
            if self.negative_prompt is None
            else normalize_prompts(self.negative_prompt, name="negative_prompt", allow_empty=True)
        )
        if self.prompt is not None:
            object.__setattr__(self, "prompt", prompts)
        if self.negative_prompt is not None:
            object.__setattr__(self, "negative_prompt", negative)
        object.__setattr__(self, "grounding_system_prompt", self.grounding_system_prompt.strip())
        object.__setattr__(self, "_resolved_prompts", prompts)
        object.__setattr__(self, "_resolved_negative_prompts", negative)

    def four_step_profile(self) -> TurboFourStepProfile:
        """The distilled four-step profile's switches, as its own module defines them."""
        return TurboFourStepProfile(
            enabled=self.turbo_4step,
            strength=self.turbo_4step_strength,
            exclusions=self.turbo_4step_exclusions,
            step_strengths=tuple(self.turbo_4step_step_strengths),
            sigma_window=self.turbo_4step_sigma_window,
        )

    def sda_profile(self) -> SDAProfile:
        """The diversity profile's switches, defaulting its gate to the trained band."""
        return SDAProfile(
            enabled=self.turbo_sda,
            strength=self.turbo_sda_strength,
            exclusions=self.turbo_sda_exclusions,
            step_strengths=tuple(self.turbo_sda_step_strengths),
            sigma_window=(
                self.turbo_sda_sigma_window
                if self.turbo_sda_sigma_window or self.turbo_sda_step_strengths
                else SDA_DEFAULT_SIGMA_WINDOW
            ),
        )

    def profile_deviations(self) -> tuple[str, ...]:
        """Report recipe values the four-step profile declares that this run changed.

        The profile supplies its distilled recipe when a field is unset, so an
        explicit value is the user deliberately leaving what the adapter was
        distilled for. That is worth announcing, not refusing.
        """
        if not self.turbo_4step:
            return ()
        changed = [
            f"{name} {getattr(self, name)!r} instead of {expected!r}"
            for name, expected in TURBO_4STEP_RECIPE.items()
            if getattr(self, name) is not None and getattr(self, name) != expected
        ]
        if self.sigmas:
            changed.append("an explicit sigma schedule instead of the distilled one")
        return tuple(changed)

    def resolved_adapters(self) -> tuple[AdapterSelection, ...]:
        """Resolve each surviving user record without broadcasting controls."""
        return tuple(
            adapter.selection(location=f"[generate].adapters[{index}]")
            for index, adapter in enumerate(self.adapters)
        )

    @property
    def prompts(self) -> tuple[str, ...]:
        return self._resolved_prompts

    @property
    def negative_prompts(self) -> tuple[str, ...] | None:
        return self._resolved_negative_prompts

    @property
    def image_count(self) -> int:
        return len(self.prompts) * len(self.selected_seeds())

    def expanded_prompts(self) -> tuple[str, ...]:
        return tuple(
            prompt for prompt in self.prompts for _index in range(len(self.selected_seeds()))
        )

    def expanded_negative_prompts(self) -> tuple[str, ...] | None:
        negative = self.negative_prompts
        if negative is None:
            return None
        if (
            self.negative_prompt is None
            and self.text_conditioning is not None
            and len(negative) == self.image_count
        ):
            return negative
        if len(negative) == 1:
            negative = negative * len(self.prompts)
        if len(negative) != len(self.prompts):
            raise ValueError("negative_prompt must have one row or match the prompt batch")
        return tuple(prompt for prompt in negative for _index in range(len(self.selected_seeds())))


@config_record
@dataclass(frozen=True)
class Krea2GenerateConfig(_Krea2InputFields):
    """TOML/CLI representation of one optional Krea generation request."""

    def request(self) -> Krea2Request | None:
        if self.prompt is None and self.text_conditioning is None and not self.prompt_unconditional:
            return None
        presence = supplied_fields(self)
        assert presence is not None
        return construct_record(
            Krea2Request,
            {item.name: getattr(self, item.name) for item in fields(self)},
            presence,
        )


__all__ = [
    "KREA2_DEFAULT_NOISE_BACKEND",
    "Krea2GenerateConfig",
    "Krea2PromptInput",
    "Krea2Request",
]
