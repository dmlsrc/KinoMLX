"""Krea output availability shared by field declarations and restart preparation."""

from dataclasses import dataclass

from kinomlx.config.stages import OutputPolicy, produced_at_stages
from kinomlx.image_restart_config import IMAGE_CONSUMES, IMAGE_PRODUCES, IMAGE_STAGES


@dataclass(frozen=True)
class KreaOutputPolicy(OutputPolicy):
    """One artifact's stage and recipe eligibility."""

    recipe_modes: frozenset[str]

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.recipe_modes or not self.recipe_modes <= {"text", "edit"}:
            raise ValueError("Krea output policy requires supported recipe modes")

    def describe(self) -> dict[str, object]:
        return {**super().describe(), "recipe_modes": sorted(self.recipe_modes)}

    def selected_for_recipe(
        self,
        value: bool | None,
        *,
        save_all: bool,
        stage: str,
        field: str,
        recipe_mode: str,
        explicitly_supplied: bool,
    ) -> bool:
        selected = super().selected(value, save_all=save_all, stage=stage, field=field)
        if recipe_mode not in self.recipe_modes:
            if explicitly_supplied and value is True:
                raise ValueError(
                    "editing cannot save a text-only restart; save final latents instead"
                )
            return False
        return selected


_PRODUCED_AT = produced_at_stages(IMAGE_STAGES, IMAGE_PRODUCES, IMAGE_CONSUMES)
OUTPUT_POLICIES = frozendict(
    {
        "save_initial_noise": KreaOutputPolicy(
            _PRODUCED_AT["initial_noise"], frozenset({"text", "edit"})
        ),
        "save_final_latents": KreaOutputPolicy(
            _PRODUCED_AT["final_latents"], frozenset({"text", "edit"})
        ),
        "save_text_conditioning": KreaOutputPolicy(
            _PRODUCED_AT["text_conditioning"], frozenset({"text"})
        ),
    }
)
