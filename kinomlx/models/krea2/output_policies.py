"""Krea output availability shared by field declarations and restart preparation."""

from dataclasses import dataclass
from types import MappingProxyType

from kinomlx.config.stations import OutputPolicy


@dataclass(frozen=True)
class KreaOutputPolicy(OutputPolicy):
    """One artifact's station and recipe eligibility."""

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
        station: str,
        field: str,
        recipe_mode: str,
        explicitly_supplied: bool,
    ) -> bool:
        selected = super().selected(value, save_all=save_all, station=station, field=field)
        if recipe_mode not in self.recipe_modes:
            if explicitly_supplied and value is True:
                raise ValueError(
                    "editing cannot save a text-only restart; save final latents instead"
                )
            return False
        return selected


# A restart station re-emits what it consumed and still holds, so a bundle it
# writes can itself be restarted: decode re-emits the final latent it decoded,
# and denoise re-emits the saved text it read back.
OUTPUT_POLICIES = MappingProxyType(
    {
        "save_initial_noise": KreaOutputPolicy(
            frozenset({"generate", "denoise"}), frozenset({"text", "edit"})
        ),
        "save_final_latents": KreaOutputPolicy(
            frozenset({"generate", "denoise", "decode"}), frozenset({"text", "edit"})
        ),
        "save_text_conditioning": KreaOutputPolicy(
            frozenset({"generate", "denoise"}), frozenset({"text"})
        ),
    }
)
