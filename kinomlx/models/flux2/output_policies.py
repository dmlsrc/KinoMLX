"""Klein output availability shared by schemas and direct output planning."""

from kinomlx.config.stages import OutputPolicy, produced_at_stages
from kinomlx.image_restart_config import IMAGE_CONSUMES, IMAGE_PRODUCES, IMAGE_STAGES

_PRODUCED_AT = produced_at_stages(IMAGE_STAGES, IMAGE_PRODUCES, IMAGE_CONSUMES)
OUTPUT_POLICIES = frozendict(
    {
        "save_initial_noise": OutputPolicy(_PRODUCED_AT["initial_noise"]),
        "save_final_latents": OutputPolicy(_PRODUCED_AT["final_latents"]),
        "save_text_conditioning": OutputPolicy(_PRODUCED_AT["text_conditioning"]),
    }
)
