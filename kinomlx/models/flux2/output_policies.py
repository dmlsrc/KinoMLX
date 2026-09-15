"""Klein output availability shared by schemas and direct output planning."""

from types import MappingProxyType

from kinomlx.config.stations import OutputPolicy

OUTPUT_POLICIES = MappingProxyType(
    {
        "save_initial_noise": OutputPolicy(frozenset({"generate", "denoise"})),
        "save_final_latents": OutputPolicy(frozenset({"generate", "denoise", "decode"})),
        "save_text_conditioning": OutputPolicy(frozenset({"generate", "denoise"})),
    }
)
