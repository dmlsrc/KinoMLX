"""Resume Klein from a saved stage or final latents."""

from dataclasses import replace

from kinomlx.config_rules import validate_record
from kinomlx.image_restart_config import ImageStage
from kinomlx.memory_policy import memory_managed
from kinomlx.reporting import Reporter

from ..components import Flux2ComponentsProvider
from ..generation import Flux2ImageResult, decode_latents
from ..resources import Flux2Resources
from ..restart import Flux2Restart, PreparedResources, load_final_latents
from ..text_conditioning import Flux2Conditioning
from ..types import Flux2Request
from .text_to_image import Flux2GenerationOutput, generate_flux2


@memory_managed
def restart_flux2(
    request: Flux2Request,
    resources: Flux2Resources | PreparedResources,
    *,
    restart: Flux2Restart,
    reporter: Reporter | None = None,
    conditioning: Flux2Conditioning | None = None,
    components: Flux2ComponentsProvider | None = None,
) -> Flux2GenerationOutput:
    """Resume from a complete stage without source encoders or persisted KV."""
    request.validate(resources.profile.variant.value)
    if len(request.prompts()) > 1 or len(request.selected_seeds()) != 1:
        raise ValueError("Saved-stage restart requires one image")
    validate_record(restart, location="[restart]")
    if restart.stage == ImageStage.DENOISE:
        if request.text_conditioning not in (None, restart.text_conditioning):
            raise ValueError("Request and restart select different conditioning artifacts")
        return generate_flux2(
            replace(request, text_conditioning=restart.text_conditioning),
            resources,
            reporter=reporter,
            conditioning=conditioning,
            components=components,
        )
    assert restart.latents is not None
    latents, width, height = load_final_latents(restart.latents)
    if (request.width is not None and request.width != width) or (
        request.height is not None and request.height != height
    ):
        raise ValueError("Decode restart must preserve the saved output grid")
    pixels = decode_latents(
        resources, latents, width, height, reporter=reporter, components=components
    )
    return Flux2GenerationOutput((Flux2ImageResult(pixels, latents, width, height, ()),))
