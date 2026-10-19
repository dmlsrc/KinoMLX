"""Sequential Klein text-to-image recipe shared by the CLI and runner."""

from collections.abc import Callable
from dataclasses import dataclass, replace

from kinomlx.memory_policy import memory_managed
from kinomlx.reporting import Reporter
from kinomlx.seed_execution import SeedModels, execute_seed_stages

from ..components import Flux2ComponentsProvider
from ..generation import Flux2ImageResult, image_stages
from ..resources import Flux2Resources
from ..restart import PreparedResources
from ..text_conditioning import Flux2Conditioning, load_conditioning
from ..types import Flux2Request


@dataclass(frozen=True)
class Flux2GenerationOutput:
    """Materialized images in prompt/variation order, with captured replay stages."""

    images: tuple[Flux2ImageResult, ...]


@memory_managed
def generate_flux2(
    request: Flux2Request,
    resources: Flux2Resources | PreparedResources,
    *,
    reporter: Reporter | None = None,
    conditioning: Flux2Conditioning | None = None,
    on_image: Callable[[int, Flux2ImageResult], None] | None = None,
    components: Flux2ComponentsProvider | None = None,
) -> Flux2GenerationOutput:
    """Generate a typed request using sequential per-image component lifetimes."""
    variant = resources.profile.variant.value
    request.validate(variant)
    recipe = request.recipe(variant)
    if conditioning is None and request.text_conditioning is not None:
        conditioning = load_conditioning(request.text_conditioning, recipe)
    if conditioning is not None:
        if request.source_images or request.reference_fit or request.match_source_aspect:
            raise ValueError("Saved conditioning replaces source images and geometry controls")
        if request.prompts() and request.prompts() != (conditioning.prompt,):
            raise ValueError("Saved conditioning prompt does not match the generation request")
        if request.negative_prompt is not None and replace(
            request,
            prompt=conditioning.prompt,
            prompt_unconditional=request.prompt_unconditional or conditioning.prompt == "",
        ).negatives() != (conditioning.negative_prompt,):
            raise ValueError("Saved negative conditioning does not match the generation request")
        request = replace(
            request,
            prompt=conditioning.prompt,
            prompt_unconditional=request.prompt_unconditional or conditioning.prompt == "",
            negative_prompt=conditioning.negative_prompt,
        )
    if not request.prompts():
        raise ValueError("Provide a prompt or saved conditioning")
    adapters = request.resolved_adapters()
    models = SeedModels()
    jobs = []
    try:
        for prompt, negative in zip(request.prompts(), request.negatives(), strict=True):
            prompt_conditioning = conditioning
            for seed in request.selected_seeds():
                job = image_stages(
                    resources,
                    prompt,
                    recipe=recipe,
                    seed=seed,
                    noise_backend=request.noise_backend,
                    images=request.source_images if prompt_conditioning is None else (),
                    width=request.width
                    if prompt_conditioning is None
                    else prompt_conditioning.width,
                    height=request.height
                    if prompt_conditioning is None
                    else prompt_conditioning.height,
                    match_source_aspect=request.match_source_aspect
                    if prompt_conditioning is None
                    else None,
                    reference_fit=request.reference_fit if prompt_conditioning is None else None,
                    negative_prompt=negative,
                    adapters=adapters,
                    conditioning=prompt_conditioning,
                    capture_stages=True,
                    reporter=reporter,
                    models=models,
                    components=components,
                )
                jobs.append(job)
                prompt_conditioning = next(job)
        results = []
        for index, result in enumerate(execute_seed_stages(jobs, models, request.seed_mode)):
            if on_image is not None:
                on_image(index, result)
            results.append(result)
        return Flux2GenerationOutput(tuple(results))
    finally:
        for job in jobs:
            job.close()
        models.close()
