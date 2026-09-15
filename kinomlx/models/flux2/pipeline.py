"""Typed Klein generation and saved-station recipes shared by the CLI and runner."""

from collections.abc import Callable
from dataclasses import dataclass, replace

from kinomlx.config_rules import validate_record
from kinomlx.memory_policy import memory_managed
from kinomlx.reporting import Reporter
from kinomlx.seed_execution import SeedModels, execute_seed_stations

from .generation import Flux2ImageResult, decode_latents, image_stations
from .resources import Flux2Resources
from .restart import Flux2Restart, PreparedResources, load_final_latents
from .stations import Flux2Conditioning, load_conditioning
from .types import Flux2Request


@dataclass(frozen=True)
class Flux2GenerationOutput:
    """Materialized images in prompt/variation order, with captured replay stations."""

    images: tuple[Flux2ImageResult, ...]


@memory_managed
def generate_flux2(
    request: Flux2Request,
    resources: Flux2Resources | PreparedResources,
    *,
    reporter: Reporter | None = None,
    conditioning: Flux2Conditioning | None = None,
    on_image: Callable[[int, Flux2ImageResult], None] | None = None,
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
                job = image_stations(
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
                    capture_stations=True,
                    reporter=reporter,
                    models=models,
                )
                jobs.append(job)
                prompt_conditioning = next(job)
        results = []
        for index, result in enumerate(execute_seed_stations(jobs, models, request.seed_mode)):
            if on_image is not None:
                on_image(index, result)
            results.append(result)
        return Flux2GenerationOutput(tuple(results))
    finally:
        for job in jobs:
            job.close()
        models.close()


@memory_managed
def restart_flux2(
    request: Flux2Request,
    resources: Flux2Resources | PreparedResources,
    *,
    restart: Flux2Restart,
    reporter: Reporter | None = None,
    conditioning: Flux2Conditioning | None = None,
) -> Flux2GenerationOutput:
    """Resume from a complete station without source encoders or persisted KV."""
    request.validate(resources.profile.variant.value)
    if len(request.prompts()) > 1 or len(request.selected_seeds()) != 1:
        raise ValueError("Saved-station restart requires one image")
    validate_record(restart, location="[restart]")
    if restart.phase == "denoise":
        if request.text_conditioning not in (None, restart.text_conditioning):
            raise ValueError("Request and restart select different conditioning artifacts")
        return generate_flux2(
            replace(request, text_conditioning=restart.text_conditioning),
            resources,
            reporter=reporter,
            conditioning=conditioning,
        )
    assert restart.latents is not None
    latents, width, height = load_final_latents(restart.latents)
    if (request.width is not None and request.width != width) or (
        request.height is not None and request.height != height
    ):
        raise ValueError("Decode restart must preserve the saved output grid")
    pixels = decode_latents(resources.component("vae"), latents, width, height, reporter=reporter)
    return Flux2GenerationOutput((Flux2ImageResult(pixels, latents, width, height, ()),))
