"""Sequential native component lifetimes for Klein image recipes."""

import gc
from collections.abc import Generator
from dataclasses import dataclass, replace
from pathlib import Path

import mlx.core as mx

from kinomlx.adapters.receipt import AdapterReceipt
from kinomlx.adapters.selection import AdapterSelection
from kinomlx.io.image import image_size
from kinomlx.modeling.dtypes import resolve_bound_dtype
from kinomlx.reporting import NullReporter, Reporter, reported_phase
from kinomlx.samplers.noise import create_normal_noise_stream
from kinomlx.seed_execution import SeedModels, execute_seed_stages
from kinomlx.seeds import SeedMode
from kinomlx.types import DEFAULT_NOISE_BACKEND, NoiseBackend

from .adapters.loading import prepare_flux2_adapters
from .components import Flux2Components, Flux2ComponentsProvider, Flux2SeedComponents
from .image_processing import (
    ReferenceGeometry,
    load_reference_pixels,
    output_size,
    pack_tokens,
    reference_geometry,
    spatial_ids,
    unpack_tokens,
)
from .resources import Flux2Resources
from .restart import PreparedResources
from .sampling import SamplingRecipe, denoise
from .source_loading import (
    component_weight_files,
    load_text_encoder,
    load_tokenizer,
    load_vae,
)
from .text import prepare_text
from .text_conditioning import Flux2Conditioning


@dataclass(frozen=True)
class Flux2ImageResult:
    pixels: mx.array
    latents: mx.array
    width: int
    height: int
    references: tuple[ReferenceGeometry, ...]
    adapter_receipts: tuple[AdapterReceipt, ...] = ()
    conditioning: Flux2Conditioning | None = None
    initial_noise: mx.array | None = None
    sigmas: mx.array | None = None


def _cleanup_models() -> None:
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


def generate_image(
    resources: Flux2Resources | PreparedResources,
    prompt: str,
    *,
    recipe: SamplingRecipe | None = None,
    seed: int = 0,
    noise_backend: NoiseBackend = DEFAULT_NOISE_BACKEND,
    images: tuple[Path, ...] = (),
    width: int | None = None,
    height: int | None = None,
    match_source_aspect: str | None = None,
    reference_fit: str | None = None,
    negative_prompt: str = "",
    negative_embeddings: mx.array | None = None,
    noise: mx.array | None = None,
    reporter: Reporter | None = None,
    adapters: tuple[AdapterSelection, ...] = (),
    conditioning: Flux2Conditioning | None = None,
    capture_stages: bool = False,
    components: Flux2ComponentsProvider | None = None,
) -> Flux2ImageResult:
    """Generate one image using normal stage lifetimes."""
    models = SeedModels()
    job = image_stages(
        resources=resources,
        prompt=prompt,
        recipe=recipe,
        seed=seed,
        noise_backend=noise_backend,
        images=images,
        width=width,
        height=height,
        match_source_aspect=match_source_aspect,
        reference_fit=reference_fit,
        negative_prompt=negative_prompt,
        negative_embeddings=negative_embeddings,
        noise=noise,
        reporter=reporter,
        adapters=adapters,
        conditioning=conditioning,
        capture_stages=capture_stages,
        models=models,
        components=components,
    )
    try:
        next(job)
        return next(execute_seed_stages([job], models, SeedMode.SEQUENTIAL))
    finally:
        job.close()
        models.close()


def image_stages(
    resources: Flux2Resources | PreparedResources,
    prompt: str,
    *,
    recipe: SamplingRecipe | None = None,
    seed: int = 0,
    noise_backend: NoiseBackend = DEFAULT_NOISE_BACKEND,
    images: tuple[Path, ...] = (),
    width: int | None = None,
    height: int | None = None,
    match_source_aspect: str | None = None,
    reference_fit: str | None = None,
    negative_prompt: str = "",
    negative_embeddings: mx.array | None = None,
    noise: mx.array | None = None,
    reporter: Reporter | None = None,
    adapters: tuple[AdapterSelection, ...] = (),
    conditioning: Flux2Conditioning | None = None,
    capture_stages: bool = False,
    models: SeedModels,
    components: Flux2ComponentsProvider | None = None,
) -> Generator[Flux2Conditioning | None, None, Flux2ImageResult]:
    """Generate with prepared resources; return pixels without choosing an output path.

    Explicit noise is packed [1, H/16 * W/16, 128]. Negative embeddings replace
    encoding the negative prompt and must be [1, 512, 12288]. This function
    releases each model before loading the next component.
    """
    selected = recipe or SamplingRecipe(resources.profile.variant)
    dtype = resolve_bound_dtype(resources.component("transformer").dtype_policy)
    if selected.variant != resources.profile.variant:
        raise ValueError("Sampling recipe must match the prepared transformer variant")
    _, guidance = selected.resolved()
    if guidance == 1 and (negative_prompt or negative_embeddings is not None):
        raise ValueError("Negative conditioning requires active base CFG")
    if negative_prompt and negative_embeddings is not None:
        raise ValueError("Choose a negative prompt or explicit negative embeddings")
    if negative_embeddings is not None and negative_embeddings.shape != (1, 512, 12288):
        raise ValueError("Explicit negative embeddings require [1, 512, 12288]")
    if conditioning is not None:
        conditioning.validate(selected)
        if images or negative_embeddings is not None or match_source_aspect or reference_fit:
            raise ValueError("Saved conditioning replaces source images and text encoding controls")
        if prompt != conditioning.prompt or negative_prompt != conditioning.negative_prompt:
            raise ValueError("Saved conditioning must match the requested prompt context")
        width = conditioning.width if width is None else width
        height = conditioning.height if height is None else height
        if conditioning.geometries and (width, height) != (conditioning.width, conditioning.height):
            raise ValueError("Edit replay must preserve the saved output geometry")
    if not images and reference_fit is not None:
        raise ValueError("reference_fit requires a reference image")
    sizes = tuple(image_size(path) for path in images)
    width, height = output_size(
        sizes[0] if sizes else None,
        reference_count=len(images),
        width=width,
        height=height,
        match_source_aspect=match_source_aspect,
    )
    geometries = tuple(
        reference_geometry(
            size, (width, height), reference_count=len(images), reference_fit=reference_fit
        )
        for size in sizes
    )
    if conditioning is not None:
        geometries = conditioning.geometries
    shape = (1, width // 16 * (height // 16), 128)
    if noise is not None and noise.shape != shape:
        raise ValueError(f"Explicit noise must have shape {shape}")
    if noise is not None:
        noise = noise.astype(dtype)
    if negative_embeddings is not None:
        negative_embeddings = negative_embeddings.astype(dtype)
    for supplied in (noise, negative_embeddings):
        if supplied is not None and not bool(mx.all(mx.isfinite(supplied)).item()):
            raise ValueError("Explicit noise and negative embeddings must be finite")
    stream = create_normal_noise_stream(seed, backend=noise_backend)
    progress = reporter or NullReporter()
    provider = components if components is not None else Flux2Components(reporter=progress)
    ports = Flux2SeedComponents(provider, models)
    with reported_phase(progress, "inspect_adapters"):
        adapter_plan = prepare_flux2_adapters(adapters)
    if conditioning is None:
        if not isinstance(resources, Flux2Resources):
            raise ValueError("Replay with prepared resources requires saved conditioning")
        conditioning = _encode_conditioning(
            resources,
            prompt,
            negative_prompt,
            negative_embeddings,
            guidance,
            dtype,
            images,
            geometries,
            width,
            height,
            progress,
        )
    else:
        conditioning = replace(conditioning.astype(dtype), width=width, height=height)
    conditioning.validate(selected)
    yield conditioning
    current = (
        noise.astype(dtype)
        if noise is not None
        else pack_tokens(stream.normal((1, 128, height // 16, width // 16), dtype))
    )

    transformer_lease = ports.transformer(resources, adapter_plan=adapter_plan)
    transformer = transformer_lease.value
    adapter_receipts = transformer.adapter_receipts
    latents = denoise(
        transformer,
        current,
        conditioning.text,
        spatial_ids(height // 16, width // 16),
        conditioning.text_ids,
        recipe=selected,
        negative=conditioning.negative,
        reporter=progress,
        references=conditioning.references,
        reference_ids=conditioning.reference_ids,
    )
    del transformer
    transformer_lease.close()
    saved_conditioning = conditioning if capture_stages else None
    del conditioning
    gc.collect()
    mx.clear_cache()
    yield None
    pixels = decode_latents(
        resources, latents, width, height, reporter=progress, models=models, components=provider
    )
    return Flux2ImageResult(
        pixels,
        latents,
        width,
        height,
        geometries,
        adapter_receipts,
        saved_conditioning,
        current if capture_stages else None,
        selected.sigma_array(current.shape[1]) if capture_stages else None,
    )


def decode_latents(
    resources: Flux2Resources | PreparedResources,
    latents: mx.array,
    width: int,
    height: int,
    *,
    reporter: Reporter | None = None,
    models: SeedModels | None = None,
    components: Flux2ComponentsProvider | None = None,
) -> mx.array:
    """Decode a final stage using only the selected VAE component."""
    if min(width, height) < 64 or width % 16 or height % 16:
        raise ValueError("Klein decode dimensions require multiples of 16 >=64")
    if latents.shape != (1, width // 16 * (height // 16), 128):
        raise ValueError("Klein final latents do not match the output dimensions")
    if latents.dtype not in {mx.float16, mx.bfloat16, mx.float32} or not bool(
        mx.all(mx.isfinite(latents)).item()
    ):
        raise ValueError("Klein final latents must be finite floating-point values")
    progress = reporter or NullReporter()
    provider = components if components is not None else Flux2Components(reporter=progress)
    ports = provider if models is None else Flux2SeedComponents(provider, models)
    vae = ports.image_decoder(resources).value
    with reported_phase(progress, "decode"):
        decoded = vae.decode_normalized(
            unpack_tokens(latents.astype(mx.bfloat16), height // 16, width // 16)
        )
        pixels = mx.clip(decoded[0].transpose(1, 2, 0).astype(mx.float32) / 2 + 0.5, 0, 1)
        mx.eval(pixels)
        if not bool(mx.all(mx.isfinite(pixels)).item()):
            raise FloatingPointError("Klein decode produced nonfinite pixels")
    return pixels


def _encode_conditioning(
    resources: Flux2Resources,
    prompt: str,
    negative_prompt: str,
    negative_embeddings: mx.array | None,
    guidance: float,
    dtype: mx.Dtype,
    images: tuple[Path, ...],
    geometries: tuple[ReferenceGeometry, ...],
    width: int,
    height: int,
    progress: Reporter,
) -> Flux2Conditioning:
    tokenizer = load_tokenizer(resources.component("tokenizer"))
    prompts = (
        (prompt, negative_prompt) if guidance > 1 and negative_embeddings is None else (prompt,)
    )
    batch = prepare_text(tokenizer, prompts)
    model = load_text_encoder(
        component_weight_files(resources.component("text_encoder")), reporter=progress
    )
    with reported_phase(progress, "encode_text"):
        text = model(batch.input_ids, batch.attention_mask).astype(dtype)
        mx.eval(text)
    negative = text[1:2] if len(prompts) == 2 else negative_embeddings
    text = text[:1]
    del model
    gc.collect()
    mx.clear_cache()
    refs, ids = [], []
    if images:
        vae = load_vae(
            component_weight_files(resources.component("vae")), mode="encode", reporter=progress
        )
        with reported_phase(progress, "encode_references"):
            for index, (path, geometry) in enumerate(zip(images, geometries, strict=True)):
                pixels = load_reference_pixels(path, geometry).astype(mx.bfloat16)
                latent = vae.encode_normalized(pixels)
                tokens = pack_tokens(latent).astype(dtype)
                mx.eval(tokens)
                refs.append(tokens)
                ids.append(
                    spatial_ids(
                        latent.shape[2],
                        latent.shape[3],
                        reference_index=index,
                        offset_hw=geometry.offset_hw,
                    )
                )
        del vae, pixels, latent, tokens
        gc.collect()
        mx.clear_cache()
    return Flux2Conditioning(
        text,
        batch.text_ids[:1],
        negative,
        mx.concatenate(refs, axis=1) if refs else None,
        mx.concatenate(ids, axis=1) if ids else None,
        geometries,
        prompt,
        negative_prompt,
        width,
        height,
    )
