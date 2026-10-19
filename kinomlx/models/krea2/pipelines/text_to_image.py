"""Sequential Raw/Turbo Krea 2 text-to-image recipe."""

import gc
import hashlib
import logging
from collections.abc import Callable, Generator
from dataclasses import dataclass, field, replace
from pathlib import Path

import mlx.core as mx

from kinomlx.adapters.selection import AdapterSelection, require_adapter_schedule_steps
from kinomlx.config_records import derive_config
from kinomlx.config_rules import validate_capabilities, validate_record
from kinomlx.memory_policy import memory_managed
from kinomlx.modeling.dtypes import resolve_bound_dtype
from kinomlx.modeling.memory import capture_decode_memory_budget
from kinomlx.reporting import NullReporter, Reporter, reported_phase
from kinomlx.samplers.sigmas import sigma_metadata
from kinomlx.seed_execution import SeedModels, execute_seed_stages

from ..adapters.loading import Krea2AdapterPlan, prepare_krea2_adapters
from ..adapters.profiles import resolve_profile_adapters
from ..adapters.sda_profile import sda_pairing_warning
from ..components import Krea2Components, Krea2ComponentsProvider, Krea2SeedComponents
from ..request_rules import admit_recipe_mode
from ..request_sources import resolve_request_sources
from ..resource_types import Krea2Resources
from ..sampling import (
    Krea2Checkpoint,
    Krea2Geometry,
    Krea2Recipe,
    create_initial_noise,
    denoise_krea2,
    pack_latents,
    prepare_branch,
    resolve_geometry,
    resolve_recipe,
    timestep_schedule,
    unpack_latents,
)
from ..settings import OPTIMIZED_DENOISE_MODE
from ..text_conditioning import (
    Krea2PromptConditioning,
    Krea2TextConditioner,
    Krea2TextConditionerPort,
    Krea2TextConditioning,
    ReplayIdentityPolicy,
    load_text_conditioning,
)
from ..types import Krea2Request
from ..vae import Krea2VAETilingConfig
from ..vae.tiling import estimate_decode_memory

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Krea2GeneratedImage:
    """One materialized H/W/RGB image and its deterministic request identity."""

    pixels: mx.array
    prompt: str
    seed: int
    index: int

    def __post_init__(self) -> None:
        if self.pixels.ndim != 3 or self.pixels.shape[-1] != 3:
            raise ValueError("Krea generated image must have shape [height, width, 3]")


@dataclass
class Krea2GenerationOutput:
    """Closeable materialized Krea images and optional tensor sidecar products."""

    images: tuple[Krea2GeneratedImage, ...]
    metadata: dict[str, object] = field(default_factory=dict)
    initial_noise: mx.array | None = field(default=None, repr=False)
    final_latents: mx.array | None = field(default=None, repr=False)
    schedule: mx.array | None = field(default=None, repr=False)
    text_conditioning: Krea2TextConditioning | None = field(default=None, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Release every output-owned MLX array exactly once."""
        if self._closed:
            return
        self.images = ()
        self.initial_noise = None
        self.final_latents = None
        self.schedule = None
        self.text_conditioning = None
        self._closed = True
        gc.collect()
        mx.clear_cache()

    def __enter__(self) -> Krea2GenerationOutput:
        if self._closed:
            raise RuntimeError("Krea generation output is closed")
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def _negative_prompts(
    request: Krea2Request,
    *,
    cfg_enabled: bool,
) -> tuple[str, ...] | None:
    validate_capabilities(
        request, {"recipe_mode": "text", "cfg_enabled": cfg_enabled}, location="[generate]"
    )
    negative = request.expanded_negative_prompts()
    if not cfg_enabled:
        return None
    if negative is None:
        return ("",) * request.image_count
    return negative


def _checkpoint(callback: Krea2Checkpoint | None) -> None:
    if callback is not None:
        callback()


def _condition_text(
    conditioner: Krea2TextConditionerPort,
    prompts: tuple[str, ...],
    negative_prompts: tuple[str, ...] | None,
    resources: Krea2Resources,
    reporter: Reporter,
    checkpoint: Krea2Checkpoint | None,
    text_conditioning: Path | None,
    replay_identity_policy: ReplayIdentityPolicy,
    system_prompt: str | None,
) -> Krea2TextConditioning:
    if text_conditioning is not None:
        _checkpoint(checkpoint)
        result = load_text_conditioning(
            text_conditioning,
            resources,
            expected_prompts=prompts,
            expected_negative_prompts=negative_prompts,
            reporter=reporter,
            metadata_policy=replay_identity_policy,
            system_prompt=system_prompt,
        )
        _checkpoint(checkpoint)
        return result
    phase = "condition text"
    branches = 2 if negative_prompts is not None else 1
    reporter.phase_start(phase, total=branches, unit="branch")
    try:
        _checkpoint(checkpoint)
        result = conditioner(
            prompts,
            resources,
            negative_prompts=negative_prompts,
            max_sequence_length=512,
            reporter=reporter,
            system_prompt=system_prompt,
        )
        reporter.phase_advance(phase, branches)
        _checkpoint(checkpoint)
        return result
    finally:
        reporter.phase_end(phase)


def resolve_request_recipe(
    resources: Krea2Resources,
    request: Krea2Request,
) -> tuple[Krea2Geometry, Krea2Recipe]:
    """Resolve the geometry and sampler recipe one request implies; needs no weights."""
    geometry = resolve_geometry(
        request,
        spatial_compression=resources.capabilities.spatial_compression,
        patch_size=resources.capabilities.transformer_patch_size,
    )
    recipe = resolve_recipe(resources.variant, request, geometry, resources.scheduler_config)
    return geometry, recipe


def decode_krea2_latents(
    request: Krea2Request,
    resources: Krea2Resources,
    final_latents: mx.array,
    *,
    geometry: Krea2Geometry,
    prompts: tuple[str, ...],
    seeds: tuple[int, ...],
    components: Krea2ComponentsProvider,
    reporter: Reporter,
    checkpoint: Krea2Checkpoint | None = None,
    on_image: Callable[[Krea2GeneratedImage], None] | None = None,
) -> tuple[
    tuple[Krea2GeneratedImage, ...],
    dict[str, object],
    dict[str, object],
]:
    """Decode one structurally fitted batch through the shared Krea VAE stage."""
    expected_shape = (
        request.image_count,
        resources.capabilities.latent_channels,
        geometry.latent_height,
        geometry.latent_width,
    )
    if tuple(final_latents.shape) != expected_shape:
        raise ValueError(
            f"Krea final latent shape {tuple(final_latents.shape)} does not match {expected_shape}"
        )
    if len(prompts) != request.image_count or len(seeds) != request.image_count:
        raise ValueError("Krea decode identities must match the latent batch")

    images: list[Krea2GeneratedImage] = []
    phase = "decode images"
    reporter.phase_start(phase, total=request.image_count, unit="image")
    try:
        decoder_lease = components.image_decoder(resources)
        try:
            budget = capture_decode_memory_budget()
            vae_dtype = resolve_bound_dtype(resources.vae.dtype_policy, owner="Krea 2 VAE")
            output_scale = resources.vae_decoder.output_scale
            reserved_output_bytes = (
                (request.image_count - 1)
                * geometry.height
                * geometry.width
                * 3
                * output_scale**2
                * (4 if vae_dtype == mx.float32 else 2)
            )
            if request.vae_tiling == "auto":
                tiling = Krea2VAETilingConfig.auto(
                    resources.vae_config,
                    geometry.height,
                    geometry.width,
                    compute_dtype=vae_dtype,
                    output_scale=output_scale,
                    budget=budget,
                    reserved_output_bytes=reserved_output_bytes,
                )
            elif request.vae_tiling == "custom":
                tiling = Krea2VAETilingConfig(
                    tile_sample_size=request.vae_tile_size,
                    tile_sample_stride=request.vae_tile_stride,
                )
            else:
                tiling = None
            tiled = tiling is not None and (
                geometry.height > tiling.tile_sample_size
                or geometry.width > tiling.tile_sample_size
            )
            full_estimate, _largest_buffer = estimate_decode_memory(
                resources.vae_config,
                geometry.height,
                geometry.width,
                compute_dtype=vae_dtype,
                output_scale=output_scale,
            )
            _log.info(
                "Krea 2 VAE plan: %s, full-image estimate=%.2f GiB%s",
                "tiled" if tiled else "full-image",
                full_estimate / 1024**3,
                (
                    f", tile={tiling.tile_sample_size}px, stride={tiling.tile_sample_stride}px"
                    if tiled and tiling is not None
                    else ""
                ),
            )
            for index, prompt in enumerate(prompts):
                _checkpoint(checkpoint)
                decoded = decoder_lease.value(
                    final_latents[index : index + 1],
                    reporter=reporter,
                    tiling=tiling,
                )
                cropped = decoded.pixels[
                    0,
                    : geometry.requested_height * output_scale,
                    : geometry.requested_width * output_scale,
                    :,
                ]
                mx.eval(cropped)
                images.append(
                    Krea2GeneratedImage(
                        pixels=cropped,
                        prompt=prompt,
                        seed=seeds[index],
                        index=index,
                    )
                )
                if on_image is not None:
                    on_image(images[-1])
                reporter.phase_advance(phase)
        finally:
            decoder_lease.close()
        _checkpoint(checkpoint)
    except BaseException:
        images.clear()
        raise
    finally:
        reporter.phase_end(phase)

    decode_metadata: dict[str, object] = {
        "vae_decoder": resources.vae_decoder.value,
        "output_scale": resources.vae_decoder.output_scale,
        "width": geometry.requested_width * resources.vae_decoder.output_scale,
        "height": geometry.requested_height * resources.vae_decoder.output_scale,
    }
    vae_metadata: dict[str, object] = {
        "requested_mode": request.vae_tiling,
        "tiling": tiled,
        "tile_sample_size": tiling.tile_sample_size if tiled and tiling else None,
        "tile_sample_stride": tiling.tile_sample_stride if tiled and tiling else None,
        "memory": budget.to_dict(),
        "estimated_full_peak_bytes": full_estimate,
        "reserved_output_bytes": reserved_output_bytes,
    }
    return tuple(images), decode_metadata, vae_metadata


@memory_managed
def generate_krea2(
    request: Krea2Request,
    resources: Krea2Resources,
    *,
    components: Krea2ComponentsProvider | None = None,
    text_conditioner: Krea2TextConditionerPort | None = None,
    reporter: Reporter | None = None,
    checkpoint: Krea2Checkpoint | None = None,
    text_replay_policy: ReplayIdentityPolicy = "require",
    on_image: Callable[[int, Krea2GenerationOutput], None] | None = None,
    leading_adapters: tuple[AdapterSelection, ...] = (),
) -> Krea2GenerationOutput:
    """Run independent seeds with shared conditioning and explicit model lifetimes.

    ``leading_adapters`` are applied before the request's own adapters without
    becoming part of the request: the editing recipe hands its resolved edit
    adapter through this port so the request still records only what the user
    wrote, which is what a later restart replays.
    """
    validate_record(request, location="[generate]")
    admit_recipe_mode(request, "text")
    request = resolve_request_sources(request)
    models = SeedModels()
    base = components if components is not None else Krea2Components(reporter=reporter)
    ports = Krea2SeedComponents(base, models)
    jobs = []
    conditioning_rows = []
    outputs = []
    try:
        negatives = request.expanded_negative_prompts()
        _, recipe = resolve_request_recipe(resources, request)
        adapters = _resolve_adapters(
            request, resources, reporter or NullReporter(), leading_adapters=leading_adapters
        )
        # One Qwen lifetime per request: every prompt row is encoded (or replayed)
        # together, and each seed borrows its prompt's row. Encoding per prompt
        # would reload and release the text encoder once per distinct prompt.
        rows = resolve_request_sources(derive_config(request, seeds=None, images_per_prompt=1))
        encoded = _condition_text(
            text_conditioner or Krea2TextConditioner(),
            rows.expanded_prompts(),
            _negative_prompts(rows, cfg_enabled=recipe.cfg_enabled),
            resources,
            reporter or NullReporter(),
            checkpoint,
            request.text_conditioning,
            text_replay_policy,
            request.system_prompt,
        )
        for index, (prompt, seed) in enumerate(
            zip(
                request.expanded_prompts(),
                request.selected_seeds() * len(request.prompts),
                strict=True,
            )
        ):
            conditioning = select_conditioning(encoded, (index // len(request.selected_seeds()),))
            single = derive_config(
                request,
                prompt=prompt,
                negative_prompt=None if negatives is None else negatives[index],
                seed=seed,
                seeds=None,
                images_per_prompt=1,
                text_conditioning=None,
            )
            job = _krea_stages(
                single,
                resources,
                components=ports,
                reporter=reporter,
                checkpoint=checkpoint,
                conditioning=conditioning,
                adapters=adapters,
            )
            jobs.append(job)
            prepared = next(job)
            assert prepared is not None
            conditioning_rows.append(prepared)
        for index, output in enumerate(execute_seed_stages(jobs, models, request.seed_mode)):
            outputs.append(output)
            if on_image is not None:
                on_image(index, output)
        images = tuple(
            replace(output.images[0], index=index) for index, output in enumerate(outputs)
        )
        metadata = dict(outputs[0].metadata)
        metadata.update(
            image_count=len(images),
            prompts=list(request.expanded_prompts()),
            negative_prompts=None if negatives is None else list(negatives),
            seeds=list(request.selected_seeds()),
            seed_mode=request.seed_mode,
            warnings=_joined_warnings(outputs),
            denoise=_joined_denoise(outputs),
        )
        first_noise = outputs[0].metadata["noise"]
        assert isinstance(first_noise, dict)
        noise_metadata = dict(first_noise)
        noise_metadata["per_image_seeds"] = [image.seed for image in images]
        streams: list[object] = []
        for output in outputs:
            receipt = output.metadata["noise"]
            assert isinstance(receipt, dict)
            assert isinstance(receipt["streams"], list)
            streams.extend(receipt["streams"])
        noise_metadata["streams"] = streams
        noises: list[mx.array] = []
        latents: list[mx.array] = []
        for output in outputs:
            assert output.initial_noise is not None
            assert output.final_latents is not None
            noises.append(output.initial_noise)
            latents.append(output.final_latents)
        metadata["noise"] = noise_metadata
        result = Krea2GenerationOutput(
            images,
            metadata,
            initial_noise=mx.concatenate(noises, axis=0),
            final_latents=mx.concatenate(latents, axis=0),
            schedule=outputs[0].schedule,
            text_conditioning=join_conditioning(conditioning_rows),
        )
        mx.eval(result.initial_noise, result.final_latents)
        return result
    finally:
        for job in jobs:
            job.close()
        models.close()
        for output in outputs:
            output.close()


def _resolve_adapters(
    request: Krea2Request,
    resources: Krea2Resources,
    reporter: Reporter,
    *,
    leading_adapters: tuple[AdapterSelection, ...] = (),
) -> tuple[dict[str, object] | None, dict[str, object] | None, Krea2AdapterPlan | None]:
    """Admit the request's adapters once, before any encoder or denoiser loads."""
    # Only a run that enables a profile needs a cache location at all.
    profiles_enabled = request.turbo_4step or request.turbo_sda
    selections, turbo_resolution, sda_resolution = resolve_profile_adapters(
        resources.hf_home if profiles_enabled else None,
        four_step=request.four_step_profile(),
        sda=request.sda_profile(),
        additional=(*leading_adapters, *request.resolved_adapters()),
    )
    # A profile's recipe is a default, so leaving it is legitimate but worth saying.
    for deviation in request.profile_deviations():
        _log.warning("Four-step Turbo profile: this run uses %s", deviation)
    if request.turbo_sda:
        _, recipe = resolve_request_recipe(resources, request)
        pairing = sda_pairing_warning(
            recipe.steps, tuple(sigma_metadata(timestep_schedule(recipe)) or ())
        )
        if pairing is not None:
            _log.warning("%s", pairing)
    if not selections:
        return turbo_resolution, sda_resolution, None
    with reported_phase(reporter, "validate adapters"):
        # A scheduled array must match the step count the recipe finally resolved,
        # which a profile can rewrite after the request was built.
        _, recipe = resolve_request_recipe(resources, request)
        require_adapter_schedule_steps(selections, recipe.steps)
        return (
            turbo_resolution,
            sda_resolution,
            prepare_krea2_adapters(resources.transformer_config, selections),
        )


def _joined_warnings(outputs: list[Krea2GenerationOutput]) -> list[str]:
    """Report every seed's warnings once, in first-seen order."""
    seen: dict[str, None] = {}
    for output in outputs:
        warnings = output.metadata["warnings"]
        if not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
            raise RuntimeError("Krea seed returned malformed warnings metadata")
        seen.update(dict.fromkeys(warnings))
    return list(seen)


def _joined_denoise(outputs: list[Krea2GenerationOutput]) -> dict[str, object]:
    """Report denoiser call counts as request totals, not one seed's share."""
    first = outputs[0].metadata["denoise"]
    assert isinstance(first, dict)
    joined = dict(first)
    for name in ("conditional_calls", "unconditional_calls"):
        total = 0
        for output in outputs:
            receipt = output.metadata["denoise"]
            assert isinstance(receipt, dict)
            value = receipt[name]
            assert isinstance(value, int)
            total += value
        joined[name] = total
    return joined


def select_conditioning(
    value: Krea2TextConditioning, rows: tuple[int, ...]
) -> Krea2TextConditioning:
    def branch(value: Krea2PromptConditioning) -> Krea2PromptConditioning:
        take = mx.array(rows, dtype=mx.int32)
        return replace(
            value,
            hidden_states=value.hidden_states[take],
            attention_mask=value.attention_mask[take],
            position_ids=value.position_ids[:, take],
            prompts=tuple(value.prompts[index] for index in rows),
        )

    return replace(
        value,
        positive=branch(value.positive),
        negative=None if value.negative is None else branch(value.negative),
    )


def join_conditioning(values: list[Krea2TextConditioning]) -> Krea2TextConditioning:
    def branch(rows: list[Krea2PromptConditioning]) -> Krea2PromptConditioning:
        return replace(
            rows[0],
            hidden_states=mx.concatenate([row.hidden_states for row in rows]),
            attention_mask=mx.concatenate([row.attention_mask for row in rows]),
            position_ids=mx.concatenate([row.position_ids for row in rows], axis=1),
            prompts=tuple(*row.prompts for row in rows),
            token_identity=rows[0].token_identity
            if len(rows) == 1
            else hashlib.sha256("\0".join(row.token_identity for row in rows).encode()).hexdigest(),
        )

    negative = [value.negative for value in values if value.negative is not None]
    return replace(
        values[0],
        positive=branch([value.positive for value in values]),
        negative=branch(negative) if negative else None,
    )


def _krea_stages(
    request: Krea2Request,
    resources: Krea2Resources,
    *,
    components: Krea2ComponentsProvider | None = None,
    reporter: Reporter | None = None,
    checkpoint: Krea2Checkpoint | None = None,
    conditioning: Krea2TextConditioning,
    adapters: tuple[dict[str, object] | None, dict[str, object] | None, Krea2AdapterPlan | None],
) -> Generator[Krea2TextConditioning | None, None, Krea2GenerationOutput]:
    """Generate one Krea still without co-resident Qwen, MMDiT, and VAE weights.

    The caller owns text conditioning and the adapter plan for the whole
    request and hands this stage its already-encoded row.
    """
    if request.source_images:
        raise ValueError("use Krea2Runner.generate or generate_krea2_edit for source images")
    sink = reporter if reporter is not None else NullReporter()
    selected_components = components if components is not None else Krea2Components(reporter=sink)
    geometry, recipe = resolve_request_recipe(resources, request)
    turbo_resolution, sda_resolution, adapter_plan = adapters
    prompts = conditioning.positive.prompts
    active_negative = conditioning.negative if recipe.cfg_enabled else None
    if recipe.cfg_enabled and active_negative is None:
        raise ValueError("Krea text conditioning has no negative branch required by guidance")
    negative_prompts = None if active_negative is None else active_negative.prompts

    yield conditioning
    activation_dtype = resolve_bound_dtype(
        resources.transformer.dtype_policy,
        owner="Krea 2 transformer",
    )
    noise = None
    denoised = None
    final_latents = None
    try:
        _checkpoint(checkpoint)
        transformer_lease = selected_components.transformer(resources, adapter_plan=adapter_plan)
        try:
            adapter_metadata = [
                receipt.to_metadata() for receipt in transformer_lease.value.adapter_receipts
            ]
            adapter_warnings = [
                f"{receipt.path.name}: {warning}"
                for receipt in transformer_lease.value.adapter_receipts
                for warning in receipt.warnings
            ]
            noise = create_initial_noise(
                request,
                geometry,
                latent_channels=resources.capabilities.latent_channels,
                dtype=activation_dtype,
            )
            packed = pack_latents(
                noise.unpacked,
                patch_size=resources.capabilities.transformer_patch_size,
            )
            positive = prepare_branch(
                packed,
                conditioning.positive,
                geometry,
                activation_dtype=activation_dtype,
                reference_denoise=resources.reference_denoise,
            )
            negative = (
                None
                if active_negative is None
                else prepare_branch(
                    packed,
                    active_negative,
                    geometry,
                    activation_dtype=activation_dtype,
                    reference_denoise=resources.reference_denoise,
                )
            )
            denoised = denoise_krea2(
                transformer_lease.value,
                positive,
                recipe,
                negative=negative,
                reporter=sink,
                checkpoint=checkpoint,
            )
            final_latents = unpack_latents(
                denoised.packed,
                geometry,
                latent_channels=resources.capabilities.latent_channels,
                patch_size=resources.capabilities.transformer_patch_size,
            ).astype(activation_dtype)
            mx.eval(final_latents)
        finally:
            transformer_lease.close()
        _checkpoint(checkpoint)

        yield None
        images, decode_metadata, vae_metadata = decode_krea2_latents(
            request,
            resources,
            final_latents,
            geometry=geometry,
            prompts=prompts,
            seeds=noise.seeds,
            components=selected_components,
            reporter=sink,
            checkpoint=checkpoint,
        )

        metadata: dict[str, object] = {
            "variant": resources.variant.value,
            "revision": resources.revision,
            "recipe_identity": resources.recipe_identity,
            "recipe": recipe.to_metadata(),
            "geometry": geometry.to_metadata(),
            "decode": decode_metadata,
            "image_count": request.image_count,
            "prompts": list(prompts),
            "negative_prompts": (None if negative_prompts is None else list(negative_prompts)),
            "noise": noise.to_metadata(),
            "text_conditioning": conditioning.provenance.to_metadata(),
            "adapters": adapter_metadata,
            "turbo_4step": turbo_resolution,
            "turbo_sda": sda_resolution,
            "check_adapter_weights": resources.check_adapter_weights,
            "warnings": adapter_warnings,
            "denoise": {
                "mode": "reference" if resources.reference_denoise else OPTIMIZED_DENOISE_MODE,
                "dtype": resources.transformer.dtype_policy,
                "conditional_calls": denoised.conditional_calls,
                "unconditional_calls": denoised.unconditional_calls,
                "euler": "FP32 scalar multiply narrowed before low-precision add",
                "sigmas": sigma_metadata(denoised.schedule),
            },
            "vae": vae_metadata,
        }
        if conditioning.replay_receipt is not None:
            metadata["text_conditioning_replay"] = dict(conditioning.replay_receipt)
        return Krea2GenerationOutput(
            images=tuple(images),
            metadata=metadata,
            initial_noise=noise.unpacked,
            final_latents=final_latents,
            schedule=denoised.schedule,
            text_conditioning=conditioning,
        )
    except BaseException:
        noise = None
        denoised = None
        final_latents = None
        gc.collect()
        mx.clear_cache()
        raise


__all__ = [
    "Krea2GeneratedImage",
    "Krea2GenerationOutput",
    "decode_krea2_latents",
    "generate_krea2",
]
