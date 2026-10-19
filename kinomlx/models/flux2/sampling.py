"""Klein Euler trajectories with fixed ordered references and external base CFG."""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import mlx.core as mx

from kinomlx.adapters.live import (
    bind_live_adapter_sigmas,
    live_adapters,
    set_live_adapter_step,
)
from kinomlx.adapters.validation import check_adapter_weights
from kinomlx.config_constraints import ChoiceConstraint
from kinomlx.config_records import config_record
from kinomlx.config_rules import validate_record
from kinomlx.reporting import NullReporter, Reporter, reported_phase
from kinomlx.samplers.config import SigmaScheduleConstraint
from kinomlx.samplers.sigmas import resolved_sigmas

from .catalog import Flux2Variant
from .kv import ReferenceKV
from .recipe_rules import SAMPLING_RECIPE, recipe_values
from .request_constraints import CFG_MODE, DENOISE_STEPS, GUIDANCE, Flux2CFGMode
from .velocity import PreparableVelocityModel, PreparedVelocity, VelocityModel


def _unprepared_velocity(
    model: VelocityModel,
    text: mx.array,
    image_ids: mx.array,
    text_ids: mx.array,
    image: mx.array,
    timestep: mx.array,
    reference_kv: ReferenceKV | None,
) -> mx.array:
    if reference_kv is None:
        return model(image, text, timestep, image_ids, text_ids)
    return model(image, text, timestep, image_ids, text_ids, reference_kv=reference_kv)


def _prepare_velocities(
    model: VelocityModel, text: mx.array, image_ids: tuple[mx.array, ...], text_ids: mx.array
) -> tuple[PreparedVelocity, ...]:
    if isinstance(model, PreparableVelocityModel):
        return model.prepare_velocities(text, image_ids, text_ids)
    return tuple(partial(_unprepared_velocity, model, text, ids, text_ids) for ids in image_ids)


@config_record
@dataclass(frozen=True)
class SamplingRecipe:
    variant: Flux2Variant = field(
        default=Flux2Variant.DISTILLED,
        metadata={
            "config_constraints": (ChoiceConstraint(tuple(Flux2Variant)),),
            "config_rules": (SAMPLING_RECIPE,),
        },
    )
    steps: int | None = field(default=None, metadata={"config_constraints": (DENOISE_STEPS,)})
    guidance: float | None = field(default=None, metadata={"config_constraints": (GUIDANCE,)})
    cfg_mode: Flux2CFGMode = field(
        default=Flux2CFGMode.SEQUENTIAL, metadata={"config_constraints": (CFG_MODE,)}
    )
    recompute_reference_kv: bool = False
    sigmas: tuple[float, ...] = field(
        default=(), metadata={"config_constraints": (SigmaScheduleConstraint(),)}
    )

    @property
    def explicit_sigmas(self) -> bool:
        """Whether the user supplied the schedule instead of shifting a uniform one."""
        return bool(self.sigmas)

    def sigma_array(self, image_tokens: int) -> mx.array:
        """Return this recipe's schedule: explicit values, else the Klein shift.

        Klein's single schedule-production point. An alternative spacing plugs in
        here as a ``kinomlx.samplers.sigmas.SigmaScheduler``; the built-in shift
        stays inline until there is a second one.
        """
        return resolved_sigmas(
            self.sigmas,
            lambda: schedule(image_tokens, self.resolved()[0]),
            label="Klein schedule",
        ).to_array()

    def resolved(self) -> tuple[int, float]:
        validate_record(self, location="[generate]")
        return recipe_values(self.variant, self.steps, self.guidance, self.sigmas)


def schedule(image_tokens: int, steps: int) -> mx.array:
    """Empirical Klein shift on the target grid; references never change it."""
    if image_tokens < 1 or steps < 1:
        raise ValueError("Schedule requires positive target token and step counts")
    long = 0.00016927 * image_tokens + 0.45666666
    short = 8.73809524e-05 * image_tokens + 1.89833333
    mu = long if image_tokens > 4300 else long + (short - long) * (200 - steps) / 190
    factor = math.exp(mu)
    # The scheduler first casts NumPy's linspace to FP32, then shifts it.
    sigma = mx.array([1 - i / steps for i in range(steps)], dtype=mx.float32)
    shifted = factor / (factor + (1 / sigma - 1))
    return mx.concatenate([shifted, mx.zeros((1,), dtype=mx.float32)])


def euler_step(current: mx.array, velocity: mx.array, delta: mx.array) -> mx.array:
    # Torch's scalar promotion multiplies dt and velocity in the model dtype,
    # then adds the increment to an FP32 sample before rounding back.
    increment = delta.astype(velocity.dtype) * velocity
    return (current.astype(mx.float32) + increment.astype(mx.float32)).astype(current.dtype)


def denoise(
    model: VelocityModel,
    noise: mx.array,
    text: mx.array,
    image_ids: mx.array,
    text_ids: mx.array,
    *,
    recipe: SamplingRecipe | None = None,
    references: mx.array | None = None,
    reference_ids: mx.array | None = None,
    negative: mx.array | None = None,
    reporter: Reporter | None = None,
    on_step: Callable[[int, mx.array, mx.array], None] | None = None,
) -> mx.array:
    """Update target tokens only; supplied conditioning is never mutated."""
    recipe = recipe or SamplingRecipe()
    steps, guidance = recipe.resolved()
    if noise.ndim != 3 or noise.shape[0] != 1 or noise.shape[-1] != 128:
        raise ValueError("Klein noise requires [1, target tokens, 128]")
    if text.ndim != 3 or text.shape[0] != 1 or text.shape[-1] != 12288:
        raise ValueError("Klein text requires [1, text tokens, 12288]")
    if image_ids.shape != (1, noise.shape[1], 4) or text_ids.shape != (1, text.shape[1], 4):
        raise ValueError("Klein positions must match their token sequences")
    if (references is None) != (reference_ids is None):
        raise ValueError("References and reference positions must be supplied together")
    if references is not None and (
        references.ndim != 3
        or references.shape[0] != 1
        or references.shape[1] < 1
        or references.shape[-1] != 128
        or reference_ids is None
        or reference_ids.shape != (1, references.shape[1], 4)
    ):
        raise ValueError("Reference tokens and positions have incompatible shapes")
    if guidance > 1 and negative is None:
        raise ValueError(
            "Base CFG requires explicit negative embeddings (encode empty text for default CFG)"
        )
    if negative is not None and (guidance == 1 or negative.shape != text.shape):
        raise ValueError(
            "Negative embeddings require active CFG and the same shape as positive text"
        )
    if noise.dtype not in {mx.float16, mx.bfloat16, mx.float32} or any(
        value is not None and value.dtype != noise.dtype for value in (text, references, negative)
    ):
        raise ValueError("Klein noise and conditioning require a common FP16, BF16, or FP32 dtype")
    for value in (noise, text, references, negative, image_ids, text_ids, reference_ids):
        if value is not None and not bool(mx.all(mx.isfinite(value)).item()):
            raise ValueError("Klein conditioning and noise must be finite")
    sigmas = recipe.sigma_array(noise.shape[1])
    current = noise
    progress = reporter or NullReporter()
    cache = None
    progress.phase_start("denoise", total=steps, unit="step")
    try:
        kv_edit = recipe.variant == Flux2Variant.KV and references is not None
        # Reference K/V is extracted once and reused, so a strength that changes
        # after that extraction would be read back from a cache built at the
        # first step's strength. Recomputing the cache each step is the fix.
        bind_live_adapter_sigmas(model, sigmas)
        controller = live_adapters(model)
        if kv_edit and not recipe.recompute_reference_kv and controller is not None:
            varying = sorted({term.order_key for term in controller.terms if term.varies})
            if varying:
                raise ValueError(
                    "Klein KV editing reuses reference K/V extracted at the first step, "
                    "so a per-step adapter schedule would be applied to a stale cache: "
                    f"{', '.join(varying)}. Fuse the adapter at one constant strength, "
                    "or enable recompute_reference_kv (--recompute-reference-kv) to extract "
                    "the reference K/V again at every step."
                )
        negative_velocity = None
        if kv_edit:
            assert reference_ids is not None
            extraction_ids = mx.concatenate([reference_ids, image_ids], axis=1)
            predict, reuse = _prepare_velocities(model, text, (extraction_ids, image_ids), text_ids)
        else:
            ids = (
                image_ids
                if reference_ids is None
                else mx.concatenate([image_ids, reference_ids], axis=1)
            )
            conditioned = (
                mx.concatenate([text, negative])
                if negative is not None and recipe.cfg_mode == Flux2CFGMode.BATCHED
                else text
            )
            (predict,) = _prepare_velocities(model, conditioned, (ids,), text_ids)
            if negative is not None and recipe.cfg_mode == "sequential":
                (negative_velocity,) = _prepare_velocities(model, negative, (ids,), text_ids)
        for index in range(steps):
            # Both CFG branches of one step share the step's adapter strengths.
            set_live_adapter_step(model, index)
            # Preserve the reference pipeline's model-dtype timestep round trip.
            timestep = ((sigmas[index : index + 1] * 1000).astype(noise.dtype) / 1000).astype(
                noise.dtype
            )
            if recipe.variant == Flux2Variant.KV and references is not None:
                assert reference_ids is not None
                if cache is None:
                    cache = model.new_reference_kv(references.shape[1])
                with reported_phase(progress, "kv_extract" if cache.extracting else "kv_reuse"):
                    if cache.extracting:
                        image = mx.concatenate([references, current], axis=1)
                    else:
                        image = current
                    velocity = (predict if cache.extracting else reuse)(image, timestep, cache)
                    # Settle the velocity before releasing extraction state or
                    # reusing the already materialized reference rows.
                    mx.eval(velocity, *cache.tensors())
                cache.extracting = False
                if recipe.recompute_reference_kv:
                    cache.clear()
                    cache = None
            else:
                velocity = _full_velocity(
                    predict,
                    negative_velocity,
                    current,
                    timestep,
                    references,
                    negative is not None and recipe.cfg_mode == Flux2CFGMode.BATCHED,
                    guidance,
                )
            current = euler_step(current, velocity, sigmas[index + 1] - sigmas[index])
            mx.eval(current)
            check_adapter_weights(model, context=f"Klein step {index + 1}", reporter=progress)
            if not bool(mx.all(mx.isfinite(current)).item()):
                raise FloatingPointError(f"Klein produced nonfinite latents at step {index + 1}")
            progress.phase_advance("denoise")
            if on_step is not None:
                on_step(index, current, velocity)
    finally:
        if cache is not None:
            cache.clear()
        progress.phase_end("denoise")
    return current


def _full_velocity(
    predict: PreparedVelocity,
    negative_velocity: PreparedVelocity | None,
    current: mx.array,
    timestep: mx.array,
    references: mx.array | None,
    batched_cfg: bool,
    guidance: float,
) -> mx.array:
    image = current if references is None else mx.concatenate([current, references], axis=1)
    if batched_cfg:
        both = predict(
            mx.concatenate([image, image]),
            mx.concatenate([timestep, timestep]),
            None,
        )[:, : current.shape[1]]
        positive, uncond = mx.split(both, 2, axis=0)
        velocity = uncond + guidance * (positive - uncond)
    else:
        velocity = predict(image, timestep, None)[:, : current.shape[1]]
        if negative_velocity is not None:
            # Materialize the positive branch before constructing the
            # negative branch so sequential CFG has a real memory bound.
            mx.eval(velocity)
            uncond = negative_velocity(image, timestep, None)[:, : current.shape[1]]
            velocity = uncond + guidance * (velocity - uncond)
    return velocity
