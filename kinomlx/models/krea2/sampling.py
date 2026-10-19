"""Exact Krea geometry, noise, schedule, CFG, packing, and denoise loop."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import mlx.core as mx

from kinomlx.adapters.live import bind_live_adapter_sigmas, set_live_adapter_step
from kinomlx.adapters.validation import check_adapter_weights
from kinomlx.errors import KinoMLXError
from kinomlx.modeling.dtypes import float_dtype_name
from kinomlx.reporting import NullReporter, Reporter
from kinomlx.samplers.noise import (
    NoiseStreamState,
    create_normal_noise_stream,
    noise_compatibility_profile,
)
from kinomlx.samplers.sigmas import SigmaSchedule, resolved_sigmas
from kinomlx.samplers.steps import euler_velocity_step

from .adapters.turbo_profile import TURBO_4STEP_RECIPE
from .catalog import Krea2Variant, variant_spec
from .components import Krea2TransformerPort
from .metadata import Krea2SchedulerConfig
from .text_conditioning import Krea2PromptConditioning
from .transformer.inputs import compact_text_groups
from .types import Krea2Request

type Krea2Checkpoint = Callable[[], None]


class Krea2DenoiseError(KinoMLXError, RuntimeError):
    """Typed sampler failure safe to render without a traceback."""


@dataclass(frozen=True)
class Krea2Geometry:
    """Requested pixels and the aligned latent/token grid used by Krea."""

    requested_width: int
    requested_height: int
    width: int
    height: int
    alignment: int
    latent_width: int
    latent_height: int
    image_token_width: int
    image_token_height: int
    image_sequence_length: int

    def to_metadata(self) -> dict[str, int | bool]:
        return {
            "requested_width": self.requested_width,
            "requested_height": self.requested_height,
            "aligned_width": self.width,
            "aligned_height": self.height,
            "alignment": self.alignment,
            "latent_width": self.latent_width,
            "latent_height": self.latent_height,
            "image_token_width": self.image_token_width,
            "image_token_height": self.image_token_height,
            "image_sequence_length": self.image_sequence_length,
            "crop_required": (
                self.requested_width != self.width or self.requested_height != self.height
            ),
        }


@dataclass(frozen=True)
class Krea2Recipe:
    """One fully resolved Raw or Turbo sampler recipe."""

    variant: Krea2Variant
    steps: int
    guidance: float
    mu: float | None
    effective_mu: float
    cfg_enabled: bool
    matches_variant_defaults: bool
    sigmas: tuple[float, ...] = ()

    @property
    def explicit_sigmas(self) -> bool:
        """Whether the user supplied the schedule instead of shifting a uniform one."""
        return bool(self.sigmas)

    def to_metadata(self) -> dict[str, object]:
        return {
            "variant": self.variant.value,
            "steps": self.steps,
            "guidance": self.guidance,
            "mu": self.mu,
            "effective_mu": self.effective_mu,
            "cfg_enabled": self.cfg_enabled,
            "matches_variant_defaults": self.matches_variant_defaults,
            "cfg_convention": "conditional + guidance * (conditional - unconditional)",
            "sigma_source": "explicit" if self.explicit_sigmas else "shifted-uniform",
        }


@dataclass(frozen=True)
class Krea2NoiseBatch:
    """Materialized per-image noise and each independent seeded stream receipt."""

    unpacked: mx.array
    seeds: tuple[int, ...]
    states: tuple[NoiseStreamState, ...]

    def to_metadata(self) -> dict[str, object]:
        backend = self.states[0].backend
        return {
            "backend": backend,
            "compatibility_profile": noise_compatibility_profile(backend),
            "per_image_seeds": list(self.seeds),
            "streams": [state.to_metadata() for state in self.states],
        }


@dataclass(frozen=True)
class Krea2PreparedBranch:
    """One text branch paired with shared image tokens, positions, and mask."""

    image_tokens: mx.array
    encoder_hidden_states: mx.array
    position_ids: mx.array
    attention_mask: mx.array | None


@dataclass(frozen=True)
class Krea2DenoiseResult:
    """Packed final state plus schedule and call-count receipts."""

    packed: mx.array
    schedule: mx.array
    conditional_calls: int
    unconditional_calls: int


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def resolve_geometry(
    request: Krea2Request,
    *,
    spatial_compression: int,
    patch_size: int,
) -> Krea2Geometry:
    """Round requested pixels up to the exact Krea latent-patch alignment."""
    if spatial_compression <= 0 or patch_size <= 0:
        raise ValueError("Krea compression and patch size must be positive")
    alignment = spatial_compression * patch_size
    width = _round_up(request.width, alignment)
    height = _round_up(request.height, alignment)
    latent_width = width // spatial_compression
    latent_height = height // spatial_compression
    token_width = latent_width // patch_size
    token_height = latent_height // patch_size
    return Krea2Geometry(
        requested_width=request.width,
        requested_height=request.height,
        width=width,
        height=height,
        alignment=alignment,
        latent_width=latent_width,
        latent_height=latent_height,
        image_token_width=token_width,
        image_token_height=token_height,
        image_sequence_length=token_width * token_height,
    )


def calculate_dynamic_mu(
    image_sequence_length: int,
    scheduler: Krea2SchedulerConfig,
) -> float:
    """Interpolate Krea's resolution shift from the scheduler declaration."""
    slope = (scheduler.max_shift - scheduler.base_shift) / (
        scheduler.max_image_seq_len - scheduler.base_image_seq_len
    )
    return slope * image_sequence_length + (
        scheduler.base_shift - slope * scheduler.base_image_seq_len
    )


def resolve_recipe(
    variant: Krea2Variant,
    request: Krea2Request,
    geometry: Krea2Geometry,
    scheduler: Krea2SchedulerConfig,
) -> Krea2Recipe:
    """Apply only the selected variant's defaults, then resolve dynamic shift."""
    spec = variant_spec(variant)
    steps = (
        (int(TURBO_4STEP_RECIPE["steps"]) if request.turbo_4step else spec.default_steps)
        if request.steps is None
        else request.steps
    )
    guidance = (
        (TURBO_4STEP_RECIPE["guidance"] if request.turbo_4step else spec.default_guidance)
        if request.guidance is None
        else request.guidance
    )
    mu = (
        (TURBO_4STEP_RECIPE["mu"] if request.turbo_4step else spec.default_mu)
        if request.mu is None
        else request.mu
    )
    sigmas = SigmaSchedule(request.sigmas) if request.sigmas else None
    if sigmas is not None:
        # The array is the schedule, so it also fixes the step count.
        steps = sigmas.steps
    if steps <= 0:
        raise ValueError("Krea sampler steps must be positive")
    if not math.isfinite(guidance) or guidance < 0.0:
        raise ValueError("Krea guidance must be finite and nonnegative")
    if mu is not None and not math.isfinite(mu):
        raise ValueError("Krea mu must be finite")
    effective_mu = (
        calculate_dynamic_mu(geometry.image_sequence_length, scheduler) if mu is None else mu
    )
    return Krea2Recipe(
        variant=variant,
        steps=steps,
        guidance=guidance,
        mu=mu,
        effective_mu=effective_mu,
        cfg_enabled=guidance > 0.0,
        matches_variant_defaults=(
            sigmas is None
            and steps == spec.default_steps
            and guidance == spec.default_guidance
            and mu == spec.default_mu
        ),
        sigmas=() if sigmas is None else sigmas.values,
    )


def _shifted_uniform(recipe: Krea2Recipe) -> mx.array:
    """Krea's uniform 1-to-0 ramp under the recipe's resolved shift."""
    timesteps = mx.linspace(1.0, 0.0, recipe.steps + 1, dtype=mx.float32)
    exp_mu = math.exp(recipe.effective_mu)
    shifted = exp_mu / (exp_mu + (1.0 / timesteps - 1.0))
    # Avoid relying on a backend's infinity arithmetic for the declared endpoint.
    return mx.concatenate([shifted[:-1], mx.zeros((1,), dtype=mx.float32)])


def timestep_schedule(recipe: Krea2Recipe) -> mx.array:
    """Return the recipe's FP32 flow schedule: explicit values, else shifted uniform.

    This is Krea's single schedule-production point. An alternative spacing plugs
    in here as a ``kinomlx.samplers.sigmas.SigmaScheduler``, which exists for that
    purpose; the built-in shift stays inline until there is a second one.
    """
    return resolved_sigmas(
        recipe.sigmas, lambda: _shifted_uniform(recipe), label="Krea 2 schedule"
    ).to_array()


def create_initial_noise(
    request: Krea2Request,
    geometry: Krea2Geometry,
    *,
    latent_channels: int,
    dtype: mx.Dtype,
) -> Krea2NoiseBatch:
    """Create one independent ``seed + image_index`` normal draw per image."""
    rows: list[mx.array] = []
    states: list[NoiseStreamState] = []
    seeds = request.selected_seeds() * len(request.prompts)
    shape = (1, latent_channels, geometry.latent_height, geometry.latent_width)
    for seed in seeds:
        stream = create_normal_noise_stream(seed, backend=request.noise_backend)
        rows.append(stream.normal(shape, dtype))
        states.append(stream.state)
    unpacked = mx.concatenate(rows, axis=0)
    mx.eval(unpacked)
    return Krea2NoiseBatch(unpacked=unpacked, seeds=seeds, states=tuple(states))


def pack_latents(latent: mx.array, *, patch_size: int) -> mx.array:
    """Patchify BCHW latent pixels into Krea image tokens."""
    if latent.ndim != 4:
        raise ValueError("Krea latent packing requires BCHW input")
    batch, channels, height, width = latent.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Krea latent dimensions must be divisible by patch_size")
    return (
        latent.reshape(
            batch,
            channels,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
        )
        .transpose(0, 2, 4, 1, 3, 5)
        .reshape(
            batch,
            (height // patch_size) * (width // patch_size),
            channels * patch_size * patch_size,
        )
    )


def unpack_latents(
    tokens: mx.array,
    geometry: Krea2Geometry,
    *,
    latent_channels: int,
    patch_size: int,
) -> mx.array:
    """Undo Krea patchification into normalized BCHW VAE latents."""
    expected = (
        tokens.shape[0],
        geometry.image_sequence_length,
        latent_channels * patch_size * patch_size,
    )
    if tuple(tokens.shape) != expected:
        raise ValueError(f"Krea packed latent has shape {tuple(tokens.shape)}, expected {expected}")
    batch = tokens.shape[0]
    return (
        tokens.reshape(
            batch,
            geometry.image_token_height,
            geometry.image_token_width,
            latent_channels,
            patch_size,
            patch_size,
        )
        .transpose(0, 3, 1, 4, 2, 5)
        .reshape(
            batch,
            latent_channels,
            geometry.latent_height,
            geometry.latent_width,
        )
    )


def prepare_branch(
    image_tokens: mx.array,
    conditioning: Krea2PromptConditioning,
    geometry: Krea2Geometry,
    *,
    activation_dtype: mx.Dtype,
    reference_denoise: bool = False,
) -> Krea2PreparedBranch:
    """Prepare positions once and compact equal-length text rows before denoising."""
    batch, image_length, _channels = image_tokens.shape
    hidden = conditioning.hidden_states.astype(activation_dtype)
    text_mask = conditioning.attention_mask
    if hidden.shape[0] != batch or text_mask.shape[0] != batch:
        raise ValueError("Krea conditioning and image batches must match")
    if image_length != geometry.image_sequence_length:
        raise ValueError("Krea image tokens do not match resolved geometry")
    text_length = hidden.shape[1]
    if tuple(text_mask.shape) != (batch, text_length):
        raise ValueError("Krea text mask must match conditioning batch and length")

    row = mx.repeat(
        mx.arange(geometry.image_token_height, dtype=mx.float32),
        geometry.image_token_width,
    )
    column = mx.tile(
        mx.arange(geometry.image_token_width, dtype=mx.float32),
        geometry.image_token_height,
    )
    image_positions = mx.stack([mx.zeros_like(row), row, column], axis=-1)
    image_positions = mx.broadcast_to(image_positions[None, :, :], (batch, image_length, 3))
    # Coordinates must not inherit BF16/FP16 activation rounding before RoPE.
    text_positions = mx.zeros((batch, text_length, 3), dtype=mx.float32)
    position_ids = mx.concatenate([text_positions, image_positions], axis=1)
    image_mask = mx.ones((batch, image_length), dtype=mx.bool_)
    attention_mask = mx.concatenate([text_mask.astype(mx.bool_), image_mask], axis=1)
    if not reference_denoise:
        groups = compact_text_groups(hidden, position_ids, attention_mask)
        if len(groups) == 1:
            group = groups[0]
            mx.eval(group.hidden_states, group.position_ids)
            return Krea2PreparedBranch(image_tokens, group.hidden_states, group.position_ids, None)
        # The transformer runs unequal-length groups independently and restores
        # their batch order. Never retain padding inside a mask-free forward.
    return Krea2PreparedBranch(
        image_tokens=image_tokens,
        encoder_hidden_states=hidden,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )


def krea_cfg(
    conditional: mx.array,
    unconditional: mx.array,
    guidance: float,
) -> mx.array:
    """Apply Krea's conditional-offset CFG convention in activation dtype."""
    if tuple(conditional.shape) != tuple(unconditional.shape):
        raise ValueError("Krea conditional and unconditional predictions must match")
    if conditional.dtype != unconditional.dtype:
        raise ValueError("Krea conditional and unconditional dtypes must match")
    # Torch's Python scalar multiply uses FP32 opmath, then rounds the product
    # before the activation-dtype addition. Narrowing guidance itself changes
    # valid settings such as 3.7; widening the subtraction/addition also differs.
    difference = conditional - unconditional
    scaled = (difference.astype(mx.float32) * guidance).astype(conditional.dtype)
    return conditional + scaled


def _nonfinite_latent_message(dtype: mx.Dtype, steps: int) -> str:
    name = float_dtype_name(dtype)
    message = f"Krea 2 denoising produced a nonfinite latent at step {steps} in {name}."
    if dtype == mx.float16:
        return (
            f"{message} FP16 activations overflowed; select --transformer-dtype bfloat16 "
            "(or float32)."
        )
    return f"{message} Check the transformer weights, conditioning, and noise inputs."


def denoise_krea2(
    transformer: Krea2TransformerPort,
    positive: Krea2PreparedBranch,
    recipe: Krea2Recipe,
    *,
    negative: Krea2PreparedBranch | None = None,
    reporter: Reporter | None = None,
    checkpoint: Krea2Checkpoint | None = None,
) -> Krea2DenoiseResult:
    """Execute exact Krea flow time, branch, CFG, and Euler semantics."""
    if recipe.cfg_enabled != (negative is not None):
        raise ValueError("Krea CFG recipe and negative branch disagree")
    schedule = timestep_schedule(recipe)
    current = positive.image_tokens
    model_times = schedule.astype(current.dtype)
    mx.eval(schedule, current, model_times)
    # Transfer this tiny fixed schedule once, not two device scalars per step.
    time_values = cast(list[float], schedule.tolist())
    # Any noise-level adapter window resolves against this exact schedule.
    bind_live_adapter_sigmas(transformer, time_values)
    sink = reporter if reporter is not None else NullReporter()
    phase = "denoise"
    conditional_calls = 0
    unconditional_calls = 0
    sink.phase_start(phase, total=recipe.steps, unit="step")
    try:
        for index in range(recipe.steps):
            if checkpoint is not None:
                checkpoint()
            # Both CFG branches of one step share the step's adapter strengths.
            set_live_adapter_step(transformer, index)
            timestep = mx.broadcast_to(
                model_times[index],
                (current.shape[0],),
            )
            conditional = transformer(
                current,
                encoder_hidden_states=positive.encoder_hidden_states,
                timestep=timestep,
                position_ids=positive.position_ids,
                attention_mask=positive.attention_mask,
            ).sample
            conditional_calls += 1
            if negative is None:
                velocity = conditional
            else:
                unconditional = transformer(
                    current,
                    encoder_hidden_states=negative.encoder_hidden_states,
                    timestep=timestep,
                    position_ids=negative.position_ids,
                    attention_mask=negative.attention_mask,
                ).sample
                unconditional_calls += 1
                velocity = krea_cfg(conditional, unconditional, recipe.guidance)
            current = euler_velocity_step(
                current,
                velocity,
                time_values[index],
                time_values[index + 1],
            )
            mx.eval(current)
            check_adapter_weights(transformer, context=f"Krea 2 step {index + 1}", reporter=sink)
            if not mx.all(mx.isfinite(current)).item():
                raise Krea2DenoiseError(_nonfinite_latent_message(current.dtype, index + 1))
            # Advance only after evaluation so pace, ETA, and recorded intervals
            # include the conditional/CFG forwards and Euler step on the device.
            sink.phase_advance(phase)
    finally:
        sink.phase_end(phase)
    return Krea2DenoiseResult(
        packed=current,
        schedule=schedule,
        conditional_calls=conditional_calls,
        unconditional_calls=unconditional_calls,
    )


__all__ = [
    "Krea2Checkpoint",
    "Krea2DenoiseError",
    "Krea2DenoiseResult",
    "Krea2Geometry",
    "Krea2NoiseBatch",
    "Krea2PreparedBranch",
    "Krea2Recipe",
    "calculate_dynamic_mu",
    "create_initial_noise",
    "denoise_krea2",
    "krea_cfg",
    "pack_latents",
    "prepare_branch",
    "resolve_geometry",
    "resolve_recipe",
    "timestep_schedule",
    "unpack_latents",
]
