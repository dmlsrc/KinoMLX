"""Pure joint audio/video distilled denoising and Euler stepping."""

import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from itertools import pairwise
from typing import Literal, Protocol, cast

import mlx.core as mx

from kinomlx.adapters.validation import AdapterWeightError, check_adapter_weights
from kinomlx.errors import KinoMLXError
from kinomlx.reporting import NullReporter, Reporter
from kinomlx.samplers.steps import euler_ancestral_rf_step, euler_step

from .components import LTX2TransformerPort
from .sparse_attention import SparseAttentionStep
from .state import post_process_latent
from .transformer import Modality
from .transformer.execution import TransformerExecution
from .types import LatentState

DenoiseStepKind = Literal["deterministic-euler", "ancestral-rf"]
NoiseModality = Literal["video", "audio"]


class LTX2DenoiseError(KinoMLXError, RuntimeError):
    """A completed denoise step produced nonfinite active latents."""


class LTX2TransitionNoiseProvider(Protocol):
    """Supply one externally controlled ancestral-noise tensor."""

    def __call__(
        self,
        *,
        stage: int,
        transition: int,
        modality: NoiseModality,
        shape: tuple[int, ...],
        dtype: mx.Dtype,
    ) -> mx.array: ...


def _validated_transition_noise(
    provider: LTX2TransitionNoiseProvider,
    state: LatentState,
    *,
    stage: int,
    transition: int,
    modality: NoiseModality,
) -> mx.array:
    shape = tuple(state.latent.shape)
    noise = provider(
        stage=stage,
        transition=transition,
        modality=modality,
        shape=shape,
        dtype=state.latent.dtype,
    )
    if tuple(noise.shape) != shape:
        raise ValueError(
            f"{modality} ancestral noise shape {tuple(noise.shape)} does not match {shape}"
        )
    if noise.dtype != state.latent.dtype:
        raise ValueError(
            f"{modality} ancestral noise dtype {noise.dtype} does not match {state.latent.dtype}"
        )
    if not mx.all(mx.isfinite(noise)).item():
        raise ValueError(f"{modality} ancestral noise must be finite")
    return noise


def _advance_sample(
    sample: mx.array,
    denoised: mx.array,
    *,
    sigma: float,
    sigma_next: float,
    step_kind: DenoiseStepKind,
    noise: mx.array | None,
) -> mx.array:
    if step_kind == "deterministic-euler":
        if noise is not None:
            raise ValueError("deterministic Euler must not receive transition noise")
        return euler_step(sample, denoised, sigma, sigma_next)
    if step_kind == "ancestral-rf":
        return euler_ancestral_rf_step(
            sample,
            denoised,
            noise,
            sigma=sigma,
            sigma_next=sigma_next,
            eta=1.0,
            s_noise=1.0,
        )
    raise ValueError(f"unknown denoise step kind {step_kind!r}")


def _advance_state(
    state: LatentState,
    denoised: mx.array,
    *,
    sigma: float,
    sigma_next: float,
    step_kind: DenoiseStepKind,
    noise: mx.array | None,
) -> LatentState:
    """Advance one state and restore conditioned tokens after renoising."""
    latent = _advance_sample(
        state.latent,
        denoised,
        sigma=sigma,
        sigma_next=sigma_next,
        step_kind=step_kind,
        noise=noise,
    )
    if step_kind == "ancestral-rf" and noise is not None:
        latent = post_process_latent(latent, state)
    return replace(state, latent=latent)


def modality_from_state(
    state: LatentState,
    context: mx.array,
    sigma: float,
    *,
    enabled: bool = True,
    context_mask: mx.array | None = None,
    keyframes_mask: mx.array | None = None,
    sparse_attention: SparseAttentionStep | None = None,
) -> Modality:
    """Build one transformer modality with the scalar-timestep fast path."""
    sigma_tensor = mx.full(
        (state.latent.shape[0],),
        sigma,
        dtype=mx.float32,
    )
    if state.uniform_mask:
        timesteps = sigma_tensor
    else:
        mask = state.denoise_mask
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        timesteps = mask * sigma
    return Modality(
        latent=state.latent,
        context=context,
        timesteps=timesteps,
        sigma=sigma_tensor,
        positions=state.positions,
        enabled=enabled,
        context_mask=context_mask,
        keyframes_mask=keyframes_mask,
        sparse_attention=sparse_attention,
    )


def denoise_step(
    video_state: LatentState,
    audio_state: LatentState | None,
    *,
    sigma: float,
    sigma_next: float,
    transformer: LTX2TransformerPort,
    video_context: mx.array,
    audio_context: mx.array | None = None,
    video_context_mask: mx.array | None = None,
    audio_context_mask: mx.array | None = None,
    video_keyframes_mask: mx.array | None = None,
    video_enabled: bool = True,
    audio_enabled: bool = True,
    step_kind: DenoiseStepKind = "deterministic-euler",
    video_noise: mx.array | None = None,
    audio_noise: mx.array | None = None,
    reporter: Reporter | None = None,
    video_sparse_attention: SparseAttentionStep | None = None,
) -> tuple[LatentState, LatentState | None]:
    """Run one X0 transformer call and the selected Euler update."""
    video_modality = modality_from_state(
        video_state,
        video_context,
        sigma,
        enabled=video_enabled,
        context_mask=video_context_mask,
        keyframes_mask=video_keyframes_mask,
        sparse_attention=video_sparse_attention,
    )
    audio_modality = None
    if audio_state is not None:
        if audio_context is None:
            raise ValueError("audio context is required for an audio latent state")
        audio_modality = modality_from_state(
            audio_state,
            audio_context,
            sigma,
            enabled=audio_enabled,
            context_mask=audio_context_mask,
        )

    video_denoised, audio_denoised = transformer(video_modality, audio_modality)
    if video_enabled:
        if video_denoised is None:
            raise RuntimeError("transformer returned no video prediction")
        video_prediction = post_process_latent(video_denoised, video_state)
        video_state = _advance_state(
            video_state,
            video_prediction,
            sigma=sigma,
            sigma_next=sigma_next,
            step_kind=step_kind,
            noise=video_noise,
        )

    if audio_state is not None and audio_enabled:
        if audio_denoised is None:
            raise RuntimeError("transformer returned no audio prediction")
        audio_prediction = post_process_latent(audio_denoised, audio_state)
        audio_state = _advance_state(
            audio_state,
            audio_prediction,
            sigma=sigma,
            sigma_next=sigma_next,
            step_kind=step_kind,
            noise=audio_noise,
        )

    arrays = [video_state.latent]
    if audio_state is not None:
        arrays.append(audio_state.latent)
    mx.eval(*arrays)
    check_adapter_weights(
        transformer,
        context="LTX-2 denoise step",
        reporter=reporter or NullReporter(),
        active=TransformerExecution.resolve(
            video_present=video_modality.latent.size > 0,
            audio_present=audio_modality is not None and audio_modality.latent.size > 0,
            video_enabled=video_enabled,
            audio_enabled=audio_enabled,
            video_keyframes=video_keyframes_mask is not None,
        ).uses_weight,
    )
    active = []
    if video_enabled:
        active.append(("video", video_state.latent))
    if audio_enabled and audio_state is not None:
        active.append(("audio", audio_state.latent))
    if active:
        finite = cast(
            list[bool], mx.stack([mx.all(mx.isfinite(value)) for _, value in active]).tolist()
        )
        for (modality, value), valid in zip(active, finite, strict=True):
            if not valid:
                raise LTX2DenoiseError(
                    f"LTX-2 denoising produced a nonfinite {modality} latent in {value.dtype}."
                )
    return video_state, audio_state


def denoise_loop(
    video_state: LatentState,
    audio_state: LatentState | None,
    sigmas: Sequence[float],
    *,
    transformer: LTX2TransformerPort,
    video_context: mx.array,
    audio_context: mx.array | None = None,
    video_context_mask: mx.array | None = None,
    audio_context_mask: mx.array | None = None,
    video_keyframes_mask: mx.array | None = None,
    reporter: Reporter | None = None,
    phase: str = "denoise",
    callback: Callable[[int, int], None] | None = None,
    step_kind: DenoiseStepKind = "deterministic-euler",
    stage: int | None = None,
    noise_provider: LTX2TransitionNoiseProvider | None = None,
    video_sparse_attention: Sequence[SparseAttentionStep | None] = (),
    transformer_dtype: mx.Dtype | None = None,
) -> tuple[LatentState, LatentState | None]:
    """Run one distilled sigma schedule with a materialized step boundary.

    ``video_sparse_attention`` gives each step's sparse video self-attention, None for a
    dense step; an empty sequence keeps every step dense. ``transformer_dtype`` names the
    transformer's compute dtype for the message of a nonfinite step.
    """
    values = tuple(float(value) for value in sigmas)
    if len(values) < 2:
        raise ValueError("a denoise schedule must contain at least two sigmas")
    if any(not math.isfinite(current) or current <= 0.0 for current in values[:-1]):
        raise ValueError("every current denoise sigma must be positive and finite")
    if not math.isfinite(values[-1]):
        raise ValueError("the final denoise sigma must be finite")
    if any(current <= following for current, following in pairwise(values)):
        raise ValueError("denoise sigmas must be strictly descending")
    if step_kind not in {"deterministic-euler", "ancestral-rf"}:
        raise ValueError(f"unknown denoise step kind {step_kind!r}")
    if step_kind == "ancestral-rf" and stage is None:
        raise ValueError("ancestral RF needs the stage whose transition noise it draws")
    total = len(values) - 1
    if video_sparse_attention and len(video_sparse_attention) != total:
        raise ValueError(
            f"sparse attention lists {len(video_sparse_attention)} steps for a {total}-step "
            "schedule"
        )
    sparse_steps = tuple(video_sparse_attention) or (None,) * total
    sink = reporter if reporter is not None else NullReporter()
    sink.phase_start(phase, total=total, unit="step")
    try:
        for index, (sigma, sigma_next) in enumerate(pairwise(values), start=1):
            transition = index - 1
            video_noise = None
            audio_noise = None
            if step_kind == "ancestral-rf" and sigma_next > 0.0:
                if noise_provider is None:
                    raise ValueError("ancestral RF requires a transition noise provider")
                if stage is None:
                    raise RuntimeError("ancestral stage validation disappeared")
                video_noise = _validated_transition_noise(
                    noise_provider,
                    video_state,
                    stage=stage,
                    transition=transition,
                    modality="video",
                )
                if audio_state is not None:
                    audio_noise = _validated_transition_noise(
                        noise_provider,
                        audio_state,
                        stage=stage,
                        transition=transition,
                        modality="audio",
                    )
            try:
                video_state, audio_state = denoise_step(
                    video_state,
                    audio_state,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    transformer=transformer,
                    video_context=video_context,
                    audio_context=audio_context,
                    video_context_mask=video_context_mask,
                    audio_context_mask=audio_context_mask,
                    video_keyframes_mask=video_keyframes_mask,
                    step_kind=step_kind,
                    video_noise=video_noise,
                    audio_noise=audio_noise,
                    reporter=sink,
                    video_sparse_attention=sparse_steps[transition],
                )
            except (LTX2DenoiseError, AdapterWeightError) as error:
                location = f"stage {stage}" if stage is not None else phase
                message = f"{location}, step {index}: {error}"
                # MLX dtypes refuse equality with None rather than returning False.
                fp16 = transformer_dtype is not None and transformer_dtype == mx.float16
                if isinstance(error, LTX2DenoiseError) and fp16:
                    message += (
                        " The float16 transformer's activations may have overflowed;"
                        " select --transformer-dtype bfloat16 (or float32)."
                    )
                raise type(error)(message) from error
            sink.phase_advance(phase)
            if callback is not None:
                callback(index, total)
        return video_state, audio_state
    finally:
        sink.phase_end(phase)


__all__ = [
    "DenoiseStepKind",
    "LTX2DenoiseError",
    "LTX2TransitionNoiseProvider",
    "denoise_loop",
    "denoise_step",
    "modality_from_state",
]
