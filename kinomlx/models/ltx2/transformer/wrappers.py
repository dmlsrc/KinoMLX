"""Input and velocity-to-denoised wrappers for the LTX-2 transformer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import mlx.core as mx

import kinomlx._mlx_nn as nn

if TYPE_CHECKING:
    from ..cache import LoRAAdapterReceipt


@dataclass(frozen=True)
class Modality:
    """One patchified audio or video stream supplied to the joint model.

    ``timesteps`` remains per token because conditioning can freeze only part
    of a latent. ``sigma`` remains a separate batch scalar for prompt AdaLN and
    the other stream's A/V cross-attention gate.
    """

    latent: mx.array
    context: mx.array
    timesteps: mx.array
    sigma: mx.array
    positions: mx.array
    enabled: bool = True
    context_mask: mx.array | None = None
    attention_mask: mx.array | None = None
    keyframes_mask: mx.array | None = None
    positional_embeddings: tuple[mx.array, mx.array] | None = None
    cross_positional_embeddings: tuple[mx.array, mx.array] | None = None


class _VelocityModel(Protocol):
    @property
    def num_blocks(self) -> int: ...

    def __call__(
        self,
        video: Modality | None,
        audio: Modality | None = None,
    ) -> tuple[mx.array | None, mx.array | None]: ...


def _to_denoised(
    modality: Modality,
    velocity: mx.array | None,
) -> mx.array | None:
    if velocity is None:
        return None
    timestep = modality.timesteps
    if timestep.ndim == 1:
        timestep = timestep[:, None, None]
    elif timestep.ndim == 2:
        timestep = timestep[..., None]
    if timestep.ndim != 3:
        raise ValueError("modality timesteps must have shape (B,) or (B, T)")
    denoised = modality.latent.astype(mx.float32) - timestep.astype(mx.float32) * velocity.astype(
        mx.float32
    )
    return denoised.astype(modality.latent.dtype)


class X0Model(nn.Module):
    """Convert a velocity model's outputs to per-token denoised predictions."""

    def __init__(self, velocity_model: _VelocityModel) -> None:
        super().__init__()
        self.velocity_model = velocity_model
        # Receipt objects are orchestration metadata, not model parameters.
        object.__setattr__(self, "lora_receipts", ())
        self.lora_receipts: tuple[LoRAAdapterReceipt, ...]

    @property
    def num_blocks(self) -> int:
        return self.velocity_model.num_blocks

    def __call__(
        self,
        video: Modality | None,
        audio: Modality | None = None,
    ) -> tuple[mx.array | None, mx.array | None]:
        video_velocity, audio_velocity = self.velocity_model(video, audio)
        return (
            _to_denoised(video, video_velocity) if video is not None else None,
            _to_denoised(audio, audio_velocity) if audio is not None else None,
        )


__all__ = ["Modality", "X0Model"]
