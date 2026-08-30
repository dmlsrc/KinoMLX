"""Model-agnostic sampling primitives.

Currently:

- :mod:`kinomlx.samplers.steps` - deterministic and RF-ancestral Euler steps.
- :mod:`kinomlx.samplers.noise` - centralized normal-noise streams.
- :mod:`kinomlx.samplers.noisers` - noise-injection helpers.

Schedulers (dynamic sigma generation) are intentionally not shipped
in M2: KinoMLX's distilled-only happy path uses literal sigma
schedules (see :mod:`kinomlx.models.ltx2.sigmas`).  Adding a
generic scheduler module is deferred until a consumer actually
needs one.
"""

from kinomlx.samplers.noise import (
    NoiseStreamState,
    create_normal_noise_stream,
    noise_compatibility_profile,
)
from kinomlx.samplers.noisers import GaussianNoiser, SeededGaussianNoise
from kinomlx.samplers.steps import euler_ancestral_rf_step, euler_step

__all__ = [
    "GaussianNoiser",
    "NoiseStreamState",
    "SeededGaussianNoise",
    "create_normal_noise_stream",
    "euler_ancestral_rf_step",
    "euler_step",
    "noise_compatibility_profile",
]
