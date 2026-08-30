"""Central normal-noise streams for reproducible MLX inference.

The Torch-MPS backend ports the Philox and Box-Muller path used by PyTorch
2.13.0 at commit ``cf30153c4c131c8164ee7798e5022d810682e2cb``. The source
semantics come from ``c10/metal/random.h`` and
``aten/src/ATen/native/mps/kernels/Distributions.metal`` under PyTorch's
BSD-style license; see ``THIRD_PARTY_LICENSES.md``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

import mlx.core as mx

from kinomlx.kernels._typing import MetalKernel
from kinomlx.types import (
    DEFAULT_NOISE_BACKEND,
    NOISE_BACKEND_CHOICES,
    NoiseBackend,
)

TORCH_MPS_COMPATIBILITY_PROFILE = "pytorch-2.13.0-mps"
MLX_COMPATIBILITY_PROFILE = "mlx-native"

_TORCH_RANDOM_HEADER = r"""
inline uint2 torch_splitlong(ulong value) {
    return uint2(value >> 32, value & 0xfffffffful);
}

inline uint2 torch_mulhilo(uint a, uint b) {
    ulong product = static_cast<ulong>(a) * b;
    return torch_splitlong(product);
}

inline uint4 torch_single_round(uint4 counter, uint2 key) {
    constexpr uint PHILOX_SA = 0xD2511F53;
    constexpr uint PHILOX_SB = 0xCD9E8D57;
    uint2 result_0 = torch_mulhilo(PHILOX_SA, counter.x);
    uint2 result_1 = torch_mulhilo(PHILOX_SB, counter.z);
    return uint4(
        result_1.x ^ counter.y ^ key.x,
        result_1.y,
        result_0.x ^ counter.w ^ key.y,
        result_0.y
    );
}

inline uint4 torch_multiple_rounds(uint4 counter, uint2 key, uint rounds) {
    constexpr uint2 PHILOX_10 = uint2(0x9E3779B9, 0xBB67AE85);
    for (uint round = 0; round < rounds - 1; ++round) {
        counter = torch_single_round(counter, key);
        key += PHILOX_10;
    }
    return counter;
}

inline uint4 torch_philox(ulong seed, ulong index) {
    uint4 counter = uint4(0);
    counter.zw = torch_splitlong(index);
    return torch_multiple_rounds(counter, torch_splitlong(seed), 10);
}

inline float torch_uniform(uint value) {
    constexpr float SCALE = 4.6566127342e-10f;
    return static_cast<float>(value & 0x7fffffffu) * SCALE;
}

inline float2 torch_box_muller(uint2 raw) {
    constexpr float EPSILON = metal::numeric_limits<float>::epsilon();
    float uniform_1 = metal::max(torch_uniform(raw.x), EPSILON);
    float uniform_2 = torch_uniform(raw.y);
    float radius = metal::precise::sqrt(
        -2.0f * metal::precise::log(uniform_1)
    );
    float cosine;
    float sine = metal::precise::sincos(2.0f * M_PI_F * uniform_2, cosine);
    return radius * float2(cosine, sine);
}
"""

_TORCH_NORMAL_SOURCE = r"""
uint thread_index = thread_position_in_grid.x;
uint base = thread_index * 4;
if (base >= N) {
    return;
}

ulong seed = state[0];
ulong offset = state[1];
uint4 raw = torch_philox(seed, offset + thread_index);
float2 normal_a = torch_box_muller(raw.xy);
float2 normal_b = torch_box_muller(raw.zw);
float values[4] = {normal_a.x, normal_a.y, normal_b.x, normal_b.y};
uint count = metal::min(4u, static_cast<uint>(N) - base);
for (uint index = 0; index < count; ++index) {
    output[base + index] = static_cast<T>(values[index]);
}
"""

_TORCH_MPS_KERNEL: MetalKernel | None = None


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("noise seed must be an unsigned 64-bit integer")
    return seed


def _validate_shape(shape: tuple[int, ...]) -> int:
    for dimension in shape:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
            raise ValueError("noise shape dimensions must be non-negative integers")
    return math.prod(shape)


def _validate_dtype(dtype: mx.Dtype) -> None:
    if dtype not in {mx.float16, mx.bfloat16, mx.float32}:
        raise ValueError(f"normal noise requires float16, bfloat16, or float32, got {dtype}")


def _validate_backend(backend: str) -> NoiseBackend:
    if backend not in NOISE_BACKEND_CHOICES:
        choices = ", ".join(NOISE_BACKEND_CHOICES)
        raise ValueError(f"noise backend must be one of: {choices}")
    return cast(NoiseBackend, backend)


def noise_compatibility_profile(backend: NoiseBackend) -> str:
    """Return the stable implementation identity recorded in run receipts."""
    selected = _validate_backend(backend)
    if selected == "torch-mps":
        return TORCH_MPS_COMPATIBILITY_PROFILE
    return MLX_COMPATIBILITY_PROFILE


@dataclass(frozen=True)
class NoiseStreamState:
    """Serializable position of one ordered normal-noise stream."""

    backend: NoiseBackend
    compatibility_profile: str
    seed: int
    draws: int
    elements: int
    philox_blocks: int

    def __post_init__(self) -> None:
        _validate_backend(self.backend)
        _validate_seed(self.seed)
        if not self.compatibility_profile:
            raise ValueError("noise compatibility profile must not be empty")
        for name in ("draws", "elements", "philox_blocks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"noise {name} must be a non-negative integer")
        if self.philox_blocks >= 2**64:
            raise ValueError("noise Philox position must be smaller than 2**64")

    def to_metadata(self) -> dict[str, object]:
        """Return a stable JSON-ready run receipt."""
        return {
            "backend": self.backend,
            "compatibility_profile": self.compatibility_profile,
            "seed": self.seed,
            "draws": self.draws,
            "elements": self.elements,
            "philox_blocks": self.philox_blocks,
            "philox_block_width": 4,
        }

    def to_artifact_metadata(
        self,
        *,
        prefix: str = "initial_noise_",
    ) -> tuple[tuple[str, str], ...]:
        """Encode the resumable position into safetensors string metadata."""
        return (
            (f"{prefix}backend", self.backend),
            (f"{prefix}compatibility_profile", self.compatibility_profile),
            (f"{prefix}seed", str(self.seed)),
            (f"{prefix}draws", str(self.draws)),
            (f"{prefix}elements", str(self.elements)),
            (f"{prefix}philox_blocks", str(self.philox_blocks)),
        )

    @classmethod
    def from_artifact_metadata(
        cls,
        metadata: Mapping[str, str],
        *,
        prefix: str = "initial_noise_",
    ) -> NoiseStreamState | None:
        """Decode a position, returning ``None`` for legacy artifacts."""
        backend_key = f"{prefix}backend"
        if backend_key not in metadata:
            return None
        required = {
            "backend",
            "compatibility_profile",
            "seed",
            "draws",
            "elements",
            "philox_blocks",
        }
        missing = sorted(name for name in required if f"{prefix}{name}" not in metadata)
        if missing:
            raise ValueError(
                "latent artifact noise state is incomplete "
                f"(first missing field: {prefix}{missing[0]})"
            )
        try:
            backend = _validate_backend(metadata[backend_key])
            return cls(
                backend=backend,
                compatibility_profile=metadata[f"{prefix}compatibility_profile"],
                seed=int(metadata[f"{prefix}seed"]),
                draws=int(metadata[f"{prefix}draws"]),
                elements=int(metadata[f"{prefix}elements"]),
                philox_blocks=int(metadata[f"{prefix}philox_blocks"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("latent artifact noise state is invalid") from exc


@dataclass
class _NoisePosition:
    draws: int = 0
    elements: int = 0
    philox_blocks: int = 0

    @classmethod
    def from_state(cls, state: NoiseStreamState | None) -> _NoisePosition:
        if state is None:
            return cls()
        return cls(
            draws=state.draws,
            elements=state.elements,
            philox_blocks=state.philox_blocks,
        )

    def consume(self, elements: int) -> None:
        if elements == 0:
            return
        blocks = (elements + 3) // 4
        if self.philox_blocks + blocks >= 2**64:
            raise ValueError("normal-noise stream exhausted its unsigned 64-bit offset")
        self.draws += 1
        self.elements += elements
        self.philox_blocks += blocks


class NormalNoiseStream(Protocol):
    """Ordered normal draws with backend-independent resume accounting."""

    @property
    def state(self) -> NoiseStreamState: ...

    def normal(self, shape: tuple[int, ...], dtype: mx.Dtype) -> mx.array: ...

    def advance(self, shape: tuple[int, ...]) -> None: ...


class MLXNormalNoiseStream:
    """Legacy MLX-native keyed normal stream."""

    def __init__(self, seed: int, *, position: NoiseStreamState | None = None) -> None:
        self._seed = _validate_seed(seed)
        self._position = _NoisePosition.from_state(position)
        self._key = mx.random.key(self._seed)
        for _ in range(self._position.draws):
            self._key, _subkey = mx.random.split(self._key)

    @property
    def state(self) -> NoiseStreamState:
        return NoiseStreamState(
            backend="mlx",
            compatibility_profile=MLX_COMPATIBILITY_PROFILE,
            seed=self._seed,
            draws=self._position.draws,
            elements=self._position.elements,
            philox_blocks=self._position.philox_blocks,
        )

    def normal(self, shape: tuple[int, ...], dtype: mx.Dtype) -> mx.array:
        count = _validate_shape(shape)
        _validate_dtype(dtype)
        if count == 0:
            return mx.zeros(shape, dtype=dtype)
        self._key, subkey = mx.random.split(self._key)
        value = mx.random.normal(shape=shape, dtype=dtype, key=subkey)
        self._position.consume(count)
        return value

    def advance(self, shape: tuple[int, ...]) -> None:
        count = _validate_shape(shape)
        if count == 0:
            return
        self._key, _subkey = mx.random.split(self._key)
        self._position.consume(count)


def _torch_mps_kernel() -> MetalKernel:
    global _TORCH_MPS_KERNEL
    if _TORCH_MPS_KERNEL is None:
        _TORCH_MPS_KERNEL = cast(
            MetalKernel,
            mx.fast.metal_kernel(
                name="kinomlx_torch_mps_normal_2_13_0",
                input_names=["state"],
                output_names=["output"],
                header=_TORCH_RANDOM_HEADER,
                source=_TORCH_NORMAL_SOURCE,
                compile_options={"math_mode": "safe"},
            ),
        )
    return _TORCH_MPS_KERNEL


class TorchMPSNormalNoiseStream:
    """MLX Metal implementation of the pinned PyTorch MPS normal stream."""

    def __init__(self, seed: int, *, position: NoiseStreamState | None = None) -> None:
        self._seed = _validate_seed(seed)
        self._position = _NoisePosition.from_state(position)

    @property
    def state(self) -> NoiseStreamState:
        return NoiseStreamState(
            backend="torch-mps",
            compatibility_profile=TORCH_MPS_COMPATIBILITY_PROFILE,
            seed=self._seed,
            draws=self._position.draws,
            elements=self._position.elements,
            philox_blocks=self._position.philox_blocks,
        )

    def normal(self, shape: tuple[int, ...], dtype: mx.Dtype) -> mx.array:
        count = _validate_shape(shape)
        _validate_dtype(dtype)
        if count == 0:
            return mx.zeros(shape, dtype=dtype)
        blocks = (count + 3) // 4
        if self._position.philox_blocks + blocks >= 2**64:
            raise ValueError("normal-noise stream exhausted its unsigned 64-bit offset")
        state = mx.array(
            [self._seed, self._position.philox_blocks],
            dtype=mx.uint64,
        )
        value = _torch_mps_kernel()(
            inputs=[state],
            output_shapes=[shape],
            output_dtypes=[dtype],
            grid=(blocks, 1, 1),
            threadgroup=(min(256, blocks), 1, 1),
            template=[("N", count), ("T", dtype)],
            stream=mx.gpu,
        )[0]
        self._position.consume(count)
        return value

    def advance(self, shape: tuple[int, ...]) -> None:
        count = _validate_shape(shape)
        self._position.consume(count)


def create_normal_noise_stream(
    seed: int,
    *,
    backend: NoiseBackend = DEFAULT_NOISE_BACKEND,
    position: NoiseStreamState | None = None,
) -> NormalNoiseStream:
    """Create one seeded stream, optionally at a saved draw boundary."""
    selected = _validate_backend(backend)
    if selected == "torch-mps":
        return TorchMPSNormalNoiseStream(seed, position=position)
    return MLXNormalNoiseStream(seed, position=position)


__all__ = [
    "MLXNormalNoiseStream",
    "NormalNoiseStream",
    "NoiseStreamState",
    "TORCH_MPS_COMPATIBILITY_PROFILE",
    "TorchMPSNormalNoiseStream",
    "create_normal_noise_stream",
    "noise_compatibility_profile",
]
