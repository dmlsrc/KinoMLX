"""Low-rank pairs, resolved loader profiles, and shared LoRA math.

A LoRA file holds, for each target weight ``W``, a low-rank pair
``A`` (``rank x in_features``) and ``B`` (``out_features x rank``)
such that the fused weight is ``W + scaling * (B @ A)``.  Most
authoring tools also write a scalar ``alpha`` per target encoding
the intrinsic scaling (``alpha / rank``); honoring it matches the
behavior of HF PEFT, Diffusers, and the original LoRA paper.

**Alpha values are normalized.**  A saved zero is the community
``alpha = rank`` convention and therefore means unit intrinsic scale.
Real-world community LoRAs also ship with broken alpha values such as
``inf``, ``nan``, negatives, or absurdly large values (millions+).
Those invalid values are dropped and the entry falls back to
``scaling = 1.0`` rather than multiplying weights by unsafe values.

Pattern matching here is *naming-convention*-aware but not
*model*-aware - it knows the standard
``lora_A``/``lora_B`` (HF / PEFT) and
``lora_down``/``lora_up`` (A1111 / ComfyUI / Kohya) forms, with or
without the trailing ``.weight``, but does not transform the base
prefix.  Per-model prefix conventions (e.g. ``diffusion_model.``
in SD-pipeline community LoRAs) live in
``kinomlx/models/<name>/adapters/``.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

# Suffix pairs to try when matching a base key against LoRA keys.
# Listed in priority order; first hit wins.
_PAIR_SUFFIXES: tuple[tuple[str, str], ...] = (
    (".lora_A.weight", ".lora_B.weight"),
    (".lora_down.weight", ".lora_up.weight"),
    (".lora_A", ".lora_B"),
    (".lora_down", ".lora_up"),
)

# Upper bound on plausible LoRA alpha.  Values above this - or
# non-finite, or ``<= 0`` - are treated as corrupt and ignored.
# Real LoRAs land in the 1-256 range; anything above ~10^4 is
# almost certainly a serialization bug (endianness, uint-as-int,
# garbage memory).  1e6 leaves generous headroom while still
# catching the obvious failure modes.
_MAX_REASONABLE_ALPHA: float = 1e6


@dataclass(frozen=True)
class LoRAConfig:
    """User-supplied LoRA with total-strength and exclusion controls.

    ``strength`` is the user-applied scaling on top of the LoRA's
    intrinsic ``alpha/rank``. It must be finite; negative and unusually
    large finite values remain available for adapters whose authors call
    for them. The model-specific loader interprets ``exclude`` category names.
    """

    path: Path
    strength: float = 1.0
    exclude: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Canonical profiles compare resolved file identity, not path spelling.
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())
        if not math.isfinite(self.strength):
            raise ValueError(f"LoRA strength must be finite, got {self.strength}")
        exclusions = tuple(self.exclude)
        if any(not isinstance(item, str) or not item for item in exclusions):
            raise ValueError("LoRA exclude categories must be non-empty strings")
        object.__setattr__(self, "exclude", tuple(sorted(set(exclusions))))


type LoRAProfile = tuple[LoRAConfig, ...]


@dataclass(frozen=True)
class LoRAEntry:
    """One matched LoRA delta - the low-rank pair plus optional alpha.

    ``scaling`` is the file-intrinsic multiplier: ``alpha / rank``
    when alpha is present and nonzero, else ``1.0`` (no scaling override).
    Community trainers use a saved zero alpha to mean ``alpha = rank``.
    The caller multiplies by the user's strength to get the final
    delta.
    """

    a: mx.array
    b: mx.array
    alpha: float | None = None

    @property
    def rank(self) -> int:
        return self.a.shape[0]

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank if self.alpha is not None and self.alpha != 0.0 else 1.0


def iter_lora_entries(
    lora_weights: dict[str, mx.array],
) -> Iterator[tuple[str, LoRAEntry]]:
    """Yield each complete LoRA target as ``(base_prefix, entry)``.

    Targets follow the declared suffix priority; when a
    file redundantly contains more than one supported convention for one base,
    only the highest-priority pair is yielded.
    """
    recognized: set[str] = set()
    complete: set[str] = set()
    for suffix_a, suffix_b in _PAIR_SUFFIXES:
        for key in lora_weights:
            if key.endswith(suffix_a):
                prefix = key[: -len(suffix_a)]
                recognized.add(prefix)
                if f"{prefix}{suffix_b}" in lora_weights:
                    complete.add(prefix)
            elif key.endswith(suffix_b):
                recognized.add(key[: -len(suffix_b)])
    malformed = sorted(recognized - complete)
    if malformed:
        raise ValueError(
            "LoRA contains unmatched A/B factors for "
            f"{len(malformed)} targets (first: {malformed[0]!r})"
        )

    seen: set[str] = set()
    for suffix_a, suffix_b in _PAIR_SUFFIXES:
        for key_a in lora_weights:
            if not key_a.endswith(suffix_a):
                continue
            prefix = key_a[: -len(suffix_a)]
            if prefix in seen:
                continue
            key_b = f"{prefix}{suffix_b}"
            if key_b not in lora_weights:
                continue
            seen.add(prefix)
            yield (
                prefix,
                LoRAEntry(
                    a=lora_weights[key_a],
                    b=lora_weights[key_b],
                    alpha=_read_alpha(lora_weights, prefix),
                ),
            )


def _read_alpha(lora_weights: dict[str, mx.array], prefix: str) -> float | None:
    """Look up ``<prefix>.alpha`` as a finite positive float, or return ``None``.

    Returns ``None`` for any of:
    - the alpha key isn't present,
    - the stored value can't be coerced to ``float``,
    - the value is ``nan``, ``inf``, or ``-inf``,
    - the value is ``<= 0`` or ``> _MAX_REASONABLE_ALPHA``.

    Zero deliberately normalizes to ``None`` because both mean unit intrinsic
    scale. Invalid values use the same safe fallback instead of making weight
    arithmetic nonfinite.
    """
    alpha_arr = lora_weights.get(f"{prefix}.alpha")
    if alpha_arr is None:
        return None
    try:
        # Stored as a 0-d or 1-d tensor; ``float(...)`` works for both.
        alpha = float(alpha_arr)
    except TypeError, ValueError:
        return None
    if not math.isfinite(alpha):
        return None
    if not (0 < alpha <= _MAX_REASONABLE_ALPHA):
        return None
    return alpha


def validated_lora_scale(entry: LoRAEntry, strength: float) -> float:
    """Validate a matrix pair and return its finite effective scale."""
    if entry.a.ndim != 2 or entry.b.ndim != 2:
        raise ValueError(
            "LoRA factors must both be 2D matrices; got "
            f"A={tuple(entry.a.shape)}, B={tuple(entry.b.shape)}"
        )
    if entry.a.shape[0] <= 0 or entry.b.shape[1] != entry.a.shape[0]:
        raise ValueError(
            "LoRA factors have incompatible ranks: "
            f"A={tuple(entry.a.shape)}, B={tuple(entry.b.shape)}"
        )
    try:
        effective_scale = float(strength) * float(entry.scaling)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid LoRA strength or scaling: {strength!r}") from exc
    if not math.isfinite(effective_scale):
        raise ValueError(f"LoRA effective scale must be finite, got {effective_scale!r}")
    return effective_scale


def compute_delta(entry: LoRAEntry, strength: float = 1.0) -> mx.array:
    """Compute the additive LoRA delta in fp32.

    ``delta = strength * entry.scaling * (B @ A)`` where ``scaling``
    is ``alpha/rank`` if the entry has alpha, else ``1.0``.

    Returns an fp32 array regardless of input dtype; the caller
    is responsible for casting back to the base weight's dtype
    after summing into the base.
    """
    effective_scale = validated_lora_scale(entry, strength)
    a_f32 = entry.a.astype(mx.float32)
    b_f32 = entry.b.astype(mx.float32)
    return mx.matmul(b_f32, a_f32) * effective_scale
