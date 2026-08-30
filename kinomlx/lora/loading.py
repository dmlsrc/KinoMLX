"""Load LoRA safetensors files and find low-rank pairs.

A LoRA file holds, for each target weight ``W``, a low-rank pair
``A`` (``rank x in_features``) and ``B`` (``out_features x rank``)
such that the fused weight is ``W + scaling * (B @ A)``.  Most
authoring tools also write a scalar ``alpha`` per target encoding
the intrinsic scaling (``alpha / rank``); honoring it matches the
behavior of HF PEFT, Diffusers, and the original LoRA paper.

**Alpha values are filtered.**  Real-world community LoRAs ship
with broken alpha values surprisingly often - ``inf``, ``nan``,
``0``, negative, or absurdly large (millions+).  Any alpha that
isn't a finite positive number in ``(0, 1e6]`` is dropped and the
entry falls back to ``scaling = 1.0``.  Better to apply the LoRA
at the rank-default scaling than to multiply weights by infinity.

Pattern matching here is *naming-convention*-aware but not
*model*-aware - it knows the standard
``lora_A``/``lora_B`` (HF / PEFT) and
``lora_down``/``lora_up`` (A1111 / ComfyUI / Kohya) forms, with or
without the trailing ``.weight``, but does not transform the base
prefix.  Per-model prefix conventions (e.g. ``diffusion_model.``
in SD-pipeline community LoRAs) live in
``kinomlx/models/<name>/cache/keys/``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
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
    """User-supplied LoRA with total, per-stage, and exclusion controls.

    ``strength`` is the user-applied scaling on top of the LoRA's
    intrinsic ``alpha/rank``. It must be finite; negative and unusually
    large finite values remain available for adapters whose authors call
    for them. Stage strengths fall back to ``strength``. The model-specific
    loader interprets ``exclude`` category names.
    """

    path: Path
    strength: float = 1.0
    stage_1_strength: float | None = None
    stage_2_strength: float | None = None
    exclude: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Canonical profiles compare resolved file identity, not path spelling.
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())
        for name, value in (
            ("strength", self.strength),
            ("stage_1_strength", self.stage_1_strength),
            ("stage_2_strength", self.stage_2_strength),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"LoRA {name} must be finite, got {value}")
        exclusions = tuple(self.exclude)
        if any(not isinstance(item, str) or not item for item in exclusions):
            raise ValueError("LoRA exclude categories must be non-empty strings")
        object.__setattr__(self, "exclude", tuple(sorted(set(exclusions))))

    def has_stage_strengths(self) -> bool:
        """Return whether either stage overrides the scalar strength."""
        return self.stage_1_strength is not None or self.stage_2_strength is not None

    def strength_for_stage(self, stage: int) -> float:
        """Resolve this adapter's total strength for stage 1 or stage 2."""
        if stage == 1:
            return self.strength if self.stage_1_strength is None else self.stage_1_strength
        if stage == 2:
            return self.strength if self.stage_2_strength is None else self.stage_2_strength
        raise ValueError(f"LoRA stage must be 1 or 2, got {stage}")

    def with_strength(self, strength: float) -> LoRAConfig:
        """Return a resolved single-stage config preserving its knockouts."""
        return LoRAConfig(
            path=self.path,
            strength=strength,
            exclude=self.exclude,
        )


type LoRAProfile = tuple[LoRAConfig, ...]


def lora_configs_have_stage_strengths(
    configs: Iterable[LoRAConfig] | None,
) -> bool:
    """Return whether any adapter declares a stage-specific strength."""
    return any(config.has_stage_strengths() for config in configs or ())


def lora_configs_for_stage(
    configs: Iterable[LoRAConfig] | None,
    stage: int,
) -> list[LoRAConfig]:
    """Resolve total adapter strengths for one stage and omit zero totals."""
    resolved: list[LoRAConfig] = []
    for config in configs or ():
        strength = config.strength_for_stage(stage)
        if strength != 0.0:
            resolved.append(config.with_strength(strength))
    return resolved


def format_lora_stage_scale_lines(
    configs: Iterable[LoRAConfig] | None,
    stage: int,
    *,
    include_unchanged: bool = False,
) -> list[str]:
    """Return deterministic human-readable total-strength lines."""
    lines: list[str] = []
    for config in configs or ():
        total = config.strength_for_stage(stage)
        if total == 0.0 and not include_unchanged:
            continue
        lines.append(f"    {config.path.name}: total={total:.4f}")
    return lines


@dataclass(frozen=True)
class LoRAEntry:
    """One matched LoRA delta - the low-rank pair plus optional alpha.

    ``scaling`` is the file-intrinsic multiplier: ``alpha / rank``
    when alpha is present, else ``1.0`` (no scaling override).
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
        return (self.alpha / self.rank) if self.alpha is not None else 1.0


def find_lora_entry(
    lora_weights: dict[str, mx.array],
    base_key: str,
) -> LoRAEntry | None:
    """Find the LoRA A/B (and optional alpha) for a base weight key.

    ``base_key`` may end with ``.weight`` or not; both are tried.
    Pattern matching only - no prefix transformation.  If the LoRA
    file uses a different base prefix than the model's weight keys,
    the caller normalizes one to the other (a per-model concern).

    Returns ``None`` when no pair matches.
    """
    prefix = base_key.removesuffix(".weight")
    for suffix_a, suffix_b in _PAIR_SUFFIXES:
        key_a = f"{prefix}{suffix_a}"
        key_b = f"{prefix}{suffix_b}"
        if key_a in lora_weights and key_b in lora_weights:
            alpha = _read_alpha(lora_weights, prefix)
            return LoRAEntry(a=lora_weights[key_a], b=lora_weights[key_b], alpha=alpha)
    return None


def iter_lora_entries(
    lora_weights: dict[str, mx.array],
) -> Iterator[tuple[str, LoRAEntry]]:
    """Yield each complete LoRA target as ``(base_prefix, entry)``.

    Targets follow the same suffix priority as :func:`find_lora_entry`; when a
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

    Returning ``None`` rather than raising lets the caller fall back
    to ``scaling = 1.0`` - applying the LoRA at the default scaling
    is almost always better than refusing to apply it at all.
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
