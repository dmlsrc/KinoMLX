"""KinoMLX STEEL attention backend for supported LTX-2.3 shapes.

The default path uses BF16-only MLX STEEL subsets specialized for the no-mask
LTX-2.3 hot path.  Unsupported shapes and FP16 calls fall back to stock MLX
SDPA in the caller.
"""

from __future__ import annotations

import atexit
import logging
from typing import NamedTuple, cast

import mlx.core as mx

from ._typing import MetalKernel

_LOG = logging.getLogger(__name__)


class _SteelConfig(NamedTuple):
    bq: int
    bk: int
    wm: int
    k_pad: int
    kv_threads: int
    name: str
    reduce_all_cols: bool = False
    scale_in_exp: bool = False


_D128_CONFIG = _SteelConfig(
    80,
    40,
    10,
    2,
    320,
    "ltx_steel_attention_bq80_bk40_q8k2v8_allactive_reduceallcols_scalefold",
    True,
    True,
)
_D64_CONFIGS = {
    "bk32": _SteelConfig(64, 32, 8, 8, 256, "ltx_steel_attention_bq64_bk32"),
    "bk32_q8k4": _SteelConfig(64, 32, 8, 4, 256, "ltx_steel_attention_bq64_bk32_q8k4v8"),
    "bk24_q8k2_scalefold": _SteelConfig(
        64,
        24,
        8,
        2,
        192,
        "ltx_steel_attention_bq64_bk24_q8k2v8_reduceallcols_scalefold",
        True,
        True,
    ),
}
_SUPPORTED_DIMS = {64, 128}
_SUPPORTED_HEADS = 32
_PROBE_ACTIVE = False
_PROBE_COUNTS = {
    "hit_d64": 0,
    "hit_d128": 0,
    "fallback": 0,
}
_PROBE_D64_CONFIGS: dict[str, int] = {}
_PROBE_REASONS: dict[str, int] = {}
_PROBE_SAMPLES: dict[str, str] = {}
_PROBE_SCOPES: set[str] = set()
_KERNELS: dict[tuple[int, str], MetalKernel] = {}


def _shape_sample(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask: mx.array | None,
) -> str:
    mask_shape = None if mask is None else tuple(mask.shape)
    return f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} mask={mask_shape}"


def _probe_hit(
    bd: int,
    d64_config: str,
    *,
    enabled: bool,
    compiled: bool,
) -> None:
    if not enabled:
        return
    global _PROBE_ACTIVE
    _PROBE_ACTIVE = True
    _PROBE_SCOPES.add("compiled_traces" if compiled else "eager_calls")
    _PROBE_COUNTS[f"hit_d{bd}"] += 1
    if bd == 64:
        _PROBE_D64_CONFIGS[d64_config] = _PROBE_D64_CONFIGS.get(d64_config, 0) + 1


def _probe_fallback(
    reason: str,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask: mx.array | None,
    *,
    enabled: bool,
    compiled: bool,
) -> None:
    if not enabled:
        return
    global _PROBE_ACTIVE
    _PROBE_ACTIVE = True
    _PROBE_SCOPES.add("compiled_traces" if compiled else "eager_calls")
    _PROBE_COUNTS["fallback"] += 1
    _PROBE_REASONS[reason] = _PROBE_REASONS.get(reason, 0) + 1
    _PROBE_SAMPLES.setdefault(reason, _shape_sample(q, k, v, mask))


def _probe_scope() -> tuple[str, str]:
    if {"compiled_traces"} == _PROBE_SCOPES:
        return "process_compiled_traces", "compiled_trace"
    if {"eager_calls"} == _PROBE_SCOPES:
        return "process_eager_calls", "attention_call"
    if _PROBE_SCOPES:
        return "process_mixed_compiled_traces_and_eager_calls", "mixed"
    return "process_no_observations", "none"


def steel_attention_summary() -> dict[str, object]:
    """Return process-local selector observations with their counting unit.

    Compiled wrappers execute the Python probe while tracing a shape signature;
    eager wrappers execute it once per attention call. Cached compiled traces
    may be reused without incrementing the counters again.
    """
    scope, counter_unit = _probe_scope()
    return {
        "enabled": _PROBE_ACTIVE,
        "scope": scope,
        "counter_unit": counter_unit,
        "hit_d128": _PROBE_COUNTS["hit_d128"],
        "hit_d64": _PROBE_COUNTS["hit_d64"],
        "fallback": _PROBE_COUNTS["fallback"],
        "d64_configs": dict(sorted(_PROBE_D64_CONFIGS.items())),
        "fallback_reasons": dict(sorted(_PROBE_REASONS.items())),
        "fallback_samples": dict(sorted(_PROBE_SAMPLES.items())),
    }


def reset_steel_attention_stats() -> None:
    """Reset selection counters without rebuilding cached kernels."""
    global _PROBE_ACTIVE
    _PROBE_ACTIVE = False
    for name in _PROBE_COUNTS:
        _PROBE_COUNTS[name] = 0
    _PROBE_D64_CONFIGS.clear()
    _PROBE_REASONS.clear()
    _PROBE_SAMPLES.clear()
    _PROBE_SCOPES.clear()


def _log_probe_summary() -> None:
    if not _PROBE_ACTIVE:
        return
    total = sum(_PROBE_COUNTS.values())
    if total == 0:
        return
    _LOG.info("STEEL attention selection: %s", steel_attention_summary())


atexit.register(_log_probe_summary)


def _d64_config(q_len: int, k_len: int) -> str:
    if q_len == k_len:
        return "bk32"
    if q_len < k_len:
        return "bk24_q8k2_scalefold"
    return "bk32_q8k4"


def _config(dim: int, d64_config: str = "") -> _SteelConfig:
    if dim == 128:
        return _D128_CONFIG
    return _D64_CONFIGS[d64_config]


def _kernel(dim: int, d64_config: str = "") -> MetalKernel:
    key = (dim, d64_config) if dim == 64 else (dim, "")
    if key not in _KERNELS:
        from ._steel_attention_ltx import HEADER, SOURCE

        _KERNELS[key] = cast(
            MetalKernel,
            mx.fast.metal_kernel(
                name=_config(dim, d64_config).name,
                input_names=["Q", "K", "V"],
                output_names=["O"],
                source=SOURCE,
                header=HEADER,
                ensure_row_contiguous=False,
            ),
        )
    return _KERNELS[key]


def _tile_config(dim: int, d64_config: str = "") -> tuple[int, int, int]:
    cfg = _config(dim, d64_config)
    return cfg.bq, cfg.bk, cfg.wm


def _template(
    dim: int,
    align_q: bool,
    align_k: bool,
    d64_config: str = "",
) -> list[tuple[str, bool | int | mx.Dtype]]:
    cfg = _config(dim, d64_config)
    kv_all_active = cfg.kv_threads == cfg.wm * 32
    return [
        # H controls the Metal output stride, so the selector gate and kernel
        # specialization share one source of truth.
        ("H", _SUPPORTED_HEADS),
        ("BD", dim),
        ("BQ", cfg.bq),
        ("BK", cfg.bk),
        ("WM", cfg.wm),
        ("Q_PAD", 8),
        ("K_PAD", cfg.k_pad),
        ("V_PAD", 8),
        ("Q_ACTIVE_THREADS", cfg.wm * 32),
        ("K_ACTIVE_THREADS", cfg.kv_threads),
        ("V_ACTIVE_THREADS", cfg.kv_threads),
        ("Q_LOADS_ALL_ACTIVE", True),
        ("K_LOADS_ALL_ACTIVE", kv_all_active),
        ("V_LOADS_ALL_ACTIVE", kv_all_active),
        ("Q_EXACT_TILES", False),
        ("K_EXACT_TILES", False),
        ("Q_FULL_TILES_CONST", 0),
        ("K_FULL_TILES_CONST", 0),
        ("Q_REM_CONST", 0),
        ("K_REM_CONST", 0),
        ("SkipUnitFactor", False),
        ("ReduceAllCols", cfg.reduce_all_cols),
        ("ScaleInExp", cfg.scale_in_exp),
        ("AlignQ", align_q),
        ("AlignK", align_k),
    ]


def _scale_supported(scale: float | None, dim: int) -> bool:
    expected = 1.0 / (dim**0.5)
    return scale is None or abs(float(scale) - expected) < 1e-7


def _select_config(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float | None,
    mask: mx.array | None,
    *,
    enable_d64: bool = True,
    inputs_last_axis_contiguous: bool = True,
) -> tuple[int | None, str]:
    if mask is not None:
        return None, "mask"
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return None, "ndim"
    if q.dtype != mx.bfloat16:
        return None, "dtype_bf16"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return None, "dtype_mismatch"
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        return None, "batch"
    if (
        q.shape[1] != _SUPPORTED_HEADS
        or k.shape[1] != _SUPPORTED_HEADS
        or v.shape[1] != _SUPPORTED_HEADS
    ):
        return None, "heads"
    if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
        return None, "dim_mismatch"
    bd = q.shape[-1]
    if bd not in _SUPPORTED_DIMS:
        return None, "dim"
    if bd == 64 and not enable_d64:
        return None, "d64_disabled"
    if k.shape[2] != v.shape[2]:
        return None, "kv_len"
    if not inputs_last_axis_contiguous:
        return None, "last_axis_stride"
    # This is a measured performance crossover, not a correctness bound.
    if q.shape[2] < 512 or k.shape[2] < 512:
        return None, "seq"
    if not _scale_supported(scale, q.shape[-1]):
        return None, "scale"
    return bd, ""


def maybe_steel_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float | None = None,
    mask: mx.array | None = None,
    enable_d64: bool = True,
    probe: bool = False,
    probe_compiled: bool = False,
    inputs_last_axis_contiguous: bool = False,
) -> mx.array | None:
    """Return custom STEEL attention output when this call matches the gate.

    MLX does not expose array strides to Python. The product caller constructs
    Q/K/V by splitting a contiguous projection dimension and explicitly
    attests that the final axis remains contiguous. Unknown direct callers
    default to stock-SDPA fallback instead of risking a strided Metal load.
    ``probe_compiled`` distinguishes trace observations from direct eager calls.
    """
    bd, reason = _select_config(
        q,
        k,
        v,
        scale,
        mask,
        enable_d64=enable_d64,
        inputs_last_axis_contiguous=inputs_last_axis_contiguous,
    )
    if bd is None:
        _probe_fallback(
            reason,
            q,
            k,
            v,
            mask,
            enabled=probe,
            compiled=probe_compiled,
        )
        return None

    batch, heads, seq, dim = q.shape
    d64_config = _d64_config(seq, k.shape[2]) if bd == 64 else ""
    bq, bk, wm = _tile_config(bd, d64_config)
    n_q_tiles = (seq + bq - 1) // bq
    align_q = (seq % bq) == 0
    align_k = (k.shape[2] % bk) == 0
    out = _kernel(bd, d64_config)(
        inputs=[q, k, v],
        output_shapes=[(batch, seq, heads, dim)],
        output_dtypes=[q.dtype],
        grid=(n_q_tiles * 32, heads * wm, batch),
        threadgroup=(32, wm, 1),
        template=_template(bd, align_q, align_k, d64_config),
    )[0]
    _probe_hit(
        bd,
        d64_config,
        enabled=probe,
        compiled=probe_compiled,
    )
    return out.transpose(0, 2, 1, 3)


__all__ = [
    "maybe_steel_attention",
    "reset_steel_attention_stats",
    "steel_attention_summary",
]
