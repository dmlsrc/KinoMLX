"""FP16 dense attention for LTX's unmasked attention calls.

One threadgroup of four SIMD groups computes a query tile of one head over every key, staged 16
keys at a time: Q held in registers after one load, K and V staged together behind two barriers
with loaders that spread their stores over more threadgroup-memory banks and issue the device
reads before the first barrier, an online softmax that skips the output rescale for SIMD groups
whose running maxima held, and FP32 accumulation throughout. Heads are 64 or 128 wide. Each SIMD
group owns one 8-row query fragment at 128 wide (32-query tiles) and two at 64 wide (64-query
tiles), where the registers allow a second fragment to reuse every staged K and V fragment.
Queries are token-major ``(query_tokens, heads * dim)`` and keys and values
``(key_tokens, heads * dim)``, the layout of LTX's projections, so neither side needs a head
transpose, and self- and cross-attention are the same call.

On an M1 Max a 32-head video self-attention call at 30,576 tokens took 1.67 s, against 1.99 s for
MLX's FP16 SDPA and 2.13 s for the BF16 STEEL kernel KinoMLX used to run, and its text
cross-attention over 1,024 tokens 59 to 60 ms against 68 and 74. At 64 wide, with 752 audio
tokens, BF16 audio-to-video cross-attention (30,576 video queries) took 23.9 ms against 28.2
for STEEL, video-to-audio 25.9 against 27.5, and audio self- and text attention 0.72 and 0.94
against 0.75 and 0.96. With fewer than 512 tokens on either side MLX's SDPA was faster on the
64-wide calls (at 127 audio tokens by 7 to 80%), so ``applies`` leaves those calls to it.

The loop is an FP16 build. M1 has no BF16 arithmetic, so BF16 inputs are converted once,
saturating at FP16's largest finite value (inside FP16's normal range the conversion is exact),
and the output is stored in the input dtype.
"""

from functools import cache
from pathlib import Path
from typing import cast

import mlx.core as mx

from kinomlx.kernels._typing import MetalKernel

HEAD_DIMS = (64, 128)
# 8-row query fragments per SIMD group, by head width.
_QUERY_FRAGMENTS = {64: 2, 128: 1}
# Keys staged per tile. 32-key tiles spilled at 128 wide and, with two query fragments at 64
# wide, needed registers enough to leave three threadgroups per core and ran slower.
_KEY_TILE = 16
_SIMD_GROUPS = 4
_THREADS = _SIMD_GROUPS * 32
# Keeps the build at four 128-thread threadgroups per GPU core.
_THREAD_CAP = 512
# 64-wide calls with fewer tokens than this on either side ran faster through MLX's SDPA.
_MIN_TOKENS_64 = 512
_SUPPORTED_DTYPES = (mx.float16, mx.bfloat16)
_FP16_MAX = 65504.0

_METAL = Path(__file__).with_name("metal")


def _read_metal(name: str) -> str:
    return (_METAL / name).read_text(encoding="utf-8")


@cache
def _kernel() -> MetalKernel:
    # MLX's generated source has no parameter for this attribute, so the header redefines
    # `kernel` to carry it and the body undefines it before MLX's explicit instantiation.
    header = (
        _read_metal("fp16_attention_header.metal")
        + f"\n#define kernel kernel, max_total_threads_per_threadgroup({_THREAD_CAP})\n"
    )
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="ltx_dense_attention",
            input_names=["q", "k", "v"],
            output_names=["out"],
            source=_read_metal("dense_attention_body.metal"),
            header=header,
            ensure_row_contiguous=True,
        ),
    )


def _saturated_fp16(value: mx.array) -> mx.array:
    # Clamp in FP32: BF16 cannot hold 65504 and would round the bound up to 65536, which
    # overflows FP16. Compiled, the three operations are one elementwise pass.
    return mx.clip(value.astype(mx.float32), -_FP16_MAX, _FP16_MAX).astype(mx.float16)


_compiled_saturated_fp16 = mx.compile(_saturated_fp16)


def applies(query: mx.array, key: mx.array, *, heads: int) -> bool:
    """Whether flattened-head ``(batch, tokens, heads * dim)`` queries and keys take the kernel.

    128-wide heads always do; 64-wide heads do with at least 512 tokens on both sides.
    """
    if not (
        query.ndim == 3
        and key.ndim == 3
        and query.dtype in _SUPPORTED_DTYPES
        and key.dtype == query.dtype
        and query.shape[0] == key.shape[0]
        and query.shape[-1] == key.shape[-1]
        and query.shape[-1] in tuple(heads * dim for dim in HEAD_DIMS)
        and query.shape[1] > 0
        and key.shape[1] > 0
    ):
        return False
    return query.shape[-1] == heads * 128 or min(query.shape[1], key.shape[1]) >= _MIN_TOKENS_64


def dense_attention(q: mx.array, k: mx.array, v: mx.array, *, heads: int) -> mx.array:
    """Dense attention of ``(query_tokens, heads * dim)`` queries over ``(key_tokens, ...)``.

    ``dim`` is 64 or 128; the output has the queries' shape and dtype.
    """
    if q.dtype not in _SUPPORTED_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(
            f"dense attention needs float16 or bfloat16 q, k and v, got {q.dtype}, {k.dtype}, "
            f"{v.dtype}"
        )
    width = q.shape[1] if q.ndim == 2 else 0
    dim = width // heads if heads > 0 and width % heads == 0 else 0
    if (
        dim not in HEAD_DIMS
        or k.ndim != 2
        or k.shape[1] != width
        or v.shape != k.shape
        or q.shape[0] == 0
        or k.shape[0] == 0
    ):
        raise ValueError(
            f"dense attention needs (query_tokens, {heads} * 64 or 128) queries and matching "
            f"(key_tokens, width) keys and values, got {q.shape}, {k.shape}, {v.shape}"
        )
    query_tokens, key_tokens = q.shape[0], k.shape[0]
    out_dtype = q.dtype
    if q.dtype == mx.bfloat16:
        q, k, v = (_compiled_saturated_fp16(x) for x in (q, k, v))
    fragments = _QUERY_FRAGMENTS[dim]
    query_tile = _SIMD_GROUPS * 8 * fragments
    tiles = (query_tokens + query_tile - 1) // query_tile
    return _kernel()(
        inputs=[q, k, v],
        template=[
            ("T", mx.float16),
            ("H", heads),
            ("D", dim),
            ("QF", fragments),
            ("BK", _KEY_TILE),
            ("TOKENS_Q", query_tokens),
            ("TOKENS_K", key_tokens),
        ],
        grid=(tiles * _THREADS, heads, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(query_tokens, width)],
        output_dtypes=[out_dtype],
    )[0]


__all__ = ["HEAD_DIMS", "applies", "dense_attention"]
