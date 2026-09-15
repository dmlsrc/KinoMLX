"""MLX reconstruction of static adapters in logical source orientation.

All reconstruction and multi-adapter accumulation use FP32. The caller casts
once to the destination dtype after preflight. A user strength blends the
complete adapted weight with the unchanged base, including for DoRA and OFT.
Thus zero means no change and multiple adapters share the same base weight.
"""

import math

import mlx.core as mx

from .basis import basis_delta
from .formats import AdapterOperation
from .fourier import fourier_delta
from .glora import peft_glora_delta
from .lora import LoRAEntry, compute_delta
from .orthogonal import peft_orthogonal_delta
from .peft_linear import peft_linear_delta
from .wavelet import wavelet_delta


def adalora_entry(operation: AdapterOperation) -> LoRAEntry | None:
    """Normalize the saved singular-value factor and original rank scaling."""
    down, up = operation.factors["down"], operation.factors["up"]
    if down.ndim != 2 or up.ndim != 2 or up.shape[1] != down.shape[0]:
        raise ValueError("AdaLoRA requires matching two-dimensional factors")
    rank = down.shape[0]
    if operation.factors["diagonal"].shape != (rank, 1):
        raise ValueError("AdaLoRA singular values must have shape (rank, 1)")
    if rank == 0:
        return None
    denominator = float(rank) if operation.rank_denominator is None else operation.rank_denominator
    if not math.isfinite(denominator) or denominator < 0:
        raise ValueError("AdaLoRA normalization rank must be finite and nonnegative")
    alpha = operation.alpha if operation.alpha is not None and operation.alpha > 0 else denominator
    return LoRAEntry(
        down.astype(mx.float32) * operation.factors["diagonal"].astype(mx.float32),
        up,
        alpha * rank / (denominator + 1e-5),
    )


def plain_lora_entry(operation: AdapterOperation) -> LoRAEntry | None:
    """Return the plain two-dimensional low-rank pair an operation reduces to.

    ``None`` means the operation needs full reconstruction: a DoRA magnitude, a
    Tucker middle factor, or a non-matrix factor shape. Shared by the fuser's
    fast path and the live activation path so both recognize the same subset.
    """
    factors = operation.factors
    if operation.kind == "adalora":
        return adalora_entry(operation)
    if operation.kind != "lora" or operation.part != "weight":
        return None
    if "magnitude" in factors or "mid" in factors:
        return None
    down, up = factors["down"], factors["up"]
    if down.ndim != 2 or up.ndim != 2:
        return None
    return LoRAEntry(down, up, operation.alpha)


def loha_pairs(
    operation: AdapterOperation, shape: tuple[int, ...]
) -> tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array], int] | None:
    """Return both LoHa low-rank pairs as ``(up, down)``, or ``None`` for Tucker.

    The Hadamard product of the two pairs does not commute through a matmul, so
    the live path expands these into one wider pair instead. Tucker forms stay
    with full reconstruction.
    """
    factors = operation.factors
    if operation.kind != "loha" or operation.part != "weight" or len(shape) != 2:
        return None
    if "magnitude" in factors or any(f"hada_t{index}" in factors for index in (1, 2)):
        return None
    pairs = []
    ranks = []
    for index in (1, 2):
        up = _matrix(factors[f"hada_w{index}_a"], "LoHa up factor")
        down = _matrix(factors[f"hada_w{index}_b"], "LoHa down factor")
        if up.shape[1] != down.shape[0]:
            raise ValueError("LoHa has mismatched low-rank factor dimensions")
        _same_shape((up.shape[0], down.shape[1]), shape, "LoHa")
        pairs.append((up, down))
        ranks.append(down.shape[0])
    if ranks[0] != ranks[1]:
        raise ValueError("LoHa factor pairs must agree on rank for alpha scaling")
    return pairs[0], pairs[1], ranks[0]


def ia3_input_scale(operation: AdapterOperation) -> bool:
    """PEFT stores input vectors as (1, in), output vectors as (out, 1)."""
    vector = operation.factors["ia3_vector"]
    if vector.ndim != 2 or 1 not in vector.shape:
        raise ValueError("PEFT IA3 requires a two-dimensional row or column vector")
    if operation.input_scale is not None:
        return operation.input_scale
    if vector.shape == (1, 1):
        raise ValueError(
            "scalar PEFT IA3 needs feedforward_modules configuration to select its axis"
        )
    return vector.shape[0] == 1


def _matrix(value: mx.array, name: str) -> mx.array:
    if value.ndim < 2 or any(size != 1 for size in value.shape[2:]):
        raise ValueError(f"{name} must be a matrix or pointwise convolution, got {value.shape}")
    return value.astype(mx.float32).reshape(value.shape[:2])


def _same_shape(actual: tuple[int, ...], expected: tuple[int, ...], name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} shape mismatch: reconstructed {actual}, target {expected}")


def _matrix_target(base: mx.array, name: str) -> None:
    if base.ndim != 2:
        raise ValueError(f"{name} needs a linear target, got {base.shape}")


def intrinsic_scale(operation: AdapterOperation, rank: int | None) -> float:
    """Return the file-intrinsic scale an operation applies before user strength."""
    denominator = rank if operation.rank_denominator is None else operation.rank_denominator
    if denominator is not None and denominator <= 0:
        raise ValueError(f"adapter {operation.prefix!r} has an invalid rank")
    # Kohya, LyCORIS, and AI Toolkit exports use a saved zero alpha to mean
    # "use the rank," yielding unit intrinsic scale. Keep that convention
    # limited to their rank-scaled LoRA/LoHa/LoKr families; OFT uses alpha as
    # a separate rotation constraint, and a dense LoKr has no rank divisor.
    zero_alpha_means_unit = (
        operation.alpha == 0.0
        and denominator is not None
        and operation.kind in ("lora", "loha", "lokr")
    )
    intrinsic = (
        1.0
        if operation.alpha is None or denominator is None or zero_alpha_means_unit
        else operation.alpha / denominator
    )
    scale = intrinsic * operation.scalar
    if not math.isfinite(scale):
        raise ValueError(f"adapter {operation.prefix!r} has a nonfinite scale")
    return scale


def _low_rank(
    up: mx.array,
    down: mx.array,
    mid: mx.array | None,
    shape: tuple[int, ...],
    name: str,
) -> mx.array:
    up = _matrix(up, f"{name} up factor")
    if down.ndim < 2:
        raise ValueError(f"{name} down factor must have at least two axes")
    down = down.astype(mx.float32)
    if mid is None:
        if up.shape[1] != down.shape[0]:
            raise ValueError(f"{name} has mismatched low-rank factor dimensions")
        spatial = down.shape[2:]
        output_shape = (up.shape[0], down.shape[1], *spatial)
        if len(shape) == 2 and all(size == 1 for size in spatial):
            output_shape = output_shape[:2]
        _same_shape(output_shape, shape, name)
        # Ordinary LoRA shares the established FP32 B @ A implementation.
        return compute_delta(LoRAEntry(down.reshape(down.shape[0], -1), up)).reshape(shape)
    down = _matrix(down, f"{name} Tucker down factor")
    if mid.ndim < 2 or mid.shape[:2] != (up.shape[1], down.shape[0]):
        raise ValueError(f"{name} has inconsistent Tucker dimensions")
    spatial = mid.shape[2:]
    output_shape = (up.shape[0], down.shape[1], *spatial)
    if len(shape) == 2 and all(size == 1 for size in spatial):
        output_shape = output_shape[:2]
    _same_shape(output_shape, shape, name)
    return mx.einsum(
        "oa,abk,bi->oik", up, mid.astype(mx.float32).reshape(*mid.shape[:2], -1), down
    ).reshape(shape)


def _loha(operation: AdapterOperation, base: mx.array) -> tuple[mx.array, int]:
    factors = operation.factors
    products = []
    ranks = []
    for index in (1, 2):
        up = factors[f"hada_w{index}_a"]
        down = factors[f"hada_w{index}_b"]
        mid = factors.get(f"hada_t{index}")
        if mid is not None:
            up = _matrix(up, "LoHa Tucker factor").T
        products.append(_low_rank(up, down, mid, base.shape, "LoHa"))
        ranks.append(down.shape[0])
    if ranks[0] != ranks[1]:
        raise ValueError("LoHa factor pairs must agree on rank for alpha scaling")
    return products[0] * products[1], ranks[0]


def lokr_factors(
    operation: AdapterOperation, shape: tuple[int, ...]
) -> tuple[mx.array, mx.array, int | None]:
    """Return the two Kronecker factors and rank one LoKr operation declares.

    Shared with the live activation path, which applies the factors directly and
    never builds their Kronecker expansion.
    """
    factors = operation.factors
    first = factors.get("lokr_w1")
    ranks = []
    if first is None:
        left = _matrix(factors["lokr_w1_a"], "LoKr W1 A")
        right = _matrix(factors["lokr_w1_b"], "LoKr W1 B")
        if left.shape[1] != right.shape[0]:
            raise ValueError("LoKr W1 factors have mismatched rank")
        first = left @ right
        ranks.append(right.shape[0])
    first = _matrix(first, "LoKr W1")
    if len(shape) < 2 or shape[0] % first.shape[0] or shape[1] % first.shape[1]:
        raise ValueError("LoKr W1 dimensions do not divide the target")
    second_shape = (
        shape[0] // first.shape[0],
        shape[1] // first.shape[1],
        *shape[2:],
    )
    second = factors.get("lokr_w2")
    if second is None:
        left = factors["lokr_w2_a"]
        right = factors["lokr_w2_b"]
        mid = factors.get("lokr_t2")
        if mid is not None:
            left = _matrix(left, "LoKr Tucker W2 A").T
        second = _low_rank(left, right, mid, second_shape, "LoKr W2")
        ranks.append(right.shape[0])
    else:
        if len(second_shape) == 2:
            second = _matrix(second, "LoKr W2")
        _same_shape(second.shape, second_shape, "LoKr W2")
    if len(set(ranks)) > 1 and operation.alpha is not None:
        raise ValueError("LoKr factors have different ranks, so alpha scaling is ambiguous")
    return first, second, (ranks[0] if ranks else None)


def _lokr(operation: AdapterOperation, base: mx.array) -> tuple[mx.array, int | None]:
    first, second, rank = lokr_factors(operation, base.shape)
    # Arrange (out1, out2, in1, in2, spatial) before flattening. A reshape of
    # the outer product without this permutation gives the wrong Kronecker map.
    expanded = first[:, None, :, None] * second.astype(mx.float32).reshape(
        1, second.shape[0], 1, -1
    )
    return expanded.reshape(base.shape), rank


def _dora_magnitude_layout(
    operation: AdapterOperation,
    base: mx.array,
) -> tuple[mx.array, bool, tuple[int, ...]]:
    magnitude = operation.factors["magnitude"].astype(mx.float32)
    output_shaped = magnitude.shape[0] == base.shape[0] and magnitude.size == base.shape[0]
    output_axis = operation.magnitude_is_peft or magnitude.ndim == 1 or output_shaped
    if output_axis:
        if magnitude.size != base.shape[0]:
            raise ValueError("DoRA output magnitude does not match the target output dimension")
        shape = (base.shape[0], *(1 for _ in base.shape[1:]))
    else:
        if magnitude.size != base.shape[1]:
            raise ValueError("DoRA input magnitude does not match the target input dimension")
        shape = (1, base.shape[1], *(1 for _ in base.shape[2:]))
    return magnitude, output_axis, shape


def _glora(operation: AdapterOperation, base: mx.array) -> tuple[mx.array, int]:
    _matrix_target(base, "GLoRA")
    factors = operation.factors
    a1, a2 = _matrix(factors["a1"], "GLoRA A1"), _matrix(factors["a2"], "GLoRA A2")
    if a1.shape[0] != base.shape[1] or a2.shape != (a1.shape[1], base.shape[1]):
        raise ValueError("GLoRA A factors do not match the target input dimension")
    additive = _low_rank(factors["b1"], factors["b2"], factors.get("bm"), base.shape, "GLoRA B")
    if factors["b2"].shape[0] != a2.shape[0]:
        raise ValueError("GLoRA A and B factors disagree on rank")
    return (base @ a1) @ a2 + additive, a2.shape[0]


def _orthogonal(operation: AdapterOperation, base: mx.array) -> mx.array:
    _matrix_target(base, operation.kind.upper())
    blocks = operation.factors["oft_blocks"].astype(mx.float32)
    ndim = 4 if operation.kind == "boft" else 3
    if blocks.ndim != ndim or blocks.shape[-1] != blocks.shape[-2]:
        raise ValueError(f"{operation.kind} needs square rotation blocks with {ndim} axes")
    block_size = blocks.shape[-1]
    block_count = blocks.shape[-3]
    if block_size * block_count != base.shape[0]:
        raise ValueError(f"{operation.kind} rotation blocks do not cover the target output rows")
    skew = blocks - mx.swapaxes(blocks, -1, -2)
    constraint = 0.0 if operation.alpha is None else operation.alpha * base.shape[0]
    if constraint < 0.0:
        raise ValueError("OFT/BOFT constraint alpha must be nonnegative")
    if constraint > 0.0:
        skew = skew * mx.minimum(1.0, constraint / (mx.sqrt(mx.sum(skew * skew)) + 1e-8))
    identity = mx.eye(block_size, dtype=mx.float32)
    # MLX 0.32 has no GPU inverse. Only these small rotation blocks use CPU;
    # dense reconstruction and the model remain on the GPU.
    rotation = (identity + skew) @ mx.linalg.inv(identity - skew, stream=mx.cpu)
    if operation.kind == "oft":
        adapted = (
            mx.swapaxes(rotation, -1, -2) @ base.reshape(block_count, block_size, -1)
        ).reshape(base.shape)
    else:
        if block_size % 2:
            raise ValueError("BOFT requires even rotation block size")
        adapted = base
        for stage in range(blocks.shape[0]):
            stride = (block_size // 2) * (2**stage)
            if base.shape[0] % (2 * stride):
                raise ValueError("BOFT butterfly stages do not divide the output dimension")
            permuted = mx.swapaxes(adapted.reshape(-1, 2, stride, base.shape[1]), 1, 2)
            rotated = rotation[stage] @ permuted.reshape(block_count, block_size, -1)
            adapted = mx.swapaxes(rotated.reshape(-1, stride, 2, base.shape[1]), 1, 2).reshape(
                base.shape
            )
    rescale = operation.factors.get("rescale")
    if rescale is not None:
        if rescale.size != base.shape[0] or rescale.shape[0] != base.shape[0]:
            raise ValueError("OFT/BOFT rescale must contain one magnitude per output row")
        adapted = adapted * rescale.astype(mx.float32).reshape(-1, 1)
    return (adapted - base) * operation.scalar


def _decompose(operation: AdapterOperation, base: mx.array, delta: mx.array) -> mx.array:
    magnitude = operation.factors.get("magnitude")
    if magnitude is None:
        return delta
    if base.ndim < 2:
        raise ValueError("DoRA needs a linear or convolution weight")
    merged = base + delta
    magnitude, output_axis, shape = _dora_magnitude_layout(operation, base)
    if output_axis:
        norm = mx.sqrt(mx.sum(merged.reshape(base.shape[0], -1) ** 2, axis=1)).reshape(shape)
    else:
        norm = mx.sqrt(
            mx.sum(mx.swapaxes(merged, 0, 1).reshape(base.shape[1], -1) ** 2, axis=1)
        ).reshape(shape)
    # A zero norm and zero magnitude have the well-defined zero result. A
    # nonzero magnitude with no direction remains nonfinite and fails preflight.
    scale = mx.where(
        (norm == 0) & (magnitude.reshape(shape) == 0), 0.0, magnitude.reshape(shape) / norm
    )
    return merged * scale - base


def reconstruct_delta(
    operation: AdapterOperation, base: mx.array, *, base_weight: mx.array | None = None
) -> mx.array:
    """Return the complete unit-strength dense delta, never modifying ``base``."""
    base = base.astype(mx.float32)
    factors = operation.factors
    kind = operation.kind
    if operation.linear_options is not None:
        if kind in ("vera", "pvera", "randlora", "vblora", "unilora", "tinylora"):
            return basis_delta(operation, base) * operation.scalar
        if kind == "waveft":
            return wavelet_delta(operation, base) * operation.scalar
        return peft_linear_delta(operation, base) * operation.scalar
    if kind == "fourierft":
        return fourier_delta(operation, base)
    if kind == "glora_peft":
        return peft_glora_delta(operation, base, base_weight)
    if kind in ("oft_peft", "boft_peft", "hra"):
        return peft_orthogonal_delta(operation, base)
    if operation.part == "bias":
        if kind == "ia3_peft":
            vector = factors["ia3_vector"].astype(mx.float32).reshape(-1)
            _same_shape(vector.shape, base.shape, "PEFT IA3 bias")
            trained_bias = factors.get("base_bias", base).astype(mx.float32)
            _same_shape(trained_bias.shape, base.shape, "PEFT IA3 trained bias")
            return (trained_bias * vector - base) * operation.scalar
        if kind == "lora":
            delta = factors["up_bias"].astype(mx.float32)
            scale = intrinsic_scale(operation, factors["down"].shape[0])
        else:
            delta, scale = factors["delta"].astype(mx.float32), operation.scalar
        _same_shape(delta.shape, base.shape, "adapter bias")
        if kind == "set":
            delta = delta - base
        if kind == "lora" and "base_bias" in factors:
            replacement = factors["base_bias"].astype(mx.float32)
            _same_shape(replacement.shape, base.shape, "saved base bias")
            return replacement - base + delta * scale
        return delta * scale
    if kind in ("full", "norm", "set"):
        role = {"full": "diff", "norm": "w_norm", "set": "set_weight"}[kind]
        value = factors[role].astype(mx.float32)
        _same_shape(value.shape, base.shape, kind)
        return (value - base if kind == "set" else value) * operation.scalar
    if kind == "ia3":
        _matrix_target(base, "IA3")
        on_input = bool(factors["on_input"].item())
        vector = factors["weight"].astype(mx.float32)
        expected = base.shape[1 if on_input else 0]
        if vector.size != expected:
            raise ValueError("IA3 vector does not match its selected input/output axis")
        return base * vector.reshape((1, -1) if on_input else (-1, 1)) * operation.scalar
    if kind == "ia3_peft":
        _matrix_target(base, "PEFT IA3")
        on_input = ia3_input_scale(operation)
        vector = factors["ia3_vector"].astype(mx.float32)
        _same_shape(
            vector.shape, (1, base.shape[1]) if on_input else (base.shape[0], 1), "PEFT IA3"
        )
        return base * (vector - 1.0) * operation.scalar
    if kind == "adalora":
        entry = adalora_entry(operation)
        expected_shape = (factors["up"].shape[0], factors["down"].shape[1])
        _same_shape(expected_shape, base.shape, "AdaLoRA")
        return mx.zeros_like(base) if entry is None else compute_delta(entry) * operation.scalar
    if kind in ("oft", "boft"):
        return _orthogonal(operation, base)
    rank: int | None
    if kind == "lora":
        delta = _low_rank(
            factors["up"], factors["down"], factors.get("mid"), base.shape, "LoRA/LoCon"
        )
        rank = factors["down"].shape[0]
    elif kind == "loha":
        delta, rank = _loha(operation, base)
    elif kind == "lokr":
        delta, rank = _lokr(operation, base)
    elif kind == "glora":
        delta, rank = _glora(operation, base)
    else:
        raise ValueError(f"unsupported static adapter family: {kind}")
    return _decompose(operation, base, delta * intrinsic_scale(operation, rank))
