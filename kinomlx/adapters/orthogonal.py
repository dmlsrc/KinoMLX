"""PEFT input-axis rotations, distinct from LyCORIS output rotations."""

import math

import mlx.core as mx

from .formats import AdapterOperation


def _cayley(skew: mx.array) -> mx.array:
    identity = mx.eye(skew.shape[-1], dtype=mx.float32)
    # MLX has no GPU inverse. Only the small rotation blocks cross to CPU.
    return (identity - skew) @ mx.linalg.inv(identity + skew, stream=mx.cpu)


def _oft(operation: AdapterOperation, base: mx.array) -> mx.array:
    options = operation.oft_options
    if options is None:
        raise ValueError("packed PEFT OFT requires adapter_config.json inference options")
    if "oft_legacy" in operation.factors:
        legacy_blocks = operation.factors["oft_legacy"].astype(mx.float32)
        scale = operation.factors["oft_scale"].astype(mx.float32)
        if (
            legacy_blocks.ndim != 3
            or legacy_blocks.shape[1] != legacy_blocks.shape[2]
            or legacy_blocks.shape[1] == 0
        ):
            raise ValueError("legacy PEFT OFT requires block x square-matrix factors")
        count, size, _ = legacy_blocks.shape
        target_blocks = base.shape[1] // size
        if (
            base.shape[1] % size
            or count != (1 if options.block_share else target_blocks)
            or scale.shape != (base.shape[0], 1)
        ):
            raise ValueError("legacy PEFT OFT blocks/scale do not match the Linear target")
        if options.coft:
            # The old export constrains full parameters before skew projection,
            # using an output-dimension-dependent radius, not packed OFT's rule.
            limit = options.eps * math.ceil(base.shape[0] / target_blocks) ** 2 / math.sqrt(count)
            norm = mx.sqrt(mx.sum(legacy_blocks * legacy_blocks, axis=(-2, -1), keepdims=True))
            legacy_blocks = legacy_blocks * mx.where(norm <= limit, 1.0, limit / norm)
        rotation = _cayley((legacy_blocks - mx.swapaxes(legacy_blocks, -1, -2)) * 0.5)
        return (rotation @ base.T.reshape(target_blocks, size, -1)).reshape(base.T.shape).T * scale
    packed = operation.factors["oft_packed"].astype(mx.float32)
    if packed.ndim != 2:
        raise ValueError("PEFT OFT needs packed upper-triangle vectors, not legacy full blocks")
    blocks, elements = packed.shape
    size = (1 + math.isqrt(1 + 8 * elements)) // 2
    if elements != size * (size - 1) // 2 or size < 2:
        raise ValueError("PEFT OFT packed length does not describe a skew matrix")
    if base.shape[1] % size or blocks != (1 if options.block_share else base.shape[1] // size):
        raise ValueError("PEFT OFT blocks do not cover the target input dimension")
    positions = [(row, column) for row in range(size) for column in range(row + 1, size)]
    rows = mx.array([row for row, _ in positions])
    columns = mx.array([column for _, column in positions])
    upper = mx.zeros((blocks, size, size), dtype=mx.float32).at[:, rows, columns].add(packed)
    skew = upper - mx.swapaxes(upper, -1, -2)
    if options.coft:
        norm = mx.sqrt(mx.sum(skew * skew, axis=(-2, -1), keepdims=True))
        limit = options.eps / math.sqrt(blocks)
        skew = skew * mx.where(norm <= limit, 1.0, limit / norm)
    if options.use_neumann:
        # Preserve the pinned export's polynomial, including its endpoint
        # coefficient. Its sign is not interchangeable with the exact branch.
        rotation = mx.broadcast_to(mx.eye(size), skew.shape)
        terms = options.neumann_terms
        last_power = max(3, terms - 1) if terms > 2 else terms - 1
        power = skew
        for exponent in range(1, last_power + 1):
            coefficient = 1.0 if terms > 2 and exponent == last_power else 2.0
            rotation = rotation + coefficient * power
            if exponent != last_power:
                power = power @ skew
    else:
        rotation = _cayley(skew)
    return (rotation @ base.T.reshape(-1, size, base.shape[0])).reshape(base.T.shape).T


def _boft(operation: AdapterOperation, base: mx.array) -> mx.array:
    blocks = operation.factors["boft_rotation"].astype(mx.float32)
    scale = operation.factors["boft_scale"].astype(mx.float32)
    if blocks.ndim != 4 or blocks.shape[-1] != blocks.shape[-2]:
        raise ValueError("PEFT BOFT requires stage x block x square-matrix factors")
    stages, count, size, _ = blocks.shape
    if count * size != base.shape[1] or scale.shape != (base.shape[0], 1):
        raise ValueError("PEFT BOFT rotation/scale dimensions do not match the linear target")
    if stages > 1 and (size % 2 or count % (2 ** (stages - 1))):
        raise ValueError("PEFT BOFT butterfly stages do not divide the input dimension")
    rotation = _cayley((blocks - mx.swapaxes(blocks, -1, -2)) * 0.5)
    adapted = base.T
    for stage in range(stages):
        if stage:
            span = size * 2**stage
            half = size // 2
            order = [
                start + chunk * half + offset
                for start in range(0, base.shape[1], span)
                for parity in (0, 1)
                for chunk in range(parity, span // half, 2)
                for offset in range(half)
            ]
            inverse = [0] * len(order)
            for destination, source in enumerate(order):
                inverse[source] = destination
            adapted = adapted[mx.array(inverse)]
        adapted = (rotation[stage] @ adapted.reshape(count, size, -1)).reshape(base.T.shape)
        if stage:
            adapted = adapted[mx.array(order)]
    return adapted.T * scale


def _hra_components(operation: AdapterOperation) -> tuple[mx.array, mx.array | None]:
    if operation.hra_apply_gs is None:
        raise ValueError("PEFT HRA requires adapter_config.json to select apply_GS")
    vectors = operation.factors["hra_vectors"].astype(mx.float32)
    if vectors.ndim != 2 or vectors.shape[1] < 1:
        raise ValueError("PEFT HRA requires input x rank reflection vectors")
    rank = vectors.shape[1]
    if operation.hra_apply_gs:
        if rank > vectors.shape[0]:
            raise ValueError("HRA Gram-Schmidt rank cannot exceed the input dimension")
        columns: list[mx.array] = []
        for index in range(rank):
            value = vectors[:, index]
            for column in columns:
                value = value - mx.sum(column * value) * column
            columns.append(value / mx.sqrt(mx.sum(value * value)))
        return mx.stack(columns, axis=1), None
    vectors = vectors / mx.sqrt(mx.sum(vectors * vectors, axis=0))
    # Compact Householder product: R = I - U T U.T. T is rank x rank;
    # never form an input x input rotation or walk every dense weight r times.
    triangular = mx.array([[2.0]])
    for index in range(1, rank):
        column = -2.0 * (triangular @ (vectors[:, :index].T @ vectors[:, index]))
        previous = mx.concatenate([triangular, mx.zeros((1, index))], axis=0)
        last = mx.concatenate([column, mx.array([2.0])])[:, None]
        triangular = mx.concatenate([previous, last], axis=1)
    return vectors, triangular


def peft_orthogonal_delta(operation: AdapterOperation, base: mx.array) -> mx.array:
    """Apply rotations/reflections without a model-sized square matrix."""
    if base.ndim != 2:
        raise ValueError("PEFT orthogonal adapters require a linear target")
    if operation.kind == "hra":
        vectors, triangular = _hra_components(operation)
        if vectors.shape[0] != base.shape[1]:
            raise ValueError("PEFT HRA vectors do not match the target input dimension")
        projected = base @ vectors
        projected = 2.0 * projected if triangular is None else projected @ triangular
        return -(projected @ vectors.T) * operation.scalar
    adapted = _oft(operation, base) if operation.kind == "oft_peft" else _boft(operation, base)
    return (adapted - base) * operation.scalar
