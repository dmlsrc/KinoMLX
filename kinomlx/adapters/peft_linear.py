"""MLX equations for configured, self-contained PEFT Linear adapters."""

import math

import mlx.core as mx

from .formats import AdapterKind, AdapterOperation, LinearAdapterOptions


def _shape(value: mx.array, shape: tuple[int, ...], name: str) -> mx.array:
    if value.shape != shape:
        raise ValueError(f"{name} shape mismatch: got {value.shape}, expected {shape}")
    return value.astype(mx.float32)


def _pair(
    operation: AdapterOperation, shape: tuple[int, int], rank: int, prefix: str = ""
) -> mx.array:
    out, inp = shape
    down = _shape(operation.factors[f"{prefix}A"], (rank, inp), f"{operation.kind} A")
    up = _shape(operation.factors[f"{prefix}B"], (out, rank), f"{operation.kind} B")
    return up @ down


def _road(operation: AdapterOperation, base: mx.array, options: LinearAdapterOptions) -> mx.array:
    out = base.shape[0]
    original = base
    if operation.part == "bias" and "base_bias" in operation.factors:
        base = _shape(operation.factors["base_bias"], base.shape, "ROAD trained bias")
    block = options.block_size
    if out % block:
        raise ValueError("ROAD group_size must divide the output width")
    count = {"road_1": out // 2, "road_2": out, "road_4": 2 * out}[options.variant]
    theta = _shape(operation.factors["theta"], (count,), "ROAD theta")
    alpha = _shape(operation.factors["alpha"], (count,), "ROAD alpha")
    if options.variant == "road_1":
        theta = mx.repeat(theta.reshape(-1, block // 2), 2, axis=0).reshape(-1)
        alpha = mx.repeat(alpha.reshape(-1, block // 2), 2, axis=0).reshape(-1)
    if options.variant == "road_4":
        theta, alpha = theta.reshape(-1, 2, block), alpha.reshape(-1, 2, block)
        first = (mx.cos(theta[:, 0]) * alpha[:, 0]).reshape(-1)
        second = (mx.sin(theta[:, 1]) * alpha[:, 1]).reshape(-1)
    else:
        first, second = mx.cos(theta) * alpha, mx.sin(theta) * alpha
    # Rotate output rows directly. No output-by-output dense rotation is built.
    pairs = base.reshape(out // block, 2, block // 2, -1)
    perpendicular = mx.stack([-pairs[:, 1], pairs[:, 0]], axis=1).reshape(base.shape)
    scale_shape = (out,) + (1,) * (base.ndim - 1)
    return (
        base * first.reshape(scale_shape) + perpendicular * second.reshape(scale_shape) - original
    )


def _grouped_lora(
    operation: AdapterOperation, base: mx.array, options: LinearAdapterOptions
) -> mx.array:
    out, inp = base.shape
    rank, block = options.rank, options.block_size
    if operation.kind == AdapterKind.QALORA:
        if inp % block:
            raise ValueError("QALoRA group size must divide the input width")
        pooled = inp // block
        down = _shape(operation.factors["A"], (rank, pooled), "QALoRA A")
        up = _shape(operation.factors["B"], (out, rank), "QALoRA B")
        # The declared forward uses mean pooling, then multiplies by the
        # number of pooled features. Lift that linear map into the delta.
        return mx.repeat(up @ down, block, axis=1) * (pooled / block * options.scale)
    if not options.variant:
        return _pair(operation, (out, inp), rank) * options.scale
    if rank % block or (inp if options.variant == "a" else out) % block:
        raise ValueError("block-diagonal LoRA groups must divide both factor dimensions")
    step = rank // block
    if options.variant == "a":
        down = _shape(operation.factors["A"], (rank, inp // block), "block-diagonal A")
        up = _shape(operation.factors["B"], (out, rank), "LoRA B")
        parts = [
            up[:, i * step : (i + 1) * step] @ down[i * step : (i + 1) * step] for i in range(block)
        ]
        delta = mx.concatenate(parts, axis=1)
    else:
        down = _shape(operation.factors["A"], (rank, inp), "LoRA A")
        up = _shape(operation.factors["B"], (out, step), "block-diagonal B")
        height = out // block
        parts = [
            up[i * height : (i + 1) * height] @ down[i * step : (i + 1) * step]
            for i in range(block)
        ]
        delta = mx.concatenate(parts, axis=0)
    return delta * options.scale


def _activation(value: mx.array, name: str) -> mx.array:
    if name == "relu":
        return mx.maximum(value, 0)
    if name in ("silu", "swish"):
        return value * mx.sigmoid(value)
    if name in ("gelu", "gelu_python", "gelu_10"):
        result = 0.5 * value * (1 + mx.erf(value / math.sqrt(2)))
        return mx.clip(result, -10, 10) if name == "gelu_10" else result
    if name in ("gelu_new", "gelu_fast", "gelu_pytorch_tanh", "gelu_python_tanh", "gelu_accurate"):
        return 0.5 * value * (1 + mx.tanh(math.sqrt(2 / math.pi) * (value + 0.044715 * value**3)))
    if name == "quick_gelu":
        return value * mx.sigmoid(1.702 * value)
    if name == "hardswish":
        return value * mx.clip(value + 3, 0, 6) / 6
    if name == "leaky_relu":
        return mx.where(value >= 0, value, 0.01 * value)
    if name == "relu2":
        return mx.square(mx.maximum(value, 0))
    if name == "relu6":
        return mx.clip(value, 0, 6)
    if name == "laplace":
        return 0.5 * (1 + mx.erf((value - 0.707107) / (0.282095 * math.sqrt(2))))
    if name in ("mish", "sqrtsoftplus"):
        positive = mx.logaddexp(value, 0)
        return value * mx.tanh(positive) if name == "mish" else mx.sqrt(positive)
    if name == "tanh":
        return mx.tanh(value)
    if name == "sigmoid":
        return mx.sigmoid(value)
    if name == "linear":
        return value
    raise ValueError(f"PEANUT activation {name!r} has no MLX implementation")


def _peanut(operation: AdapterOperation, base: mx.array, options: LinearAdapterOptions) -> mx.array:
    out = base.shape[0]
    rank = options.rank
    down = _shape(operation.factors["A"], (rank, out), "PEANUT A")
    up = _shape(operation.factors["B"], (out, rank), "PEANUT B")
    value = _activation(base.T @ down.T, options.activation)
    residuals = []
    for index in range(options.depth):
        residuals.append(value)
        factor = _shape(operation.factors[f"encoders.{index}"], (rank, rank), "PEANUT encoder")
        value = _activation(value @ factor.T, options.activation)
    for index in range(options.depth):
        factor = _shape(operation.factors[f"decoders.{index}"], (rank, rank), "PEANUT decoder")
        value = _activation(value @ factor.T, options.activation) + residuals[-index - 1]
    return (value @ up.T).T * options.scale


def _gralora(
    operation: AdapterOperation, base: mx.array, options: LinearAdapterOptions
) -> mx.array:
    out, inp = base.shape
    groups, rank = options.block_size, options.rank
    if inp % groups or out % groups:
        raise ValueError("GRALORA groups must divide input and output widths")
    down = _shape(operation.factors["A"], (groups, inp // groups, rank), "GRALORA A")
    up = _shape(operation.factors["B"], (groups, rank, out // groups), "GRALORA B")
    # Each input/output group pair selects a different rank slice. The two
    # group coordinates exchange roles between A and B; no padded scatter.
    down = down.reshape(groups, inp // groups, groups, rank // groups)
    up = up.reshape(groups, groups, rank // groups, out // groups)
    delta = mx.einsum("aibr,baro->boai", down, up)
    delta = delta.reshape(base.shape)
    if options.hybrid_rank:
        a = _shape(operation.factors["A_general"], (options.hybrid_rank, inp), "GRALORA general A")
        b = _shape(operation.factors["B_general"], (out, options.hybrid_rank), "GRALORA general B")
        delta = mx.addmm(delta, b, a)
    return delta * options.scale


def peft_linear_delta(operation: AdapterOperation, base: mx.array) -> mx.array:
    options = operation.linear_options
    if options is None:
        raise ValueError(f"PEFT {operation.kind} requires validated PEFT configuration")
    factors = operation.factors
    if operation.kind == AdapterKind.BEFT:
        if operation.part != "bias" or base.ndim != 1:
            raise ValueError("BEFT requires an existing Linear bias target")
        added = _shape(factors["bias"], (1, base.size), "BEFT bias").reshape(base.shape)
        if "base_bias" in factors:
            added = added + _shape(factors["base_bias"], base.shape, "BEFT trained bias") - base
        return added
    if operation.kind == AdapterKind.ROAD:
        if base.ndim != (1 if operation.part == "bias" else 2):
            raise ValueError("ROAD requires a Linear weight or bias")
        return _road(operation, base, options)
    if operation.part != "weight" or base.ndim != 2:
        raise ValueError(f"{operation.kind} requires a Linear weight target")
    out, inp = base.shape
    if operation.kind in (AdapterKind.BDLORA, AdapterKind.QALORA):
        return _grouped_lora(operation, base, options)
    if operation.kind == AdapterKind.HIRA:
        return base * _pair(operation, (out, inp), options.rank)
    if operation.kind == AdapterKind.DELORA:
        rank = options.rank
        down = _shape(factors["A"], (rank, inp), "DeLoRA A")
        up = _shape(factors["B"], (out, rank), "DeLoRA B")
        gain = _shape(factors["lambda"], (1,), "DeLoRA lambda")
        norms = _shape(factors["w_norm"], (inp,), "DeLoRA frozen weight norms")
        a_norm = mx.maximum(mx.sqrt(mx.sum(down * down, axis=1)), 1e-4)
        b_norm = mx.maximum(mx.sqrt(mx.sum(up * up, axis=0)), 1e-4)
        return ((up * (gain / rank / (a_norm * b_norm))[None, :]) @ down) * norms[None, :]
    if operation.kind == AdapterKind.C3A:
        size = options.block_size
        if inp % size or out % size:
            raise ValueError("C3A block_size must divide input and output widths")
        kernel = _shape(factors["kernel"], (out // size, inp // size, size), "C3A kernel")
        positions = (mx.arange(size)[None, :] - mx.arange(size)[:, None]) % size
        blocks = kernel[:, :, positions]
        return blocks.transpose(0, 2, 1, 3).reshape(base.shape) / inp
    if operation.kind == AdapterKind.MISS:
        rank = options.rank
        if options.variant == "bat":
            if inp % rank or out % rank:
                raise ValueError("MISS bat rank must divide input and output widths")
            block = _shape(factors["block"], (out // rank, rank, rank), "MISS bat block")
            grouped = base.reshape(out // rank, rank, inp // rank, rank).transpose(2, 0, 1, 3)
            return (grouped @ block + block).transpose(1, 2, 0, 3).reshape(base.shape)
        width = options.block_size if options.variant == "mini" else out
        if out % width:
            raise ValueError("MISS mini_r must divide the output width")
        block = _shape(factors["block"], (rank, width), "MISS block")
        return mx.tile(block.T, (out // width, math.ceil(inp / rank)))[:, :inp]
    if operation.kind == AdapterKind.GRALORA:
        return _gralora(operation, base, options)
    if operation.kind == AdapterKind.PEANUT:
        return _peanut(operation, base, options)
    if operation.kind == AdapterKind.DEFT:
        rank = min(options.rank, out)
        projection = _shape(factors["P"], (out, rank), "DEFT P")
        if options.variant == "qr":
            # MLX QR is CPU-only; just the skinny saved factor crosses this boundary.
            projection, _ = mx.linalg.qr(projection, stream=mx.cpu)
            right = projection
        else:
            right = mx.maximum(projection, 0)
        removed = right.T @ base
        if options.projection_only:
            return -(projection @ removed)
        learned = _shape(factors["R"], (rank, inp), "DEFT R")
        scale = options.scale / rank if options.scale else 1.0
        return projection @ (learned * scale - removed)
    if operation.kind == AdapterKind.SHIRA:
        rows, columns = options.index_bounds
        if rows[1] >= out or columns[1] >= inp:
            raise ValueError("SHIRA COO index is outside the target weight")
        values = factors["weight"].astype(mx.float32)
        if values.ndim != 1:
            raise ValueError("SHIRA weight must be a vector")
        indices = factors["indices"].astype(mx.int32)
        return mx.zeros(base.shape, dtype=mx.float32).at[indices[0], indices[1]].add(values)
    if operation.kind == AdapterKind.SUPERTUNING:
        if options.index_bounds[0][1] >= base.size:
            raise ValueError("SUPERTUNING flat index is outside the target weight")
        indices = factors["indices"].astype(mx.int32)
        delta = (
            mx.zeros((base.size,), dtype=mx.float32)
            .at[indices]
            .add(factors["values"].astype(mx.float32))
        )
        delta = delta.reshape(base.shape)
        if options.rank:
            delta = delta + _pair(operation, (out, inp), options.rank, "lora_") * options.scale
        return delta
    raise ValueError(f"PEFT {operation.kind} has no Linear equation")
