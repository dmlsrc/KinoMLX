"""PEFT GLoRA paths, distinct from LyCORIS matrix-product GLoRA."""

import mlx.core as mx

from .formats import AdapterOperation


def _path(operation: AdapterOperation, label: str, shape: tuple[int, ...]) -> mx.array:
    options = operation.glora_options
    if options is None:
        raise ValueError("PEFT GLoRA requires adapter_config.json path choices")
    mode = (
        options.weight_path
        if label in "AB"
        else (options.coupling_path if label == "C" else options.bias_path)
    )
    prefix = f"glora_{label}_"
    factors = operation.factors
    if prefix + "zeros" in factors:
        if mode != "none" and label in "AB":
            raise ValueError(f"PEFT GLoRA {label} is disabled but configuration declares {mode}")
        if factors[prefix + "zeros"].shape != shape:
            raise ValueError(f"PEFT GLoRA {label} disabled path shape does not match its target")
        return mx.array(0.0)
    if mode == "lora":
        down, up = factors.get(prefix + "Xd"), factors.get(prefix + "Xu")
        if (
            down is None
            or up is None
            or down.shape != (shape[0], options.rank)
            or up.shape != (options.rank, shape[1])
        ):
            raise ValueError(
                f"PEFT GLoRA {label} low-rank path disagrees with its configured shape"
            )
        return down.astype(mx.float32) @ up.astype(mx.float32)
    vector = factors.get(prefix + "X")
    expected = (1,) if mode == "constant" else ((shape[0], 1) if len(shape) == 2 else shape)
    if mode not in ("constant", "vector") or vector is None or vector.shape != expected:
        raise ValueError(f"PEFT GLoRA {label} path disagrees with its configured shape")
    return (
        vector.astype(mx.float32).reshape(()) if mode == "constant" else vector.astype(mx.float32)
    )


def glora_has_bias_paths(operation: AdapterOperation) -> bool:
    """Whether an export requires a biased base layer, not just zero buffers."""
    return "base_bias" in operation.factors or any(
        f"glora_{label}_zeros" not in operation.factors for label in "CDE"
    )


def peft_glora_delta(
    operation: AdapterOperation, base: mx.array, base_weight: mx.array | None
) -> mx.array:
    if operation.part == "weight":
        if base.ndim != 2:
            raise ValueError("PEFT GLoRA requires a Linear weight")
        outputs, inputs = base.shape
        # Validate bias factors even when applying only the weight half.
        _path(operation, "C", (inputs, 1))
        _path(operation, "D", (outputs,))
        _path(operation, "E", (outputs,))
        return (
            base * _path(operation, "A", base.shape) + _path(operation, "B", base.shape)
        ) * operation.scalar
    if base_weight is None or base_weight.ndim != 2 or base.shape != (base_weight.shape[0],):
        raise ValueError("PEFT GLoRA bias needs the unchanged companion Linear weight")
    weight = base_weight.astype(mx.float32)
    bias = operation.factors.get("base_bias", base).astype(mx.float32)
    if bias.shape != base.shape:
        raise ValueError("PEFT GLoRA saved base bias does not match its target")
    coupling = _path(operation, "C", (weight.shape[1], 1))
    coupled = mx.zeros_like(base) if coupling.ndim == 0 else (weight @ coupling).reshape(base.shape)
    return (
        bias
        - base
        + bias * _path(operation, "D", base.shape)
        + _path(operation, "E", base.shape)
        + coupled
    ) * operation.scalar
