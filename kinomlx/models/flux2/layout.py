"""Measured Klein storage layouts, separate from logical weight shapes."""

from kinomlx.modeling.dtypes import FloatDType


def pretransposed_weight(key: str) -> bool:
    """Select repeated FF-down and fused attention/MLP-down projections."""
    return (
        key.startswith("transformer_blocks.")
        and key.endswith((".ff.linear_out.weight", ".ff_context.linear_out.weight"))
    ) or (key.startswith("single_transformer_blocks.") and key.endswith(".attn.to_out.weight"))


def weight_layout(kind: str, dtype: FloatDType) -> str:
    name = {
        FloatDType.BFLOAT16: "bf16",
        FloatDType.FLOAT16: "fp16",
        FloatDType.FLOAT32: "fp32",
    }[dtype]
    suffix = "down-pretransposed-v2" if kind == "transformer" else "v1"
    return f"diffusers-{name}-{suffix}"
