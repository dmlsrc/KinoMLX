"""Measured Klein transformer storage layouts, separate from logical weight shapes."""


def pretransposed_weight(key: str) -> bool:
    """Select repeated FF-down and fused attention/MLP-down projections."""
    return (
        key.startswith("transformer_blocks.")
        and key.endswith((".ff.linear_out.weight", ".ff_context.linear_out.weight"))
    ) or (key.startswith("single_transformer_blocks.") and key.endswith(".attn.to_out.weight"))


def weight_layout(kind: str, dtype: str) -> str:
    name = {"bfloat16": "bf16", "float16": "fp16"}[dtype]
    suffix = "down-pretransposed-v2" if kind == "transformer" else "v1"
    return f"diffusers-{name}-{suffix}"
