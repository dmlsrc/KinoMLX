"""Weight layout tags and quantization modes."""

import mlx.core as mx

LAYOUT_KEY_PREFIX = "__layout__."
QUANT_KEY_PREFIX = "__quant__."

TRANSFORMER_QUANTIZE_OFF = "off"
TRANSFORMER_QUANTIZE_MXFP8_BLOCKS = "mxfp8-blocks"
TRANSFORMER_QUANTIZE_MXFP8_BLOCKS_PRETRANSPOSE = "mxfp8-blocks-pretranspose"
TRANSFORMER_QUANTIZE_MODES = (
    TRANSFORMER_QUANTIZE_OFF,
    TRANSFORMER_QUANTIZE_MXFP8_BLOCKS,
    TRANSFORMER_QUANTIZE_MXFP8_BLOCKS_PRETRANSPOSE,
)


def dtype_payload_name(dtype: str | mx.Dtype | None) -> str | None:
    """Map an MLX dtype to the spelling used by preparation policies."""
    if dtype is None:
        return None
    if dtype == mx.bfloat16:
        return "bfloat16"
    if dtype == mx.float16:
        return "float16"
    if dtype == mx.float32:
        return "float32"
    return str(dtype)
