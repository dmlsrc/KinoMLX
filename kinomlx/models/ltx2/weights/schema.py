"""Weight layout tags and quantization modes."""

from enum import StrEnum

import mlx.core as mx

from kinomlx.modeling.dtypes import FloatDType, float_dtype_name

LAYOUT_KEY_PREFIX = "__layout__."
QUANT_KEY_PREFIX = "__quant__."


class TransformerQuantize(StrEnum):
    """How the transformer's feed-forward weights are stored for execution."""

    OFF = "off"
    MXFP8_BLOCKS = "mxfp8-blocks"
    MXFP8_BLOCKS_PRETRANSPOSE = "mxfp8-blocks-pretranspose"


TRANSFORMER_QUANTIZE_MODES = tuple(item.value for item in TransformerQuantize)


def dtype_payload_name(dtype: FloatDType | mx.Dtype | None) -> FloatDType | None:
    """Name the dtype a preparation policy reports, leaving an absent one absent."""
    return None if dtype is None else float_dtype_name(dtype)
