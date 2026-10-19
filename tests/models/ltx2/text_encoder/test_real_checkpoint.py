"""Opt-in end-to-end prompt encoding with official local weights."""

import gc

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.resources import ComponentKind
from kinomlx.models.ltx2.text_encoder import encode_prompt
from tests.models.ltx2._source_weights import (
    NO_LTX23_CHECKPOINT,
    ltx2_resources,
    ltx23_checkpoint,
)


def _gemma_snapshot():
    """The Gemma weights directory the product selects for LTX-2.3, or None."""
    resources = ltx2_resources()
    encoder = None if resources is None else resources.optional(ComponentKind.TEXT_ENCODER)
    if encoder is None or not tuple(encoder.source_path.glob("model-*.safetensors")):
        return None
    return encoder.source_path


@pytest.mark.heavy
def test_real_ltx23_prompt_encoding() -> None:
    checkpoint = ltx23_checkpoint()
    if checkpoint is None:
        pytest.skip(NO_LTX23_CHECKPOINT)
    gemma = _gemma_snapshot()
    if gemma is None:
        pytest.skip("the product selects no cached Gemma 3 12B weights")
    connector = checkpoint
    output = None
    try:
        output = encode_prompt(
            "a red cube",
            text_encoder_path=gemma,
            connector_path=connector,
            config_path=checkpoint,
        )
        assert tuple(output.video_encoding.shape) == (1, 1024, 4096)
        assert tuple(output.audio_encoding.shape) == (1, 1024, 2048)
        assert tuple(output.attention_mask.shape) == (1, 1024)
        assert mx.all(mx.isfinite(output.video_encoding)).item()
        assert mx.all(mx.isfinite(output.audio_encoding)).item()
        assert mx.all(output.attention_mask == 1).item()
    finally:
        del output
        gc.collect()
        mx.clear_cache()
