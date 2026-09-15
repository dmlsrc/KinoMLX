"""Bounded-residency LTX prompt encoding orchestration."""

import gc
from pathlib import Path
from typing import Protocol, cast

import mlx.core as mx

from kinomlx.reporting import NullReporter, Reporter

from . import loading as _loading
from .encoder import (
    AudioVideoGemmaEncoderOutput,
    create_av_text_encoder_v2_from_checkpoint,
)
from .gemma3 import Gemma3Config, Gemma3Model
from .gemma3_loading import load_gemma3_weights
from .gemma4 import Gemma4Config, Gemma4Model
from .gemma4_loading import load_gemma4_weights
from .tokenizer import GemmaTokenizer
from .tokenizer_derivation import TokenizerDerivation


class _GemmaModel(Protocol):
    def __call__(
        self,
        input_ids: mx.array,
        *,
        attention_mask: mx.array | None = None,
        output_hidden_states: bool = True,
        reporter: Reporter | None = None,
    ) -> tuple[mx.array, tuple[mx.array, ...] | None]: ...


def _trim_left_padding(
    hidden_states: tuple[mx.array, ...],
    attention_mask: mx.array,
) -> tuple[tuple[mx.array, ...], mx.array]:
    """Retain real-token Gemma states after a stock left-padded forward."""
    real_tokens = int(cast(int | float, mx.sum(attention_mask).item()))
    if real_tokens <= 0:
        raise ValueError("prompt attention mask contains no real tokens")
    if real_tokens == attention_mask.shape[1]:
        return hidden_states, attention_mask
    return (
        tuple(mx.contiguous(state[:, -real_tokens:, :]) for state in hidden_states),
        mx.contiguous(attention_mask[:, -real_tokens:]),
    )


def encode_prompt(
    prompt: str,
    *,
    gemma_path: Path | str,
    connector_path: Path | str,
    config_path: Path | str,
    model_generation: str = "2.3",
    projection_path: Path | str | None = None,
    tokenizer: TokenizerDerivation | None = None,
    max_length: int = 1024,
    pad_prompt_to_max: bool = True,
    reporter: Reporter | None = None,
) -> AudioVideoGemmaEncoderOutput:
    """Run the selected Gemma and packaged AV station with bounded residency.

    The default runs Gemma on the reference's left-padded 1,024-token sequence,
    then retains only real-token states before loading the connectors. Set
    ``pad_prompt_to_max=False`` for the faster variable-length path, which can
    introduce small BF16 drift relative to stock prompt conditioning.
    """
    if model_generation not in {"2.3", "2.5"}:
        raise ValueError(f"unsupported LTX text generation {model_generation!r}")
    if model_generation == "2.5" and projection_path is None:
        projection_path = gemma_path
    sink = reporter if reporter is not None else NullReporter()
    encoder_tokenizer = None
    gemma: _GemmaModel | None = None
    input_ids = None
    last_hidden = None
    raw_hidden_states = None
    try:
        encoder_tokenizer = GemmaTokenizer(gemma_path if tokenizer is None else tokenizer)
        input_ids, attention_mask = encoder_tokenizer.encode(
            prompt,
            max_length=max_length,
            pad_to_max=pad_prompt_to_max,
        )
        if model_generation == "2.5":
            gemma = Gemma4Model(Gemma4Config())
            load_gemma4_weights(gemma, gemma_path, reporter=sink)
        else:
            gemma = Gemma3Model(Gemma3Config())
            load_gemma3_weights(gemma, gemma_path, reporter=sink)
        last_hidden, raw_hidden_states = gemma(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            reporter=sink,
        )
        if raw_hidden_states is None:
            raise RuntimeError(f"LTX-{model_generation} Gemma did not return hidden states")
        hidden_states = tuple(raw_hidden_states)
        if pad_prompt_to_max:
            hidden_states, attention_mask = _trim_left_padding(
                hidden_states,
                attention_mask,
            )
        mx.eval(*hidden_states, attention_mask)
    finally:
        del last_hidden, raw_hidden_states, gemma, encoder_tokenizer, input_ids
        gc.collect()
        mx.clear_cache()

    encoder = None
    try:
        encoder = create_av_text_encoder_v2_from_checkpoint(config_path)
        _loading.load_av_text_encoder_v2_weights(
            encoder,
            connector_path,
            projection_path=projection_path,
            reporter=sink,
        )
        output = encoder(hidden_states, attention_mask, reporter=sink)
        mx.eval(output.video_encoding, output.audio_encoding, output.attention_mask)
        return output
    finally:
        del hidden_states, attention_mask, encoder
        gc.collect()
        mx.clear_cache()


__all__ = ["encode_prompt"]
