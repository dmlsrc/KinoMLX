"""Allocation-light Qwen3-VL language graph used by Krea 2.

The final language-model norm, logits, and KV caching are not constructed.
Optional image spans carry features from a separately owned vision backbone.
"""

from dataclasses import dataclass

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.modeling.layers import embedding_shell
from kinomlx.modeling.qwen import QwenDecoderLayer
from kinomlx.reporting import NullReporter, Reporter

from ..metadata import Krea2TextConfig
from .multimodal import Qwen3VLImageSpan, multimodal_positions

_INTEGER_DTYPES = frozenset(
    (mx.int8, mx.int16, mx.int32, mx.int64, mx.uint8, mx.uint16, mx.uint32, mx.uint64)
)


class Qwen3VLTextRotaryEmbedding:
    """Three-axis interleaved MRoPE table for the Qwen text graph."""

    def __init__(self, config: Krea2TextConfig) -> None:
        if sum(config.mrope_section) * 2 != config.head_dim:
            raise ValueError("Qwen MRoPE sections must cover the complete head dimension")
        self.head_dim = config.head_dim
        self.inv_freq = 1.0 / (
            config.rope_theta
            ** (mx.arange(0, config.head_dim, 2, dtype=mx.float32) / config.head_dim)
        )
        selector = [0] * (config.head_dim // 2)
        for axis, offset in enumerate((1, 2), start=1):
            for index in range(offset, config.mrope_section[axis] * 3, 3):
                selector[index] = axis
        self.axis_selector = mx.array(selector, dtype=mx.int32)[None, None, :]

    def __call__(
        self,
        value: mx.array,
        position_ids: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError("Qwen position_ids must have shape [3, batch, sequence]")
        frequencies = (
            position_ids.astype(mx.float32)[..., None] * self.inv_freq[None, None, None, :]
        )
        selected = mx.where(
            self.axis_selector == 1,
            frequencies[1],
            mx.where(self.axis_selector == 2, frequencies[2], frequencies[0]),
        )
        embedding = mx.concatenate([selected, selected], axis=-1)
        return (
            mx.cos(embedding).astype(value.dtype),
            mx.sin(embedding).astype(value.dtype),
        )


@dataclass(frozen=True)
class Qwen3VLTextOutput:
    """Only the selected intermediate states Krea consumes."""

    hidden_states: mx.array
    selected_layers: tuple[int, ...]


class Qwen3VLTextModel(nn.Module):
    """Qwen text backbone truncated immediately after the last selected tap."""

    def __init__(
        self,
        config: Krea2TextConfig,
        selected_layers: tuple[int, ...],
    ) -> None:
        super().__init__()
        if not selected_layers:
            raise ValueError("Krea requires at least one selected Qwen hidden state")
        if tuple(sorted(set(selected_layers))) != selected_layers:
            raise ValueError("Krea selected Qwen layers must be sorted and unique")
        if selected_layers[0] <= 0 or selected_layers[-1] > config.num_hidden_layers:
            raise ValueError("Krea selected Qwen layer is outside the source graph")
        self.config = config
        self.selected_layers = selected_layers
        self.executed_layers = selected_layers[-1]
        self.embed_tokens = embedding_shell()
        self.layers = [QwenDecoderLayer(config) for _index in range(self.executed_layers)]
        self.rotary = Qwen3VLTextRotaryEmbedding(config)

    @staticmethod
    def _attention_mask(attention_mask: mx.array, length: int) -> mx.array:
        causal = mx.tril(mx.ones((length, length), dtype=mx.bool_), k=0)
        valid_keys = attention_mask[:, None, None, :].astype(mx.bool_)
        return causal[None, None, :, :] & valid_keys

    @staticmethod
    def _validate_positions(
        attention_mask: mx.array,
        position_ids: mx.array,
    ) -> mx.array:
        if position_ids.dtype not in _INTEGER_DTYPES:
            raise ValueError("Qwen position_ids must use an integer dtype")
        validated = position_ids.astype(mx.int32)
        valid = attention_mask.astype(mx.bool_)
        cumulative = mx.cumsum(valid.astype(mx.int32), axis=-1) - 1
        expected_row = mx.where(valid, cumulative, mx.array(1, dtype=mx.int32))
        expected = mx.broadcast_to(expected_row[None, :, :], validated.shape)
        if not mx.array_equal(validated, expected).item():
            raise ValueError(
                "Qwen position_ids must use cumulative valid positions and literal 1 padding"
            )
        return validated

    def __call__(
        self,
        input_ids: mx.array,
        *,
        attention_mask: mx.array,
        position_ids: mx.array,
        reporter: Reporter | None = None,
        image_spans: tuple[Qwen3VLImageSpan, ...] = (),
    ) -> Qwen3VLTextOutput:
        if input_ids.ndim != 2:
            raise ValueError("Qwen input_ids must have shape [batch, sequence]")
        batch, length = input_ids.shape
        if tuple(attention_mask.shape) != (batch, length):
            raise ValueError("Qwen attention_mask must match input_ids")
        if tuple(position_ids.shape) != (3, batch, length):
            raise ValueError("Qwen position_ids must have shape [3, batch, sequence]")
        if image_spans:
            if position_ids.dtype not in _INTEGER_DTYPES:
                raise ValueError("Qwen position_ids must use an integer dtype")
            if batch != 1 or not mx.all(attention_mask).item():
                raise ValueError("Qwen image grounding requires one unpadded prompt")
            expected = multimodal_positions(length, image_spans)
            if not mx.array_equal(position_ids, expected).item():
                raise ValueError("Qwen image positions disagree with the supplied image spans")
            for span in image_spans:
                shape = (span.length, self.config.hidden_size)
                features = span.features
                if len(features.deepstack) > self.executed_layers:
                    raise ValueError("Qwen visual residuals exceed the executed language layers")
                if features.embeddings.shape != shape or any(
                    value.shape != shape for value in features.deepstack
                ):
                    raise ValueError("Qwen visual features disagree with image grid or text width")
        else:
            position_ids = self._validate_positions(attention_mask, position_ids)

        sink = reporter if reporter is not None else NullReporter()
        phase = "encode prompt with Qwen3-VL"
        # Indeterminate phase: the executed layers form one lazy graph that is
        # evaluated once at the tap stack. Evaluating after every layer only
        # bought per-layer progress and cost about 16 percent of the phase
        # with no peak-memory benefit.
        sink.phase_start(phase)
        try:
            hidden = self.embed_tokens(input_ids)
            for span in image_spans:
                hidden[0, span.start : span.start + span.length] = span.features.embeddings.astype(
                    hidden.dtype
                )
            mask = "causal" if image_spans else self._attention_mask(attention_mask, length)
            rotary = self.rotary(hidden, position_ids)
            selected: list[mx.array] = []
            selected_set = frozenset(self.selected_layers)
            for completed_layers, layer in enumerate(self.layers, start=1):
                hidden = layer(hidden, mask, rotary)
                for span in image_spans:
                    if completed_layers <= len(span.features.deepstack):
                        hidden = hidden.at[0, span.start : span.start + span.length].add(
                            span.features.deepstack[completed_layers - 1].astype(hidden.dtype)
                        )
                if completed_layers in selected_set:
                    selected.append(hidden)
            if len(selected) != len(self.selected_layers):
                raise RuntimeError("Qwen execution did not produce every selected Krea tap")
            stacked = mx.stack(selected, axis=2)
            mx.eval(stacked)
            return Qwen3VLTextOutput(
                hidden_states=stacked,
                selected_layers=self.selected_layers,
            )
        finally:
            sink.phase_end(phase)


__all__ = [
    "Qwen3VLTextModel",
    "Qwen3VLTextOutput",
    "Qwen3VLTextRotaryEmbedding",
]
