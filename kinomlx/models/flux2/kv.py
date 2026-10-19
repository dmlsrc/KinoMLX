"""Reference-only K/V owned by one denoising invocation, never by the model."""

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class LayerKV:
    key: mx.array | None = None
    value: mx.array | None = None

    def store(self, key: mx.array, value: mx.array, start: int, count: int) -> None:
        # A contiguous slice can retain the entire projection allocation. Gather
        # owns exactly the reference rows, including a one-token reference.
        indices = mx.arange(start, start + count)
        self.key = mx.take(key, indices, axis=1)
        self.value = mx.take(value, indices, axis=1)
        # Settle this layer's compact copies before constructing later blocks.
        # Otherwise pending gathers retain all full-sequence projections until
        # the final velocity is evaluated, inflating the extraction peak.
        mx.eval(self.key, self.value)

    def tensors(self) -> tuple[mx.array, mx.array]:
        if self.key is None or self.value is None:
            raise ValueError("Reference K/V has not been extracted")
        return self.key, self.value


class ReferenceKV:
    def __init__(self, reference_tokens: int, layers: int) -> None:
        if reference_tokens < 1 or layers < 1:
            raise ValueError("Reference K/V requires positive token and layer counts")
        self.reference_tokens = reference_tokens
        self.layers = [LayerKV() for _ in range(layers)]
        self.extracting = True
        self.closed = False

    def tensors(self) -> tuple[mx.array, ...]:
        return tuple(*layer.tensors() for layer in self.layers)

    def clear(self) -> None:
        for layer in self.layers:
            layer.key = layer.value = None
        self.closed = True


def attend(query: mx.array, key: mx.array, value: mx.array) -> mx.array:
    return mx.fast.scaled_dot_product_attention(
        query.transpose(0, 2, 1, 3),
        key.transpose(0, 2, 1, 3),
        value.transpose(0, 2, 1, 3),
        scale=query.shape[-1] ** -0.5,
    ).transpose(0, 2, 1, 3)


def reference_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    text_tokens: int,
    state: ReferenceKV,
    layer: int,
) -> mx.array:
    """Post-RoPE attention: [text, reference, target] or cached [text, target]."""
    cached = state.layers[layer]
    start = text_tokens
    if not state.extracting:
        rk, rv = cached.tensors()
        return attend(
            query,
            mx.concatenate([key[:, :start], rk, key[:, start:]], axis=1),
            mx.concatenate([value[:, :start], rv, value[:, start:]], axis=1),
        )
    end = start + state.reference_tokens
    cached.store(key, value, start, state.reference_tokens)
    active = attend(mx.concatenate([query[:, :start], query[:, end:]], axis=1), key, value)
    reference = attend(query[:, start:end], key[:, start:end], value[:, start:end])
    return mx.concatenate([active[:, :start], reference, active[:, start:]], axis=1)


def blend_modulation(
    current: mx.array, reference: mx.array, start: int, count: int, length: int
) -> mx.array:
    """Use fixed-reference modulation only at its ordered sequence positions."""
    shape = (current.shape[0], length, current.shape[-1])
    expanded = mx.broadcast_to(current[:, None], shape)
    fixed = mx.broadcast_to(reference[:, None], (shape[0], count, shape[-1]))
    return mx.concatenate([expanded[:, :start], fixed, expanded[:, start + count :]], axis=1)
