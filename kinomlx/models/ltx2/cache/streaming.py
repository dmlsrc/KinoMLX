"""Transformer block streaming over sharded disposable caches."""

from __future__ import annotations

import gc
from pathlib import Path

import mlx.core as mx

import kinomlx._mlx_nn as nn

from .keys import flatten_to_nested
from .layout import clear_block_layout_weights, install_block_layout_weight
from .quantization import (
    prepare_block_quantized_linears,
    quant_bases_for_block_keys,
    restore_block_quantized_linears,
)
from .schema import (
    LAYOUT_KEY_PREFIX,
    QUANT_KEY_PREFIX,
    TRANSFORMER_CACHE_QUANTIZE_OFF,
)
from .storage import load_cache_weights
from .validation import validate_model_cache_graph


class TransformerBlockStreamer:
    """Bind cached transformer-block weights into a small resident block pool."""

    def __init__(
        self,
        cache_file: Path,
        *,
        expected_model: nn.Module | None = None,
        include_audio: bool | None = None,
        expected_block_schemas: tuple[frozenset[str], ...] | None = None,
        transformer_cache_quantize: str = TRANSFORMER_CACHE_QUANTIZE_OFF,
        video_ff_quantize_specs: tuple[tuple[str, str], ...] = (),
        video_ff_quantize_group_size: int | None = None,
        video_ff_quantize_bits: int | None = None,
    ) -> None:
        self.cache_file = Path(cache_file)
        self.transformer_cache_quantize = transformer_cache_quantize
        self.video_ff_quantize_specs = tuple(video_ff_quantize_specs)
        self.video_ff_quantize_group_size = video_ff_quantize_group_size
        self.video_ff_quantize_bits = video_ff_quantize_bits
        self._weights = load_cache_weights(self.cache_file)
        if expected_model is not None:
            validate_model_cache_graph(
                expected_model,
                self._weights,
                include_audio=include_audio,
            )
        self._block_keys: dict[int, list[tuple[str, str]]] = {}
        self._layout_keys: dict[int, list[tuple[str, str]]] = {}
        self._quant_keys: dict[int, list[tuple[str, str]]] = {}
        self._non_block_weights: dict[str, mx.array] = {}

        for full_key in list(self._weights):
            is_layout = full_key.startswith(LAYOUT_KEY_PREFIX)
            is_quant = full_key.startswith(QUANT_KEY_PREFIX)
            if is_layout:
                logical_key = full_key[len(LAYOUT_KEY_PREFIX) :]
            elif is_quant:
                logical_key = full_key[len(QUANT_KEY_PREFIX) :]
            else:
                logical_key = full_key
            parts = logical_key.split(".")
            if len(parts) < 3 or parts[0] != "transformer_blocks":
                self._non_block_weights[full_key] = self._weights.pop(full_key)
                continue
            try:
                block_index = int(parts[1])
            except ValueError:
                self._weights.pop(full_key, None)
                continue
            block_key = ".".join(parts[2:])
            if is_layout:
                target = self._layout_keys
            elif is_quant:
                target = self._quant_keys
            else:
                target = self._block_keys
            target.setdefault(block_index, []).append((full_key, block_key))

        discovered = set(self._block_keys) | set(self._layout_keys) | set(self._quant_keys)
        if not discovered:
            raise ValueError(f"No transformer block weights found in cache {self.cache_file}")
        self.block_count = max(discovered) + 1
        if expected_block_schemas is not None:
            expected_count = len(expected_block_schemas)
            if discovered != set(range(expected_count)):
                raise ValueError(
                    f"Transformer cache expected {expected_count} transformer blocks, "
                    f"found {self.block_count} (indices {sorted(discovered)})"
                )
            self._validate_block_schemas(expected_block_schemas)
        missing = [index for index in range(self.block_count) if index not in discovered]
        if missing:
            raise ValueError(
                f"Transformer cache {self.cache_file} is missing block weights for layers {missing}"
            )
        self.loaded_count = (
            sum(map(len, self._block_keys.values()))
            + sum(map(len, self._layout_keys.values()))
            + sum(map(len, self._quant_keys.values()))
        )
        self.layout_count = sum(map(len, self._layout_keys.values()))
        self.quant_count = sum(map(len, self._quant_keys.values()))

    def _validate_block_schemas(
        self,
        expected_block_schemas: tuple[frozenset[str], ...],
    ) -> None:
        for block_index, expected in enumerate(expected_block_schemas):
            normal = {block_key for _full_key, block_key in self._block_keys.get(block_index, ())}
            layout: set[str] = set()
            for _full_key, block_key in self._layout_keys.get(block_index, ()):
                if not block_key.endswith(".weight_t"):
                    raise ValueError(
                        f"Transformer cache block {block_index} has unsupported "
                        f"layout tensor {block_key}"
                    )
                layout.add(f"{block_key[: -len('.weight_t')]}.weight")

            quant_parameters: dict[str, set[str]] = {}
            for _full_key, block_key in self._quant_keys.get(block_index, ()):
                base, separator, parameter = block_key.rpartition(".")
                if not separator or parameter not in {"weight", "scales", "biases"}:
                    raise ValueError(
                        f"Transformer cache block {block_index} has unsupported "
                        f"quantized tensor {block_key}"
                    )
                quant_parameters.setdefault(base, set()).add(parameter)
            quant: set[str] = set()
            for base, parameters in quant_parameters.items():
                missing_parameters = {"weight", "scales"} - parameters
                if missing_parameters:
                    missing_names = ", ".join(sorted(missing_parameters))
                    raise ValueError(
                        f"Transformer cache block {block_index} has incomplete "
                        f"quantized target {base}: missing {missing_names}"
                    )
                quant.add(f"{base}.weight")

            collisions = (normal & layout) | (normal & quant) | (layout & quant)
            if collisions:
                raise ValueError(
                    f"Transformer cache block {block_index} represents parameters "
                    f"more than once: {sorted(collisions)}"
                )
            actual = normal | layout | quant
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            if missing or extra:
                details = []
                if missing:
                    details.append(f"missing {missing}")
                if extra:
                    details.append(f"unexpected {extra}")
                raise ValueError(
                    f"Transformer cache block {block_index} schema mismatch: {'; '.join(details)}"
                )

    def take_non_block_weights(self) -> dict[str, mx.array]:
        """Transfer top-level tensors to the model loader exactly once."""
        weights = self._non_block_weights
        self._non_block_weights = {}
        return weights

    def bind(
        self,
        block: nn.Module,
        block_idx: int,
        *,
        evict_block_idx: int | None = None,
    ) -> nn.Module:
        """Load one logical block's cached weights into a resident module."""
        if not 0 <= block_idx < self.block_count:
            raise IndexError(f"block index {block_idx} is outside 0-{self.block_count - 1}")
        if evict_block_idx is not None and evict_block_idx != block_idx:
            for table in (
                self._block_keys,
                self._layout_keys,
                self._quant_keys,
            ):
                for full_key, _block_key in table.get(evict_block_idx, ()):
                    self._weights.pop(full_key, None)

        normal_keys = self._block_keys.get(block_idx, ())
        layout_keys = self._layout_keys.get(block_idx, ())
        quant_keys = self._quant_keys.get(block_idx, ())
        sample = next(
            (items[0][0] for items in (normal_keys, layout_keys, quant_keys) if items),
            None,
        )
        if sample is not None and sample not in self._weights:
            self._weights = load_cache_weights(self.cache_file)
            self._drop_non_block_keys()

        quant_bases = quant_bases_for_block_keys(quant_keys)
        restore_block_quantized_linears(block, keep_bases=quant_bases)
        clear_block_layout_weights(block)
        if quant_bases:
            prepare_block_quantized_linears(
                block,
                quant_bases,
                quant_keys,
                normal_keys,
                transformer_cache_quantize=self.transformer_cache_quantize,
                quantization_specs=self.video_ff_quantize_specs,
                group_size=self.video_ff_quantize_group_size,
                bits=self.video_ff_quantize_bits,
            )
        normal_weights = {block_key: self._weights[full_key] for full_key, block_key in normal_keys}
        if normal_weights:
            block.update(flatten_to_nested(normal_weights))
        quant_weights = {block_key: self._weights[full_key] for full_key, block_key in quant_keys}
        if quant_weights:
            block.update(flatten_to_nested(quant_weights))
        for full_key, layout_key in layout_keys:
            install_block_layout_weight(
                block,
                layout_key,
                self._weights[full_key],
            )
        if hasattr(block, "idx"):
            block.idx = block_idx
        return block

    def close(self) -> None:
        """Release mmap-backed arrays and key tables held by the streamer."""
        self._weights = {}
        self._block_keys = {}
        self._layout_keys = {}
        self._quant_keys = {}
        self._non_block_weights = {}
        gc.collect()

    def _drop_non_block_keys(self) -> None:
        for full_key in list(self._weights):
            if full_key.startswith(LAYOUT_KEY_PREFIX):
                logical_key = full_key[len(LAYOUT_KEY_PREFIX) :]
            elif full_key.startswith(QUANT_KEY_PREFIX):
                logical_key = full_key[len(QUANT_KEY_PREFIX) :]
            else:
                logical_key = full_key
            if not logical_key.startswith("transformer_blocks."):
                self._weights.pop(full_key, None)


__all__ = ["TransformerBlockStreamer"]
