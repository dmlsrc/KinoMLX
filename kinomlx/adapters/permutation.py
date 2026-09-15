"""Torch CPU seeded permutation prefixes without a Torch runtime dependency.

FourierFT saves coefficients and a seed, not their locations. The generator
and shuffle convention therefore form part of its serialized representation.
Only the requested prefix is retained; no full model-sized permutation exists.
"""

import random
from functools import lru_cache
from typing import cast

import mlx.core as mx

from kinomlx.io.buffer import mlx_array_from_buffer

_LARGE_PERMUTATION = (2**32 - 1) // 20
_CHUNK = 262144


def _generator(seed: int) -> random.Random:
    # Both engines use MT19937. Python's seed expansion is different, so install
    # the integer-seeded state explicitly rather than calling Random(seed).
    state = [seed & 0xFFFFFFFF]
    for index in range(1, 624):
        previous = state[-1]
        state.append((1812433253 * (previous ^ (previous >> 30)) + index) & 0xFFFFFFFF)
    generator = random.Random()
    generator.setstate((3, (*state, 624), None))
    return generator


def _inside_out_prefix(size: int, count: int, generator: random.Random) -> tuple[int, ...]:
    # The large-size CPU branch uses 64-bit draws and inside-out shuffling.
    # After the first count entries, slot j < count changes only when z == j;
    # its final value is the greatest index that wrote that slot. Bounded CPU
    # batches reduce these writes without retaining or iterating a huge list.
    prefix: list[int] = []
    for index in range(count):
        draw = (generator.getrandbits(32) << 32) | generator.getrandbits(32)
        selected = draw % (index + 1)
        prefix.append(index if selected == index else prefix[selected])
        prefix[selected] = index
    with mx.stream(mx.cpu):
        result = mx.array([*prefix, 0], dtype=mx.uint64)
        for start in range(count, size, _CHUNK):
            stop = min(start + _CHUNK, size)
            raw = generator.getrandbits(64 * (stop - start)).to_bytes(8 * (stop - start), "little")
            words = mlx_array_from_buffer(memoryview(raw).cast("Q")).astype(mx.uint64)
            draws = (words << 32) | (words >> 32)
            indices = mx.arange(start, stop, dtype=mx.uint64)
            destinations = mx.minimum(draws % (indices + 1), count).astype(mx.uint32)
            result = result.at[destinations].maximum(indices)
            mx.eval(result)
        return tuple(cast(list[int], result[:count].tolist()))


@lru_cache(maxsize=32)
def torch_cpu_permutation_prefix(size: int, count: int, seed: int) -> tuple[int, ...]:
    """Match ``torch.randperm(size, CPU_generator(seed))[:count]`` exactly."""
    if not 0 <= count <= size or size < 1:
        raise ValueError("permutation prefix must fit a positive population")
    if count == 0:
        return ()
    generator = _generator(seed)
    if size >= _LARGE_PERMUTATION:
        return _inside_out_prefix(size, count, generator)
    positions: dict[int, int] = {}
    prefix = []
    for index in range(count):
        selected = (
            index if index == size - 1 else index + generator.getrandbits(32) % (size - index)
        )
        value = positions.get(selected, selected)
        positions[selected] = positions.pop(index, index)
        prefix.append(value)
    return tuple(prefix)
