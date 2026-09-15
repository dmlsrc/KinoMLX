"""FP32 parity between Krea's padded reference denoiser and the compact default.

Reference mode keeps every text slot, applies the boolean mask, aligns the combined
sequence to 256 rows, and consumes source-layout feed-forward weights. The default
path gathers only the valid text rows, passes no mask, adds no alignment rows, and
consumes baked pretransposed layouts. With identical logical weights and FP32
activations, every valid boundary row and the image prediction must agree at FP32
rounding level. A larger gap would be a logic difference in compaction rather than
the low-precision rounding that separates the two paths on real BF16 weights.
"""

import mlx.core as mx
import pytest

import kinomlx.models.krea2.transformer.loading as loading_module
from kinomlx.modeling.layers import PretransposedLinear
from kinomlx.models.krea2.transformer.inputs import compact_text_groups
from kinomlx.models.krea2.transformer.model import Krea2Transformer

from ._transformer_fixtures import populate_transformer, tiny_transformer_config

TEXT_SLOTS = 12
IMAGE_ROWS = 2
IMAGE_COLS = 3
IMAGE_TOKENS = IMAGE_ROWS * IMAGE_COLS
# Krea prompts keep five valid suffix tokens after the middle padding.
SUFFIX = (7, 8, 9, 10, 11)
MAX_ABS = 1e-5
MAX_RELATIVE_L2 = 1e-5

CASES: dict[str, tuple[tuple[int, ...], ...]] = {
    # Same valid count, different valid slots: one unmasked group of two rows.
    "equal": ((0, 1, 2, 3, *SUFFIX), (0, 1, 2, 4, *SUFFIX)),
    # Different valid counts: two single-row groups restored to batch order.
    "mixed": ((0, 1, 2, 3, *SUFFIX), (0, 1, *SUFFIX)),
    # No text padding: the default path skips gathering; reference still aligns to 256.
    "full": (tuple(range(TEXT_SLOTS)),),
}
TEXT_BOUNDARIES = frozenset({"text_fusion", "text_projection"})
COMBINED_BOUNDARIES = frozenset({"combined_input", "rotary_cos", "rotary_sin"})


def _paired_models() -> tuple[Krea2Transformer, Krea2Transformer]:
    """Build reference and default graphs sharing one set of logical weights."""
    config = tiny_transformer_config()
    reference = Krea2Transformer(config, reference_denoise=True)
    optimized = Krea2Transformer(config)
    reference_targets = loading_module.transformer_weight_targets(reference)
    optimized_targets = loading_module.transformer_weight_targets(optimized)
    assert set(reference_targets) == set(optimized_targets)
    populate_transformer(reference, reference_targets, dtype=mx.float32)
    for name, target in optimized_targets.items():
        source = reference_targets[name]
        value = getattr(source.owner, source.attribute)
        if isinstance(target.owner, PretransposedLinear):
            assert name.startswith("transformer_blocks.")
            assert name.endswith(".ff.down.weight")
            value = mx.contiguous(value.T)
        setattr(target.owner, target.attribute, value)
    assert all(
        isinstance(block.ff.down, PretransposedLinear) for block in optimized.transformer_blocks
    )
    assert not any(
        isinstance(block.ff.down, PretransposedLinear) for block in reference.transformer_blocks
    )
    return reference, optimized


def _inputs(
    valid_text: tuple[tuple[int, ...], ...],
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    config = tiny_transformer_config()
    batch = len(valid_text)
    mx.random.seed(20260902)
    image = mx.random.uniform(low=-0.2, high=0.2, shape=(batch, IMAGE_TOKENS, config.in_channels))
    # Padded slots carry random content so the comparison also proves they cannot leak.
    context = mx.random.uniform(
        low=-0.2,
        high=0.2,
        shape=(batch, TEXT_SLOTS, config.num_text_layers, config.text_hidden_dim),
    )
    timestep = mx.array((0.75, 0.3)[:batch])
    # Distinct positions per row make a scrambled gather visible in the rotary tables.
    text_positions = [(float(slot + 1), 0.0, 0.0) for slot in range(TEXT_SLOTS)]
    image_positions = [
        (0.0, float(row + 1), float(col + 1))
        for row in range(IMAGE_ROWS)
        for col in range(IMAGE_COLS)
    ]
    positions = mx.array([text_positions + image_positions] * batch, dtype=mx.float32)
    mask = mx.array(
        [
            [slot in valid for slot in range(TEXT_SLOTS)] + [True] * IMAGE_TOKENS
            for valid in valid_text
        ],
        dtype=mx.bool_,
    )
    return image, context, timestep, positions, mask


def _run(
    model: Krea2Transformer, inputs: tuple[mx.array, ...]
) -> tuple[object, dict[str, mx.array]]:
    image, context, timestep, positions, mask = inputs
    observed: dict[str, mx.array] = {}
    output = model(
        image,
        encoder_hidden_states=context,
        timestep=timestep,
        position_ids=positions,
        attention_mask=mask,
        observer=lambda name, value: observed.__setitem__(name, value),
    )
    mx.eval(output.sample, *observed.values())
    return output, observed


def _metrics(actual: mx.array, expected: mx.array) -> tuple[float, float]:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == mx.float32
    difference = (actual - expected).astype(mx.float32)
    max_abs = mx.max(mx.abs(difference)).item()
    scale = max(mx.sqrt(mx.sum(mx.square(expected))).item(), 1e-30)
    return max_abs, mx.sqrt(mx.sum(mx.square(difference))).item() / scale


def _rows(name: str, valid: tuple[int, ...]) -> tuple[list[int], list[int]] | None:
    """Reference rows and the compact rows holding the same tokens, or None for whole tensors."""
    compact = len(valid)
    if name in TEXT_BOUNDARIES:
        return list(valid), list(range(compact))
    if name in COMBINED_BOUNDARIES or name.startswith("block_"):
        image_rows = [TEXT_SLOTS + token for token in range(IMAGE_TOKENS)]
        return list(valid) + image_rows, list(range(compact + IMAGE_TOKENS))
    return None


def _group_rows(inputs: tuple[mx.array, ...]) -> dict[int, tuple[str, int]]:
    """Map each batch row to the observer prefix and row index the default path used."""
    _image, context, _timestep, positions, mask = inputs
    groups = compact_text_groups(context, positions, mask)
    if len(groups) == 1:
        return {row: ("", row) for row in groups[0].rows}
    return {
        row: (f"group_{index:02d}.", position)
        for index, group in enumerate(groups)
        for position, row in enumerate(group.rows)
    }


@pytest.mark.requires_metal
@pytest.mark.parametrize("valid_text", list(CASES.values()), ids=list(CASES))
def test_compact_default_matches_padded_reference_in_fp32(
    valid_text: tuple[tuple[int, ...], ...],
) -> None:
    reference, optimized = _paired_models()
    inputs = _inputs(valid_text)
    batch = len(valid_text)

    reference_output, reference_observed = _run(reference, inputs)
    optimized_output, optimized_observed = _run(optimized, inputs)

    assert reference_output.padded_sequence_length == 256
    assert reference_output.padding_tokens == 256 - (TEXT_SLOTS + IMAGE_TOKENS)
    assert optimized_output.padding_tokens == 0
    assert optimized_output.combined_sequence_length == (
        max(len(valid) for valid in valid_text) + IMAGE_TOKENS
    )
    assert optimized_output.sample.shape == reference_output.sample.shape

    group_rows = _group_rows(inputs)
    observed_names = {name.split(".")[-1] for name in optimized_observed}
    assert observed_names == set(reference_observed)

    # Walk boundaries in forward order so the first failure names the earliest divergence.
    for name, reference_value in reference_observed.items():
        for row, valid in enumerate(valid_text):
            prefix, compact_row = group_rows[row]
            optimized_value = optimized_observed[prefix + name]
            rows = _rows(name, valid)
            if rows is None:
                expected = reference_value[row]
                actual = optimized_value[compact_row]
            else:
                reference_rows, compact_rows = rows
                expected = reference_value[row, mx.array(reference_rows)]
                actual = optimized_value[compact_row, mx.array(compact_rows)]
            max_abs, relative_l2 = _metrics(actual, expected)
            where = f"{name} batch row {row}"
            assert max_abs <= MAX_ABS, f"{where}: max_abs={max_abs:.3e}"
            assert relative_l2 <= MAX_RELATIVE_L2, f"{where}: relative_l2={relative_l2:.3e}"

    max_abs, relative_l2 = _metrics(optimized_output.sample, reference_output.sample)
    assert max_abs <= MAX_ABS, f"sample: max_abs={max_abs:.3e}"
    assert relative_l2 <= MAX_RELATIVE_L2, f"sample: relative_l2={relative_l2:.3e}"
    assert mx.all(mx.isfinite(optimized_output.sample)).item()
    assert batch == optimized_output.sample.shape[0]
