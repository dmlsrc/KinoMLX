"""FourierFT seed fidelity, normalized inverse FFT, and bounded preflight."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from kinomlx.adapters.algebra import reconstruct_delta
from kinomlx.adapters.fusion import WeightedAdapterOperation
from kinomlx.adapters.peft import parse_configured_adapter
from kinomlx.adapters.permutation import (
    _generator,
    _inside_out_prefix,
    torch_cpu_permutation_prefix,
)
from kinomlx.modeling.loading import WeightTarget
from tests.adapters._fusion import fuse_and_check


def test_torch_cpu_seeded_prefix_golden():
    assert torch_cpu_permutation_prefix(24, 7, 777) == (23, 2, 19, 17, 1, 0, 15)
    assert torch_cpu_permutation_prefix(1, 1, 777) == (0,)
    assert len(set(torch_cpu_permutation_prefix(1009, 1009, 0))) == 1009
    assert torch_cpu_permutation_prefix(20000, 1500, 2**63 + 7) == torch_cpu_permutation_prefix(
        20000, 1500, 7
    )


@pytest.mark.parametrize("count", [1, 7, 41])
def test_large_shuffle_branch_matches_independent_inside_out(count, monkeypatch):
    from kinomlx.adapters import permutation

    monkeypatch.setattr(permutation, "_CHUNK", 9)
    generator = np.random.RandomState(777)
    draws = generator.randint(0, 2**32, size=82, dtype=np.uint32)
    expected = []
    for index in range(41):
        selected = ((int(draws[2 * index]) << 32) | int(draws[2 * index + 1])) % (index + 1)
        expected.append(index if selected == index else expected[selected])
        expected[selected] = index
    assert _inside_out_prefix(41, count, _generator(777)) == tuple(expected[:count])


@pytest.mark.parametrize("transposed", [False, True])
def test_normalized_fft_and_pretransposed_fusion(transposed):
    coefficients = np.array([0.1, -0.2, 0.3, 0.2, -0.1, 0.4, 0.5], dtype=np.float32)
    op = parse_configured_adapter(
        {"layer.fourierft_spectrum.default": mx.array(coefficients)},
        {"peft_type": "FOURIERFT", "n_frequency": 7, "random_loc_seed": 777, "scaling": 2.5},
    ).operations[0]
    spectrum = np.zeros(24, dtype=np.float32)
    spectrum[[23, 2, 19, 17, 1, 0, 15]] = coefficients
    expected = np.fft.ifft2(spectrum.reshape(4, 6)).real * 2.5
    np.testing.assert_allclose(
        np.array(reconstruct_delta(op, mx.zeros((4, 6)))), expected, atol=1e-7
    )
    owner = SimpleNamespace(weight=mx.ones((6, 4) if transposed else (4, 6), dtype=mx.float16))
    fuse_and_check(
        {"weight": WeightTarget(owner, "weight", (4, 6))},
        {"weight": (WeightedAdapterOperation(op, 0.5, "a"),)},
        transposed=frozenset({"weight"}) if transposed else frozenset(),
    )
    actual = owner.weight.T if transposed else owner.weight
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        (1 + expected * 0.5).astype(np.float16).astype(np.float32),
        atol=0,
    )


@pytest.mark.parametrize(
    "config",
    [
        {"n_frequency": 6},
        {"random_loc_seed": True},
        {"random_loc_seed": 2**64},
        {"scaling": float("nan")},
    ],
)
def test_invalid_frequency_config_is_rejected(config):
    with pytest.raises(ValueError, match="FourierFT"):
        parse_configured_adapter(
            {"layer.fourierft_spectrum": mx.ones((7,))},
            {"peft_type": "FOURIERFT", "n_frequency": 7, **config},
        )
