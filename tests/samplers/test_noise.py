"""Central normal-noise stream parity and ownership tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import kinomlx
from kinomlx.samplers.noise import (
    NoiseStreamState,
    create_normal_noise_stream,
    noise_compatibility_profile,
)


def _float32_sha256(value: mx.array) -> str:
    mx.eval(value)
    payload = np.asarray(value.astype(mx.float32)).tobytes()
    return hashlib.sha256(payload).hexdigest()


def test_default_noise_backend_preserves_mlx_sequence() -> None:
    default = create_normal_noise_stream(42)
    explicit = create_normal_noise_stream(42, backend="mlx")

    expected = explicit.normal((17,), mx.float32)
    actual = default.normal((17,), mx.float32)

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert default.state.backend == "mlx"
    assert noise_compatibility_profile("mlx") == "mlx-native"


def test_torch_mps_odd_tails_match_pinned_torch_oracles_exactly() -> None:
    bf16 = create_normal_noise_stream(42, backend="torch-mps")
    fp32 = create_normal_noise_stream(42, backend="torch-mps")

    assert _float32_sha256(bf16.normal((17,), mx.bfloat16)) == (
        "c30e7ec47b198c12a34ba4e7b6a142be31dd6690f2e36edfd9a56cdde89426b8"
    )
    assert _float32_sha256(fp32.normal((17,), mx.float32)) == (
        "43f6127832dbda2c78ee261c51786d1ca810fb88e0e7dde517afae5cb8a504d3"
    )
    assert bf16.state.philox_blocks == 5
    assert fp32.state.philox_blocks == 5


def test_torch_mps_initial_joint_sequence_matches_pinned_torch_oracles_exactly() -> None:
    stream = create_normal_noise_stream(42, backend="torch-mps")

    stage_1_video = stream.normal((1, 1_344, 128), mx.bfloat16)
    stage_1_audio = stream.normal((1, 126, 128), mx.bfloat16)
    stage_2_video = stream.normal((1, 5_376, 128), mx.bfloat16)

    assert _float32_sha256(stage_1_video) == (
        "01958a7227c4e2bda99d65b5fe636420d9b56ad03fdae8202bfc60cd807d1482"
    )
    assert _float32_sha256(stage_1_audio) == (
        "0b14a8fbe506771c5ec2575ab28393cdc855190d3e98a1d7e25016e727cce533"
    )
    assert _float32_sha256(stage_2_video) == (
        "a1069159827debefdb8510250df618fee8f15bc0e7af6a729252d66c36ae3cf9"
    )
    assert stream.state.draws == 3
    assert stream.state.philox_blocks == 219_072


def test_torch_mps_ancestral_stream_matches_pinned_torch_oracles_exactly() -> None:
    stream = create_normal_noise_stream(10_042, backend="torch-mps")

    video = stream.normal((1, 1_344, 128), mx.bfloat16)
    audio = stream.normal((1, 126, 128), mx.bfloat16)

    assert _float32_sha256(video) == (
        "86375d4cb09208fc7e04eea9e540d8eea5ead798806fcbf8666edefa66c2d8d1"
    )
    assert _float32_sha256(audio) == (
        "6c53d1ce034b5890b54298a01507ccf7ac253b922576e51a8ae6a5b4415f91cc"
    )
    assert stream.state.philox_blocks == 47_040


def test_saved_position_can_resume_with_either_backend() -> None:
    source = create_normal_noise_stream(42, backend="mlx")
    source.normal((17,), mx.float32)
    position = source.state

    resumed = create_normal_noise_stream(
        42,
        backend="torch-mps",
        position=position,
    )
    uninterrupted = create_normal_noise_stream(42, backend="torch-mps")
    uninterrupted.advance((17,))

    expected = uninterrupted.normal((9,), mx.float32)
    actual = resumed.normal((9,), mx.float32)

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert resumed.state.draws == 2
    assert resumed.state.philox_blocks == 8


def test_noise_state_round_trips_through_artifact_metadata() -> None:
    state = NoiseStreamState(
        backend="torch-mps",
        compatibility_profile="pytorch-2.13.0-mps",
        seed=42,
        draws=2,
        elements=18,
        philox_blocks=6,
    )
    metadata = dict(state.to_artifact_metadata())

    assert NoiseStreamState.from_artifact_metadata(metadata) == state
    assert NoiseStreamState.from_artifact_metadata({}) is None


def test_partial_artifact_noise_state_is_rejected() -> None:
    with pytest.raises(ValueError, match="noise state is incomplete"):
        NoiseStreamState.from_artifact_metadata({"initial_noise_backend": "mlx"})


def test_all_production_normal_noise_is_centralized() -> None:
    package_root = Path(kinomlx.__file__).parent
    owners = []
    for path in package_root.rglob("*.py"):
        if "mx.random.normal" in path.read_text(encoding="utf-8"):
            owners.append(path.relative_to(package_root).as_posix())

    assert owners == ["samplers/noise.py"]


def test_product_source_has_no_uniform_randomness() -> None:
    package_root = Path(kinomlx.__file__).parent
    owners = []
    for path in package_root.rglob("*.py"):
        if "mx.random.uniform" in path.read_text(encoding="utf-8"):
            owners.append(path.relative_to(package_root).as_posix())

    assert owners == []
