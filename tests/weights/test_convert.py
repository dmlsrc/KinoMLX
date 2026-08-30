"""Generic checkpoint conversion stays value-preserving and transactional."""

from __future__ import annotations

import io
import zipfile
from collections import OrderedDict

import numpy as np
import pytest

from kinomlx.io.reservation import reservation_path
from kinomlx.io.safetensors import load_weights_with_metadata
from kinomlx.weights.convert import WeightConversionError, convert_checkpoint
from tests.models.gmnet.test_convert import (
    _FakeStorage,
    _FakeTensor,
    _install_fake_torch,
    _StoragePickler,
    _write_zip_checkpoint,
)


def _write_nested_checkpoint(path, tree: object) -> None:
    storages: dict[str, np.ndarray] = {}

    def replace(node: object) -> object:
        if isinstance(node, np.ndarray):
            key = str(len(storages))
            array = np.ascontiguousarray(node)
            storages[key] = array
            return _FakeTensor(_FakeStorage(key, int(array.size)), tuple(array.shape))
        if isinstance(node, dict):
            return type(node)((key, replace(value)) for key, value in node.items())
        if isinstance(node, list):
            return [replace(value) for value in node]
        if isinstance(node, tuple):
            return tuple(replace(value) for value in node)
        return node

    buffer = io.BytesIO()
    _StoragePickler(buffer, protocol=2).dump(replace(tree))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", buffer.getvalue())
        for key, array in storages.items():
            archive.writestr(f"archive/data/{key}", array.tobytes())


def test_generic_convert_preserves_layout_and_strips_prefix(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    source = tmp_path / "plain.pth"
    expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    _write_zip_checkpoint(
        source,
        {
            "module.encoder.weight": expected,
            "module.decoder.bias": np.array([1.0, 2.0], dtype=np.float32),
        },
    )
    output = tmp_path / "plain.safetensors"

    receipt = convert_checkpoint(source, output)

    weights, metadata = load_weights_with_metadata(output)
    assert set(weights) == {"encoder.weight", "decoder.bias"}
    np.testing.assert_array_equal(np.array(weights["encoder.weight"]), expected)
    assert receipt.tensor_count == 2
    assert receipt.stripped_keys == 2
    assert metadata["format"] == "generic-torch-state-dict"


def test_generic_convert_filters_without_changing_retained_tensor(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    source = tmp_path / "plain.pth"
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    _write_zip_checkpoint(
        source,
        {
            "generator.keep": expected,
            "critic.drop": np.ones((2,), dtype=np.float32),
        },
    )

    receipt = convert_checkpoint(
        source,
        tmp_path / "plain.safetensors",
        only_prefix="generator.",
        strip_prefix="generator.",
    )

    weights, _metadata = load_weights_with_metadata(receipt.output)
    assert set(weights) == {"keep"}
    np.testing.assert_array_equal(np.array(weights["keep"]), expected)
    assert receipt.filtered_keys == 1


def test_generic_convert_refuses_ambiguous_parameter_branches(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    source = tmp_path / "nested.pth"
    _write_nested_checkpoint(
        source,
        {
            "params": OrderedDict(
                weight=np.array([1.0], dtype=np.float32),
            ),
            "params_ema": OrderedDict(
                weight=np.array([2.0], dtype=np.float32),
            ),
        },
    )

    with pytest.raises(WeightConversionError, match="both 'params' and 'params_ema'"):
        convert_checkpoint(source, tmp_path / "ambiguous.safetensors")

    receipt = convert_checkpoint(
        source,
        tmp_path / "selected.safetensors",
        param_key="params_ema",
    )
    weights, _metadata = load_weights_with_metadata(receipt.output)
    np.testing.assert_array_equal(np.array(weights["weight"]), np.array([2.0]))


def test_generic_convert_refuses_multiple_recognized_state_mappings(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    source = tmp_path / "nested.pth"
    _write_nested_checkpoint(
        source,
        {
            "state_dict": {"weight": np.array([1.0], dtype=np.float32)},
            "model": {"weight": np.array([2.0], dtype=np.float32)},
        },
    )

    with pytest.raises(WeightConversionError, match="multiple parameter mappings"):
        convert_checkpoint(source, tmp_path / "ambiguous.safetensors")


def test_generic_convert_refuses_root_tensors_mixed_with_nested_state(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    source = tmp_path / "nested.pth"
    _write_nested_checkpoint(
        source,
        {
            "diagnostic": np.array([0.0], dtype=np.float32),
            "state_dict": {"weight": np.array([1.0], dtype=np.float32)},
        },
    )

    with pytest.raises(WeightConversionError, match="root tensors and nested"):
        convert_checkpoint(source, tmp_path / "ambiguous.safetensors")


def test_generic_convert_records_dropped_non_tensor_entries(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    source = tmp_path / "mixed.pth"
    _write_nested_checkpoint(
        source,
        OrderedDict(
            weight=np.array([1.0], dtype=np.float32),
            epoch=7,
        ),
    )

    receipt = convert_checkpoint(source, tmp_path / "mixed.safetensors")

    assert receipt.dropped_entries == ("epoch",)


def test_generic_convert_refuses_existing_output(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    source = tmp_path / "plain.pth"
    _write_zip_checkpoint(source, {"weight": np.ones((2,), dtype=np.float32)})
    output = tmp_path / "plain.safetensors"
    output.write_bytes(b"keep")

    with pytest.raises(WeightConversionError, match="exists"):
        convert_checkpoint(source, output)

    assert output.read_bytes() == b"keep"


def test_generic_existing_output_is_refused_before_checkpoint_load(tmp_path, monkeypatch):
    source = tmp_path / "plain.pth"
    source.write_bytes(b"source only needs to be hashable")
    output = tmp_path / "plain.safetensors"
    output.write_bytes(b"keep")
    monkeypatch.setattr(
        "kinomlx.weights.convert.load_restricted_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not load")),
    )

    with pytest.raises(WeightConversionError, match="exists"):
        convert_checkpoint(source, output)

    assert output.read_bytes() == b"keep"


def test_generic_conversion_reserves_a_hidden_peer_not_the_final_name(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "plain.pth"
    source.write_bytes(b"source only needs to be hashable")
    output = tmp_path / "plain.safetensors"

    def inspect_reservation(*_args, **_kwargs):
        assert not output.exists()
        assert reservation_path(output).is_file()
        raise RuntimeError("stop after reservation check")

    monkeypatch.setattr(
        "kinomlx.weights.convert.load_restricted_checkpoint",
        inspect_reservation,
    )

    with pytest.raises(RuntimeError, match="reservation check"):
        convert_checkpoint(source, output)

    assert not output.exists()
    assert not reservation_path(output).exists()


def test_generic_conversion_refuses_a_stale_reservation_marker(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "plain.pth"
    source.write_bytes(b"source only needs to be hashable")
    output = tmp_path / "plain.safetensors"
    marker = reservation_path(output)
    marker.write_text("interrupted run")
    monkeypatch.setattr(
        "kinomlx.weights.convert.load_restricted_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not load")),
    )

    with pytest.raises(WeightConversionError, match="remove that marker and retry"):
        convert_checkpoint(source, output)

    assert marker.read_text() == "interrupted run"


def test_forced_conversion_preserves_old_output_when_verification_fails(
    tmp_path,
    monkeypatch,
):
    _install_fake_torch(monkeypatch)
    source = tmp_path / "plain.pth"
    _write_zip_checkpoint(source, {"weight": np.ones((2,), dtype=np.float32)})
    output = tmp_path / "plain.safetensors"
    output.write_bytes(b"old artifact")
    output.chmod(0o640)
    monkeypatch.setattr(
        "kinomlx.weights.convert.load_weights",
        lambda _path: (_ for _ in ()).throw(ValueError("verification failed")),
    )

    with pytest.raises(ValueError, match="verification failed"):
        convert_checkpoint(source, output, force=True)

    assert output.read_bytes() == b"old artifact"
    assert output.stat().st_mode & 0o777 == 0o640


def test_default_output_is_refused_inside_source_collection(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "weights-src" / "owner" / "plain.pth"
    source.parent.mkdir(parents=True)
    _write_zip_checkpoint(source, {"weight": np.ones((2,), dtype=np.float32)})

    with pytest.raises(WeightConversionError, match="state an output"):
        convert_checkpoint(source)
