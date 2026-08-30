"""The gated converter: synthesized torch-format checkpoints, no torch installed.

A minimal fake ``torch`` module tree is installed only while *writing* the
fixture checkpoints (pickle resolves globals at dump time); the converter
under test never sees it and stays torch-free.
"""

from __future__ import annotations

import io
import pickle
import sys
import types
import zipfile
from collections import OrderedDict

import mlx.core as mx
import numpy as np
import pytest

from kinomlx.models.gmnet.catalog import GMNetVariant
from kinomlx.models.gmnet.convert import (
    CheckpointConversionError,
    _state_dict_from_tree,
    convert_checkpoint,
    scan_pickle_globals,
    suspicious_globals,
)
from kinomlx.models.gmnet.net import load_gmnet_weights

from .test_net import _upstream_layout_state_dict


class _FakeStorage:
    def __init__(self, key: str, numel: int) -> None:
        self.key = key
        self.numel = numel


class _FakeTensor:
    def __init__(self, storage: _FakeStorage, shape: tuple[int, ...]) -> None:
        self.storage = storage
        self.shape = shape

    def __reduce_ex__(self, protocol: int):
        stride = []
        accumulated = 1
        for extent in reversed(self.shape):
            stride.append(accumulated)
            accumulated *= extent
        rebuild = sys.modules["torch._utils"]._rebuild_tensor_v2
        return (
            rebuild,
            (self.storage, 0, self.shape, tuple(reversed(stride)), False, None),
        )


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    utils = types.ModuleType("torch._utils")

    def _rebuild_tensor_v2(*_args):  # never called; exists to be pickled by name
        raise AssertionError("fixture rebuild function must not execute")

    _rebuild_tensor_v2.__module__ = "torch._utils"
    _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
    utils._rebuild_tensor_v2 = _rebuild_tensor_v2

    torch_module = types.ModuleType("torch")

    class FloatStorage:
        pass

    FloatStorage.__module__ = "torch"
    FloatStorage.__qualname__ = "FloatStorage"
    torch_module.FloatStorage = FloatStorage
    torch_module._utils = utils
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch._utils", utils)


class _StoragePickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, _FakeStorage):
            storage_type = sys.modules["torch"].FloatStorage
            return ("storage", storage_type, obj.key, "cpu", obj.numel)
        return None


def _write_zip_checkpoint(path, arrays: dict[str, np.ndarray]) -> None:
    state = OrderedDict()
    storages: dict[str, np.ndarray] = {}
    for index, (key, array) in enumerate(arrays.items()):
        storage_key = str(index)
        storages[storage_key] = np.ascontiguousarray(array)
        state[key] = _FakeTensor(_FakeStorage(storage_key, int(array.size)), tuple(array.shape))
    buffer = io.BytesIO()
    _StoragePickler(buffer, protocol=2).dump(state)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", buffer.getvalue())
        archive.writestr("archive/version", "3\n")
        for storage_key, array in storages.items():
            archive.writestr(f"archive/data/{storage_key}", array.tobytes())


def _generator_arrays(prefix: str = "") -> dict[str, np.ndarray]:
    return {
        f"{prefix}{key}": np.array(value, copy=True)
        for key, value in _upstream_layout_state_dict().items()
    }


def test_convert_round_trips_a_valid_generator_checkpoint(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    source = tmp_path / "G_fixture.pth"
    arrays = _generator_arrays()
    _write_zip_checkpoint(source, arrays)
    output = tmp_path / "converted.safetensors"

    receipt = convert_checkpoint(source, output, declared_variant=GMNetVariant.SYNTHETIC)

    assert receipt.output == output
    assert receipt.variant is GMNetVariant.SYNTHETIC
    assert receipt.tensor_count == 126
    assert not receipt.flagged_globals
    model, metadata = load_gmnet_weights(output)
    assert metadata["variant"] == "synthetic"
    assert metadata["source_sha256"] == receipt.source_sha256
    assert metadata["license"] == "MIT"
    probe = np.array(dict(mx.load(str(output)))["conv_last.weight"])
    np.testing.assert_array_equal(probe, arrays["conv_last.weight"])
    del model


def test_convert_strips_the_dataparallel_prefix(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    source = tmp_path / "G_prefixed.pth"
    _write_zip_checkpoint(source, _generator_arrays(prefix="module."))
    output = tmp_path / "converted.safetensors"
    receipt = convert_checkpoint(source, output, declared_variant=GMNetVariant.REALWORLD)
    assert receipt.tensor_count == 126


def test_default_conversion_output_lives_under_the_configured_cache(
    tmp_path,
    monkeypatch,
):
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(
        "kinomlx.models.gmnet.catalog._editable_checkout_weights_dir",
        lambda: None,
    )
    source = tmp_path / "G_fixture.pth"
    _write_zip_checkpoint(source, _generator_arrays())

    receipt = convert_checkpoint(
        source,
        declared_variant=GMNetVariant.SYNTHETIC,
        cache_dir=tmp_path / "cache",
    )

    assert receipt.output == (
        tmp_path / "cache" / "weights" / "gmnet" / "gmnet_synthetic.safetensors"
    )
    assert receipt.output.is_file()


def test_conversion_refuses_non_floating_generator_weights(tmp_path):
    with pytest.raises(CheckpointConversionError, match="must be floating-point"):
        _state_dict_from_tree(
            {"conv_last.weight": mx.array([1], dtype=mx.int32)},
            tmp_path / "integer.pth",
        )


def test_convert_refuses_existing_output_without_force(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    source = tmp_path / "G_fixture.pth"
    _write_zip_checkpoint(source, _generator_arrays())
    output = tmp_path / "converted.safetensors"
    output.write_bytes(b"keep me")

    with pytest.raises(CheckpointConversionError, match="exists"):
        convert_checkpoint(source, output, declared_variant=GMNetVariant.SYNTHETIC)
    assert output.read_bytes() == b"keep me"


def test_existing_output_is_refused_before_checkpoint_load(tmp_path, monkeypatch):
    source = tmp_path / "G_fixture.pth"
    source.write_bytes(b"source only needs to be hashable")
    output = tmp_path / "converted.safetensors"
    output.write_bytes(b"keep me")
    monkeypatch.setattr(
        "kinomlx.models.gmnet.convert.load_restricted_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not load")),
    )

    with pytest.raises(CheckpointConversionError, match="exists"):
        convert_checkpoint(source, output, declared_variant=GMNetVariant.SYNTHETIC)

    assert output.read_bytes() == b"keep me"


def test_convert_requires_a_safetensors_output_suffix(tmp_path):
    source = tmp_path / "G_fixture.pth"
    source.write_bytes(b"source only needs to be hashable")
    with pytest.raises(CheckpointConversionError, match="must end in .safetensors"):
        convert_checkpoint(
            source,
            tmp_path / "converted.bin",
            declared_variant=GMNetVariant.SYNTHETIC,
        )


def test_convert_force_replaces_and_preserves_permissions(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    source = tmp_path / "G_fixture.pth"
    arrays = _generator_arrays()
    _write_zip_checkpoint(source, arrays)
    output = tmp_path / "converted.safetensors"
    output.write_bytes(b"old")
    output.chmod(0o640)

    receipt = convert_checkpoint(
        source,
        output,
        declared_variant=GMNetVariant.SYNTHETIC,
        force=True,
    )

    assert receipt.output == output
    assert output.stat().st_mode & 0o777 == 0o640


def test_convert_removes_a_new_target_when_verification_fails(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    source = tmp_path / "G_fixture.pth"
    _write_zip_checkpoint(source, _generator_arrays())
    output = tmp_path / "converted.safetensors"
    monkeypatch.setattr(
        "kinomlx.models.gmnet.convert.load_gmnet_weights",
        lambda _path: (_ for _ in ()).throw(ValueError("verification failed")),
    )

    with pytest.raises(ValueError, match="verification failed"):
        convert_checkpoint(source, output, declared_variant=GMNetVariant.SYNTHETIC)
    assert not output.exists()


def test_convert_preserves_forced_target_when_verification_fails(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    source = tmp_path / "G_fixture.pth"
    _write_zip_checkpoint(source, _generator_arrays())
    output = tmp_path / "converted.safetensors"
    output.write_bytes(b"old artifact")
    output.chmod(0o640)
    monkeypatch.setattr(
        "kinomlx.models.gmnet.convert.load_gmnet_weights",
        lambda _path: (_ for _ in ()).throw(ValueError("verification failed")),
    )

    with pytest.raises(ValueError, match="verification failed"):
        convert_checkpoint(
            source,
            output,
            declared_variant=GMNetVariant.SYNTHETIC,
            force=True,
        )

    assert output.read_bytes() == b"old artifact"
    assert output.stat().st_mode & 0o777 == 0o640


def test_convert_refuses_wrong_key_sets(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    source = tmp_path / "partial.pth"
    _write_zip_checkpoint(source, {"conv_last.weight": np.zeros((1, 64, 3, 3), dtype=np.float32)})
    with pytest.raises(CheckpointConversionError, match="not a GMNet generator state dict"):
        convert_checkpoint(source, tmp_path / "out.safetensors")


def test_convert_refuses_unknown_variant_without_declaration(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    source = tmp_path / "G_fixture.pth"
    _write_zip_checkpoint(source, _generator_arrays())
    with pytest.raises(CheckpointConversionError, match="--declare-variant"):
        convert_checkpoint(source)


def test_convert_refuses_non_zip_files(tmp_path):
    source = tmp_path / "legacy.pth"
    source.write_bytes(b"definitely not a zip archive")
    with pytest.raises(CheckpointConversionError, match="zip-format"):
        convert_checkpoint(source, tmp_path / "out.safetensors")


class TestSourceResolution:
    """Bare inputs resolve against the cwd-relative weights-src collection."""

    def test_literal_path_wins(self, tmp_path, monkeypatch):
        from kinomlx.models.gmnet.convert import resolve_source

        monkeypatch.chdir(tmp_path)
        (tmp_path / "weights-src" / "gmnet").mkdir(parents=True)
        (tmp_path / "weights-src" / "gmnet" / "x.pth").write_bytes(b"collected")
        (tmp_path / "x.pth").write_bytes(b"local")
        assert resolve_source("x.pth").read_bytes() == b"local"

    def test_bare_name_resolves_uniquely(self, tmp_path, monkeypatch):
        from kinomlx.models.gmnet.convert import resolve_source

        monkeypatch.chdir(tmp_path)
        target = tmp_path / "weights-src" / "gmnet" / "x.pth"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"collected")
        assert resolve_source("x.pth").read_bytes() == b"collected"

    def test_owner_relative_path_resolves(self, tmp_path, monkeypatch):
        from kinomlx.models.gmnet.convert import resolve_source

        monkeypatch.chdir(tmp_path)
        target = tmp_path / "weights-src" / "gmnet" / "x.pth"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"collected")
        assert resolve_source("gmnet/x.pth") == target.relative_to(tmp_path)

    def test_ambiguous_name_is_refused(self, tmp_path, monkeypatch):
        from kinomlx.models.gmnet.convert import resolve_source

        monkeypatch.chdir(tmp_path)
        for owner in ("a", "b"):
            target = tmp_path / "weights-src" / owner / "x.pth"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"collected")
        with pytest.raises(CheckpointConversionError, match="ambiguous"):
            resolve_source("x.pth")

    def test_missing_everywhere_names_the_collection(self, tmp_path, monkeypatch):
        from kinomlx.models.gmnet.convert import resolve_source

        monkeypatch.chdir(tmp_path)
        with pytest.raises(CheckpointConversionError, match="weights-src"):
            resolve_source("nope.pth")


class _Evil:
    def __reduce__(self):
        import os

        return (os.system, ("true",))


def test_static_scan_flags_code_execution_globals():
    payload = pickle.dumps({"weight": _Evil()}, protocol=2)
    flagged = suspicious_globals(scan_pickle_globals(payload))
    assert flagged, "os.system pickle must be flagged"


def test_convert_refuses_a_malicious_pickle_even_when_forced(tmp_path):
    payload = pickle.dumps({"weight": _Evil()}, protocol=2)
    source = tmp_path / "evil.pth"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("archive/data.pkl", payload)

    with pytest.raises(CheckpointConversionError, match="allowlist"):
        convert_checkpoint(source, tmp_path / "out.safetensors")
    # The static scan can be overridden for trusted files, but the restricted
    # unpickler must still refuse to resolve the global.
    with pytest.raises(CheckpointConversionError, match="restricted checkpoint load"):
        convert_checkpoint(source, tmp_path / "out.safetensors", allow_suspicious=True)
