"""Hostile checkpoint metadata is refused before it controls MLX work."""

from __future__ import annotations

import io
import pickle
import sys
import zipfile
from collections import OrderedDict, defaultdict

import pytest

import kinomlx.weights.torch_checkpoint as checkpoint_module
from kinomlx.weights.torch_checkpoint import (
    CheckpointReadLimits,
    RestrictedCheckpointError,
    load_restricted_checkpoint,
    scan_pickle_globals,
    suspicious_globals,
)
from tests.models.gmnet.test_convert import (
    _FakeStorage,
    _install_fake_torch,
    _StoragePickler,
)


class _TensorSpec:
    def __init__(
        self,
        storage: _FakeStorage,
        *,
        offset: int = 0,
        shape: tuple[int, ...] = (1,),
        strides: tuple[int, ...] = (1,),
    ) -> None:
        self.storage = storage
        self.offset = offset
        self.shape = shape
        self.strides = strides

    def __reduce_ex__(self, protocol: int):
        rebuild = sys.modules["torch._utils"]._rebuild_tensor_v2
        return (
            rebuild,
            (
                self.storage,
                self.offset,
                self.shape,
                self.strides,
                False,
                None,
            ),
        )


def _write_checkpoint(
    path,
    tree: object,
    storages: dict[str, bytes],
    *,
    compression: int = zipfile.ZIP_STORED,
) -> None:
    buffer = io.BytesIO()
    _StoragePickler(buffer, protocol=4).dump(tree)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", buffer.getvalue())
        for key, raw in storages.items():
            archive.writestr(
                zipfile.ZipInfo(f"archive/data/{key}"),
                raw,
                compress_type=compression,
            )


def test_protocol_four_stack_global_is_resolved_from_the_pickle_memo():
    payload = pickle.dumps(defaultdict, protocol=4)

    references = scan_pickle_globals(payload)

    assert any(reference.endswith(" defaultdict") for reference in references)
    assert suspicious_globals(references)


def test_static_scan_uses_the_restricted_loaders_exact_allowlist():
    references = {
        "collections OrderedDict",
        "collections defaultdict",
        "torch._utils _rebuild_tensor_v2",
    }

    assert suspicious_globals(references) == ["collections defaultdict"]


def test_pickle_member_limit_is_checked_before_reading(tmp_path):
    source = tmp_path / "large-pickle.pth"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("archive/data.pkl", b"x" * 32)

    with pytest.raises(RestrictedCheckpointError, match="pickle is 32 bytes"):
        load_restricted_checkpoint(
            source,
            limits=CheckpointReadLimits(max_pickle_bytes=16),
        )


def test_multiple_pickle_members_are_ambiguous(tmp_path):
    source = tmp_path / "ambiguous.pth"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("one/data.pkl", pickle.dumps({}))
        archive.writestr("two/data.pkl", pickle.dumps({}))

    with pytest.raises(RestrictedCheckpointError, match="multiple data.pkl"):
        load_restricted_checkpoint(source)


def test_compressed_storage_is_refused_before_decompression(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    source = tmp_path / "compressed.pth"
    tensor = _TensorSpec(_FakeStorage("0", 1))
    _write_checkpoint(
        source,
        OrderedDict(weight=tensor),
        {"0": b"\0\0\0\0"},
        compression=zipfile.ZIP_DEFLATED,
    )

    with pytest.raises(RestrictedCheckpointError, match="must be uncompressed"):
        load_restricted_checkpoint(source)


def test_storage_bytes_must_match_the_declared_element_count(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    source = tmp_path / "truncated.pth"
    tensor = _TensorSpec(_FakeStorage("0", 2), shape=(2,), strides=(1,))
    _write_checkpoint(source, OrderedDict(weight=tensor), {"0": b"\0\0\0\0"})

    with pytest.raises(RestrictedCheckpointError, match="metadata declares 8"):
        load_restricted_checkpoint(source)


def test_conflicting_storage_declarations_are_refused(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    source = tmp_path / "conflict.pth"
    tree = OrderedDict(
        first=_TensorSpec(_FakeStorage("0", 1)),
        second=_TensorSpec(_FakeStorage("0", 2), shape=(2,), strides=(1,)),
    )
    _write_checkpoint(source, tree, {"0": b"\0" * 8})

    with pytest.raises(RestrictedCheckpointError, match="conflicting declarations"):
        load_restricted_checkpoint(source)


def test_tensor_shape_cannot_request_an_allocation_bomb(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    source = tmp_path / "shape-bomb.pth"
    tensor = _TensorSpec(
        _FakeStorage("0", 1),
        shape=(2**32, 2**32),
        strides=(2**32, 1),
    )
    _write_checkpoint(source, OrderedDict(weight=tensor), {"0": b"\0\0\0\0"})

    with pytest.raises(RestrictedCheckpointError, match="per-tensor limit"):
        load_restricted_checkpoint(source)


@pytest.mark.parametrize(
    ("offset", "shape", "strides"),
    [
        (2, (1,), (1,)),
        (0, (2,), (1,)),
        (0, (2,), (-1,)),
    ],
)
def test_tensor_index_range_must_stay_inside_storage(
    tmp_path,
    monkeypatch,
    offset,
    shape,
    strides,
):
    _install_fake_torch(monkeypatch)
    source = tmp_path / "bad-view.pth"
    tensor = _TensorSpec(
        _FakeStorage("0", 1),
        offset=offset,
        shape=shape,
        strides=strides,
    )
    _write_checkpoint(source, OrderedDict(weight=tensor), {"0": b"\0\0\0\0"})

    with pytest.raises(RestrictedCheckpointError, match="outside storage"):
        load_restricted_checkpoint(source)


def test_object_tree_depth_is_bounded(tmp_path):
    source = tmp_path / "deep.pth"
    tree: object = "leaf"
    for _ in range(8):
        tree = [tree]
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps(tree, protocol=4))

    with pytest.raises(RestrictedCheckpointError, match="exceeds depth 3"):
        load_restricted_checkpoint(
            source,
            limits=CheckpointReadLimits(max_tree_depth=3),
        )


def test_recursive_unpickler_failure_is_reported_as_a_typed_refusal(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "recursive.pth"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps({}))

    class RecursiveUnpickler:
        storage_meta = {}

        def load(self):
            raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(
        checkpoint_module,
        "_make_restricted_unpickler",
        lambda *_args: RecursiveUnpickler(),
    )

    with pytest.raises(RestrictedCheckpointError, match="maximum recursion depth"):
        load_restricted_checkpoint(source)
