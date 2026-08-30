"""Restricted, torch-free reader for zip-format PyTorch tensor checkpoints."""

from __future__ import annotations

import collections
import io
import pickle
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import mlx.core as mx

from .pickle_scan import (
    _STORAGE_TOKENS,
    RestrictedCheckpointError,
    _is_allowed_global,
    scan_pickle_globals,
    suspicious_globals,
)


@dataclass(frozen=True)
class CheckpointReadLimits:
    """Resource limits applied before checkpoint-controlled MLX allocations."""

    max_pickle_bytes: int = 64 * 1024 * 1024
    max_storage_bytes: int = 64 * 1024**3
    max_total_storage_bytes: int = 64 * 1024**3
    max_tensor_bytes: int = 64 * 1024**3
    max_total_tensor_bytes: int = 96 * 1024**3
    max_archive_members: int = 200_000
    max_storages: int = 200_000
    max_tensors: int = 1_000_000
    max_tree_nodes: int = 2_000_000
    max_tree_depth: int = 256
    max_rank: int = 64


_DEFAULT_LIMITS = CheckpointReadLimits()
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_STORAGE_DTYPES = {
    "FloatStorage": (mx.float32, 4),
    "HalfStorage": (mx.float16, 2),
    "BFloat16Storage": (mx.bfloat16, 2),
    "LongStorage": (mx.int64, 8),
    "IntStorage": (mx.int32, 4),
    "ShortStorage": (mx.int16, 2),
    "CharStorage": (mx.int8, 1),
    "ByteStorage": (mx.uint8, 1),
    "BoolStorage": (mx.bool_, 1),
}

if frozenset(_STORAGE_DTYPES) != _STORAGE_TOKENS:
    raise RuntimeError("torch checkpoint scanner and reader storage tokens differ")


def _require_plain_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise pickle.UnpicklingError(f"{label} must be an integer")
    if not _INT64_MIN <= value <= _INT64_MAX:
        raise pickle.UnpicklingError(f"{label} is outside the int64 range")
    return value


@dataclass(frozen=True)
class _StorageRef:
    token: str
    key: str
    numel: int


class _LazyTensor:
    __slots__ = ("storage", "offset", "shape", "strides")

    def __init__(
        self,
        storage: _StorageRef,
        offset: int,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
    ) -> None:
        self.storage = storage
        self.offset = offset
        self.shape = shape
        self.strides = strides


class _RestrictedUnpickler(Protocol):
    storage_meta: dict[str, _StorageRef]

    def load(self) -> object: ...


def _make_restricted_unpickler(
    handle: io.BytesIO,
    limits: CheckpointReadLimits,
) -> _RestrictedUnpickler:
    def rebuild_lazy(
        storage_ref: object,
        offset: object,
        size: object,
        stride: object,
        *_rest: object,
    ) -> _LazyTensor:
        if not isinstance(storage_ref, _StorageRef):
            raise pickle.UnpicklingError("tensor rebuild received an invalid storage")
        if not isinstance(size, (list, tuple)) or not isinstance(stride, (list, tuple)):
            raise pickle.UnpicklingError("tensor size and stride must be sequences")
        if len(size) != len(stride):
            raise pickle.UnpicklingError("tensor size and stride ranks differ")
        if len(size) > limits.max_rank:
            raise pickle.UnpicklingError(f"tensor rank {len(size)} exceeds limit {limits.max_rank}")
        shape = tuple(
            _require_plain_int(value, f"tensor extent {index}") for index, value in enumerate(size)
        )
        strides = tuple(
            _require_plain_int(value, f"tensor stride {index}")
            for index, value in enumerate(stride)
        )
        return _LazyTensor(
            storage_ref,
            _require_plain_int(offset, "tensor storage offset"),
            shape,
            strides,
        )

    class _Restricted(pickle.Unpickler):
        storage_meta: dict[str, _StorageRef]

        def find_class(self, module: str, name: str) -> object:
            if not _is_allowed_global(module, name):
                raise pickle.UnpicklingError(
                    f"{module}.{name} is outside the tensor-rebuild allowlist"
                )
            if (module, name) == ("collections", "OrderedDict"):
                return collections.OrderedDict
            if module == "torch._utils" and name in {
                "_rebuild_tensor_v2",
                "_rebuild_tensor",
            }:
                return rebuild_lazy
            if module == "torch" and name in _STORAGE_DTYPES:
                return name
            if (module, name) == ("torch", "Size"):
                return tuple
            raise pickle.UnpicklingError(f"unsupported allowlisted global {module}.{name}")

        def persistent_load(self, pid: object) -> object:
            if not (isinstance(pid, tuple) and len(pid) >= 5 and pid[0] == "storage"):
                raise pickle.UnpicklingError(f"unsupported persistent id {pid!r}")
            token = pid[1]
            if not isinstance(token, str) or token not in _STORAGE_DTYPES:
                raise pickle.UnpicklingError(f"unsupported checkpoint storage token {token!r}")
            raw_key = pid[2]
            if isinstance(raw_key, bool) or not isinstance(raw_key, (str, int)):
                raise pickle.UnpicklingError("storage key must be a string or integer")
            key = str(raw_key)
            if not key or len(key) > 256 or any(char in key for char in "/\\\0"):
                raise pickle.UnpicklingError(f"invalid storage key {key!r}")
            numel = _require_plain_int(pid[4], f"storage {key!r} element count")
            if numel < 0:
                raise pickle.UnpicklingError(f"storage {key!r} has a negative element count")
            _dtype, itemsize = _STORAGE_DTYPES[token]
            if numel > limits.max_storage_bytes // itemsize:
                raise pickle.UnpicklingError(
                    f"storage {key!r} exceeds the {limits.max_storage_bytes}-byte limit"
                )
            incoming = _StorageRef(token, key, numel)
            existing = self.storage_meta.get(key)
            if existing is not None and existing != incoming:
                raise pickle.UnpicklingError(f"storage key {key!r} has conflicting declarations")
            if existing is None and len(self.storage_meta) >= limits.max_storages:
                raise pickle.UnpicklingError(
                    f"checkpoint exceeds the {limits.max_storages}-storage limit"
                )
            self.storage_meta[key] = incoming
            return existing or incoming

    unpickler = _Restricted(handle)
    unpickler.storage_meta = {}
    return unpickler


def _storage_to_array(token: str, raw: bytes) -> mx.array:
    dtype_info = _STORAGE_DTYPES.get(token)
    if dtype_info is None:
        raise RestrictedCheckpointError(f"unsupported storage type torch.{token}")
    dtype, _itemsize = dtype_info
    # MLX accepts Python buffers, but its public type stub omits that overload.
    flat = mx.array(memoryview(raw))  # type: ignore[arg-type]
    return flat.view(dtype)


@dataclass
class _ResolveBudget:
    nodes: int = 0
    tensors: int = 0
    tensor_bytes: int = 0


def _validated_tensor_size(
    tensor: _LazyTensor,
    flat_size: int,
    limits: CheckpointReadLimits,
) -> int:
    _dtype, itemsize = _STORAGE_DTYPES[tensor.storage.token]
    max_elements = limits.max_tensor_bytes // itemsize
    count = 1
    for axis, extent in enumerate(tensor.shape):
        if extent < 0:
            raise RestrictedCheckpointError(f"tensor extent {axis} is negative: {extent}")
        if extent > max_elements or (count and extent > max_elements // count):
            raise RestrictedCheckpointError(
                f"tensor shape {tensor.shape} exceeds the "
                f"{limits.max_tensor_bytes}-byte per-tensor limit"
            )
        count *= extent
    logical_bytes = count * itemsize
    if count == 0:
        if not 0 <= tensor.offset <= tensor.storage.numel:
            raise RestrictedCheckpointError(
                f"empty tensor offset {tensor.offset} is outside storage {tensor.storage.key!r}"
            )
        return logical_bytes
    minimum = tensor.offset
    maximum = tensor.offset
    for extent, stride in zip(tensor.shape, tensor.strides, strict=True):
        delta = (extent - 1) * stride
        minimum += min(delta, 0)
        maximum += max(delta, 0)
    if not (_INT64_MIN <= minimum <= _INT64_MAX and _INT64_MIN <= maximum <= _INT64_MAX):
        raise RestrictedCheckpointError("tensor index range exceeds int64")
    available = min(tensor.storage.numel, flat_size)
    if minimum < 0 or maximum >= available:
        raise RestrictedCheckpointError(
            f"tensor index range [{minimum}, {maximum}] is outside storage "
            f"{tensor.storage.key!r} with {available} elements"
        )
    return logical_bytes


def _materialize(
    tensor: _LazyTensor,
    flat: mx.array,
    limits: CheckpointReadLimits,
) -> tuple[mx.array, int]:
    logical_bytes = _validated_tensor_size(tensor, flat.size, limits)
    count = logical_bytes // _STORAGE_DTYPES[tensor.storage.token][1]
    if count == 0:
        return mx.zeros(tensor.shape, dtype=flat.dtype), logical_bytes
    contiguous = []
    accumulated = 1
    for extent in reversed(tensor.shape):
        contiguous.append(accumulated)
        accumulated *= extent
    if tensor.strides == tuple(reversed(contiguous)) or count == 1:
        return (
            flat[tensor.offset : tensor.offset + count].reshape(tensor.shape),
            logical_bytes,
        )
    index = mx.array(tensor.offset, dtype=mx.int64)
    for axis, (extent, step) in enumerate(zip(tensor.shape, tensor.strides, strict=True)):
        axis_shape = [1] * len(tensor.shape)
        axis_shape[axis] = extent
        index = index + (mx.arange(extent, dtype=mx.int64) * step).reshape(axis_shape)
    return mx.take(flat, index.reshape(-1)).reshape(tensor.shape), logical_bytes


def _resolve_lazy(
    node: object,
    storages: dict[str, mx.array],
    limits: CheckpointReadLimits,
    budget: _ResolveBudget,
    active: set[int],
    depth: int = 0,
) -> object:
    budget.nodes += 1
    if budget.nodes > limits.max_tree_nodes:
        raise RestrictedCheckpointError(
            f"checkpoint object tree exceeds the {limits.max_tree_nodes}-node limit"
        )
    if depth > limits.max_tree_depth:
        raise RestrictedCheckpointError(
            f"checkpoint object tree exceeds depth {limits.max_tree_depth}"
        )
    if isinstance(node, _LazyTensor):
        budget.tensors += 1
        if budget.tensors > limits.max_tensors:
            raise RestrictedCheckpointError(
                f"checkpoint exceeds the {limits.max_tensors}-tensor limit"
            )
        try:
            flat = storages[node.storage.key]
        except KeyError as exc:
            raise RestrictedCheckpointError(
                f"tensor references missing storage {node.storage.key!r}"
            ) from exc
        value, logical_bytes = _materialize(node, flat, limits)
        budget.tensor_bytes += logical_bytes
        if budget.tensor_bytes > limits.max_total_tensor_bytes:
            raise RestrictedCheckpointError(
                "checkpoint tensors exceed the cumulative "
                f"{limits.max_total_tensor_bytes}-byte limit"
            )
        return value
    if not isinstance(node, (dict, list, tuple)):
        return node
    identity = id(node)
    if identity in active:
        raise RestrictedCheckpointError("cyclic checkpoint object trees are unsupported")
    active.add(identity)
    try:
        if isinstance(node, dict):
            mapping = cast(Mapping[object, object], node)
            resolved_items = (
                (
                    key,
                    _resolve_lazy(
                        value,
                        storages,
                        limits,
                        budget,
                        active,
                        depth + 1,
                    ),
                )
                for key, value in mapping.items()
            )
            if isinstance(node, collections.OrderedDict):
                return collections.OrderedDict(resolved_items)
            return dict(resolved_items)
        resolved = [
            _resolve_lazy(value, storages, limits, budget, active, depth + 1) for value in node
        ]
        return tuple(resolved) if isinstance(node, tuple) else resolved
    finally:
        active.remove(identity)


@dataclass(frozen=True)
class RestrictedCheckpoint:
    """A safely reconstructed object tree plus its static scan receipt."""

    tree: object
    flagged_globals: tuple[str, ...]


def _read_pickle_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    limits: CheckpointReadLimits,
) -> bytes:
    if info.flag_bits & 0x1:
        raise RestrictedCheckpointError("encrypted checkpoint members are unsupported")
    if info.file_size > limits.max_pickle_bytes:
        raise RestrictedCheckpointError(
            f"checkpoint pickle is {info.file_size} bytes; limit is {limits.max_pickle_bytes}"
        )
    data = archive.read(info)
    if len(data) != info.file_size:
        raise RestrictedCheckpointError("checkpoint pickle member was truncated")
    return data


def _read_storages(
    archive: zipfile.ZipFile,
    prefix: str,
    storage_meta: dict[str, _StorageRef],
    member_index: dict[str, list[zipfile.ZipInfo]],
    limits: CheckpointReadLimits,
) -> dict[str, mx.array]:
    storages: dict[str, mx.array] = {}
    total_bytes = 0
    for key, storage in storage_meta.items():
        member_name = f"{prefix}data/{key}"
        matches = member_index.get(member_name, [])
        if len(matches) != 1:
            detail = "missing" if not matches else "duplicated"
            raise RestrictedCheckpointError(
                f"checkpoint storage member {member_name!r} is {detail}"
            )
        info = matches[0]
        if info.flag_bits & 0x1:
            raise RestrictedCheckpointError(
                f"encrypted storage member {member_name!r} is unsupported"
            )
        if info.compress_type != zipfile.ZIP_STORED:
            raise RestrictedCheckpointError(f"storage member {member_name!r} must be uncompressed")
        _dtype, itemsize = _STORAGE_DTYPES[storage.token]
        expected_bytes = storage.numel * itemsize
        if info.file_size != expected_bytes or info.compress_size != expected_bytes:
            raise RestrictedCheckpointError(
                f"storage member {member_name!r} has {info.file_size} bytes; "
                f"metadata declares {expected_bytes}"
            )
        total_bytes += expected_bytes
        if total_bytes > limits.max_total_storage_bytes:
            raise RestrictedCheckpointError(
                "checkpoint storages exceed the cumulative "
                f"{limits.max_total_storage_bytes}-byte limit"
            )
        raw = archive.read(info)
        if len(raw) != expected_bytes:
            raise RestrictedCheckpointError(f"storage member {member_name!r} was truncated")
        try:
            storages[key] = _storage_to_array(storage.token, raw)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise RestrictedCheckpointError(
                f"cannot materialize storage {member_name!r} as {storage.token}: {exc}"
            ) from exc
    return storages


def load_restricted_checkpoint(
    source: Path | str,
    *,
    allow_suspicious: bool = False,
    limits: CheckpointReadLimits | None = None,
) -> RestrictedCheckpoint:
    """Scan, then restricted-load, one zip-format torch checkpoint."""
    path = Path(source)
    policy = limits or _DEFAULT_LIMITS
    try:
        if not zipfile.is_zipfile(path):
            raise RestrictedCheckpointError(
                f"{path} is not a zip-format torch checkpoint; legacy stream "
                "checkpoints are not supported"
            )
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > policy.max_archive_members:
                raise RestrictedCheckpointError(
                    f"checkpoint has {len(infos)} members; limit is {policy.max_archive_members}"
                )
            member_index: dict[str, list[zipfile.ZipInfo]] = {}
            for info in infos:
                member_index.setdefault(info.filename, []).append(info)
            pickle_infos = [
                info
                for info in infos
                if info.filename == "data.pkl" or info.filename.endswith("/data.pkl")
            ]
            if len(pickle_infos) != 1:
                detail = "no" if not pickle_infos else "multiple"
                raise RestrictedCheckpointError(f"{path} carries {detail} data.pkl member(s)")
            pickle_info = pickle_infos[0]
            pickle_data = _read_pickle_member(archive, pickle_info, policy)
            flagged = suspicious_globals(scan_pickle_globals(pickle_data))
            if flagged and not allow_suspicious:
                raise RestrictedCheckpointError(
                    f"{path} references globals outside the tensor-rebuild "
                    f"allowlist: {flagged}; refusing to load"
                )
            unpickler = _make_restricted_unpickler(io.BytesIO(pickle_data), policy)
            try:
                tree = unpickler.load()
            except (
                AttributeError,
                EOFError,
                IndexError,
                OverflowError,
                RecursionError,
                TypeError,
                ValueError,
                pickle.UnpicklingError,
            ) as exc:
                raise RestrictedCheckpointError(
                    f"restricted checkpoint load refused: {exc}; the file carries "
                    "constructs outside a plain tensor state dict"
                ) from exc
            prefix = pickle_info.filename[: -len("data.pkl")]
            storages = _read_storages(
                archive,
                prefix,
                unpickler.storage_meta,
                member_index,
                policy,
            )
        try:
            tree = _resolve_lazy(tree, storages, policy, _ResolveBudget(), set())
        except RecursionError as exc:
            raise RestrictedCheckpointError(
                "checkpoint object tree exceeds the safe recursion depth"
            ) from exc
    except RestrictedCheckpointError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise RestrictedCheckpointError(f"cannot read checkpoint {path}: {exc}") from exc
    return RestrictedCheckpoint(tree=tree, flagged_globals=tuple(flagged))


__all__ = [
    "CheckpointReadLimits",
    "RestrictedCheckpoint",
    "RestrictedCheckpointError",
    "load_restricted_checkpoint",
    "scan_pickle_globals",
    "suspicious_globals",
]
