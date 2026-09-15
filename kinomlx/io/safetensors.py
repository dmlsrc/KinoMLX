"""Safetensors I/O helpers - typed, opinionated wrappers.

Wraps ``mlx.core.load`` / ``mlx.core.save_safetensors`` for the
project's two safetensors use cases:

- **Weight loading** - read model checkpoints and adapter files into a
  ``dict[str, mx.array]`` for assignment into MLX modules.
- **Artifact writing** - save converted weights and generation sidecars with metadata.

For ``.npy`` / ``.npz`` / ``.gguf`` use ``mlx.core.load`` directly;
these helpers assume safetensors throughout.
"""

import json
import math
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mlx.core as mx

from kinomlx._json import JsonObject

from .atomic import atomic_output_path

_DTYPE_BYTES = {
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BF16": 2,
    "F16": 2,
    "I16": 2,
    "U16": 2,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}
_MAX_HEADER_BYTES = 100_000_000


@dataclass(frozen=True)
class SafetensorsTensor:
    """One validated tensor descriptor from a safetensors header."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)

    @property
    def byte_count(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]


@dataclass(frozen=True)
class SafetensorsInventory:
    """Header-only validated inventory with no materialized tensor arrays."""

    path: Path
    tensors: tuple[SafetensorsTensor, ...]
    metadata: tuple[tuple[str, str], ...]
    header_bytes: int
    payload_bytes: int

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(tensor.name for tensor in self.tensors)


@dataclass(frozen=True)
class SafetensorsBinding:
    """One raw source tensor's canonical identity in a logical component."""

    source_file: Path
    source_key: str
    target_key: str
    target_shape: tuple[int, ...] | None = None
    reshape_rows: int | None = None


def read_header(path: Path | str) -> JsonObject:
    """Read the complete safetensors JSON header without loading tensors."""
    with Path(path).open("rb") as stream:
        stream.seek(0, 2)
        file_size = stream.tell()
        stream.seek(0)
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"{path}: truncated safetensors length prefix")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > _MAX_HEADER_BYTES:
            raise ValueError(
                f"{path}: safetensors header exceeds the {_MAX_HEADER_BYTES}-byte safety limit"
            )
        if header_length > file_size - 8:
            raise ValueError(
                f"{path}: truncated safetensors header (declared length exceeds file size)"
            )
        raw_header = stream.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError(f"{path}: truncated safetensors header")
    try:
        parsed = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ValueError(f"{path}: invalid safetensors header") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{path}: safetensors header must be an object")
    return cast(JsonObject, parsed)


def read_dtypes(path: Path | str) -> dict[str, str]:
    """Return raw safetensors dtype names without materializing tensor data."""
    header = read_header(path)
    return {
        key: str(entry["dtype"])
        for key, entry in header.items()
        if key != "__metadata__" and isinstance(entry, dict) and "dtype" in entry
    }


def inspect_safetensors(path: Path | str) -> SafetensorsInventory:
    """Validate tensor shapes and byte spans without loading tensor payloads."""
    source = Path(path).expanduser().resolve()
    header = read_header(source)
    file_size = source.stat().st_size
    with source.open("rb") as stream:
        raw_length = stream.read(8)
    if len(raw_length) != 8:
        raise ValueError(f"{source}: truncated safetensors length prefix")
    header_bytes = struct.unpack("<Q", raw_length)[0]
    payload_bytes = file_size - 8 - header_bytes

    metadata_value = header.get("__metadata__")
    if metadata_value is None:
        metadata: tuple[tuple[str, str], ...] = ()
    elif isinstance(metadata_value, dict) and all(
        isinstance(key, str) and isinstance(value, str) for key, value in metadata_value.items()
    ):
        metadata = tuple(sorted((str(key), str(value)) for key, value in metadata_value.items()))
    else:
        raise ValueError(f"{source}: safetensors metadata must contain only strings")

    tensors: list[SafetensorsTensor] = []
    spans: list[tuple[int, int, str]] = []
    for name, raw in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(raw, dict):
            raise ValueError(f"{source}: invalid safetensors tensor entry {name!r}")
        dtype = raw.get("dtype")
        shape = raw.get("shape")
        offsets = raw.get("data_offsets")
        if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
            raise ValueError(f"{source}: tensor {name!r} has unsupported dtype {dtype!r}")
        if not isinstance(shape, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape
        ):
            raise ValueError(f"{source}: tensor {name!r} has an invalid shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
        ):
            raise ValueError(f"{source}: tensor {name!r} has invalid data offsets")
        start, end = offsets
        expected_bytes = math.prod(shape) * _DTYPE_BYTES[dtype]
        if start < 0 or end < start or end > payload_bytes or end - start != expected_bytes:
            raise ValueError(f"{source}: tensor {name!r} has an invalid payload span")
        descriptor = SafetensorsTensor(
            name=name,
            dtype=dtype,
            shape=tuple(shape),
            data_offsets=(start, end),
        )
        tensors.append(descriptor)
        spans.append((start, end, name))
    ordered_spans = sorted(spans)
    cursor = 0
    for start, end, name in ordered_spans:
        if start != cursor:
            relation = "overlaps" if start < cursor else "leaves a gap before"
            raise ValueError(f"{source}: tensor payload {name!r} {relation} byte {cursor}")
        cursor = end
    if cursor != payload_bytes:
        raise ValueError(f"{source}: safetensors payload has unclaimed trailing bytes")
    return SafetensorsInventory(
        path=source,
        tensors=tuple(sorted(tensors, key=lambda tensor: tensor.name)),
        metadata=metadata,
        header_bytes=header_bytes,
        payload_bytes=payload_bytes,
    )


@dataclass(frozen=True)
class BoundTensorDescriptor:
    """One validated source tensor and its logical target binding."""

    binding: SafetensorsBinding
    source: Path
    descriptor: SafetensorsTensor

    @property
    def target_key(self) -> str:
        return self.binding.target_key

    @property
    def byte_count(self) -> int:
        return self.descriptor.byte_count


def inspect_bindings(
    bindings: Iterable[SafetensorsBinding],
    *,
    inventories: Mapping[Path, SafetensorsInventory] | None = None,
) -> tuple[BoundTensorDescriptor, ...]:
    """Validate bindings against their headers and order them by target key."""
    selected = tuple(bindings)
    if not selected:
        raise ValueError("component binding requires at least one tensor binding")
    indexed = {
        path.expanduser().resolve(): {tensor.name: tensor for tensor in inventory.tensors}
        for path, inventory in (inventories or {}).items()
    }
    entries: list[BoundTensorDescriptor] = []
    seen_targets: set[str] = set()
    for binding in selected:
        source = binding.source_file.expanduser().resolve()
        by_name = indexed.get(source)
        if by_name is None:
            by_name = {tensor.name: tensor for tensor in inspect_safetensors(source).tensors}
            indexed[source] = by_name
        descriptor = by_name.get(binding.source_key)
        if descriptor is None:
            raise ValueError(f"{source}: no tensor named {binding.source_key!r}")
        if binding.target_key in seen_targets:
            raise ValueError(f"component binding repeats target {binding.target_key!r}")
        seen_targets.add(binding.target_key)
        if binding.target_shape is not None and binding.reshape_rows is not None:
            raise ValueError(f"{binding.target_key!r}: target shape and reshape rows conflict")
        target_shape: tuple[int, ...]
        if binding.reshape_rows is not None:
            rows = binding.reshape_rows
            if rows <= 0 or descriptor.element_count % rows:
                raise ValueError(
                    f"{binding.target_key!r}: cannot reshape {descriptor.element_count} "
                    f"elements into {rows} rows"
                )
            target_shape = (rows, descriptor.element_count // rows)
        else:
            target_shape = binding.target_shape or descriptor.shape
        if math.prod(target_shape) != descriptor.element_count:
            raise ValueError(
                f"{binding.target_key!r}: target shape {target_shape} changes element count"
            )
        entries.append(BoundTensorDescriptor(binding, source, descriptor))
    entries.sort(key=lambda entry: entry.target_key)
    return tuple(entries)


def read_u8_tensor(path: Path | str, name: str) -> bytes:
    """Seek-read one one-dimensional U8 tensor without loading neighboring weights."""
    source = Path(path)
    header = read_header(source)
    entry = header.get(name)
    if not isinstance(entry, dict):
        raise KeyError(f"{source}: no tensor named {name!r}")
    if entry.get("dtype") != "U8":
        raise ValueError(f"{source}: tensor {name!r} is not U8")
    shape = entry.get("shape")
    offsets = entry.get("data_offsets")
    if (
        not isinstance(shape, list)
        or len(shape) != 1
        or isinstance(shape[0], bool)
        or not isinstance(shape[0], int)
        or shape[0] < 0
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
    ):
        raise ValueError(f"{source}: tensor {name!r} has an invalid U8 asset layout")
    start, end = offsets
    if start < 0 or end < start or end - start != shape[0]:
        raise ValueError(f"{source}: tensor {name!r} has inconsistent U8 asset offsets")
    file_size = source.stat().st_size
    with source.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"{source}: truncated safetensors length prefix")
        header_length = struct.unpack("<Q", raw_length)[0]
        data_start = 8 + header_length
        if end > file_size - data_start:
            raise ValueError(f"{source}: tensor {name!r} extends beyond the file payload")
        stream.seek(8 + header_length + start)
        payload = stream.read(end - start)
    if len(payload) != end - start:
        raise ValueError(f"{source}: tensor {name!r} has a truncated U8 payload")
    return payload


def load_weights(path: Path | str) -> dict[str, mx.array]:
    """Load a safetensors file's tensors into a name -> array dict.

    Discards metadata.  Use :func:`load_weights_with_metadata` when
    you need both.
    """
    weights, _metadata = cast(
        tuple[dict[str, mx.array], dict[str, str]],
        mx.load(
            str(path),
            return_metadata=True,
            format="safetensors",
        ),
    )
    return weights


def load_weights_with_metadata(
    path: Path | str,
) -> tuple[dict[str, mx.array], dict[str, str]]:
    """Load weights and string-keyed metadata together."""
    weights, metadata = cast(
        tuple[dict[str, mx.array], dict[str, str]],
        mx.load(
            str(path),
            return_metadata=True,
            format="safetensors",
        ),
    )
    return weights, metadata


def save_weights(
    path: Path | str,
    weights: Mapping[str, mx.array],
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Save weights (and optional metadata) to a safetensors file.

    ``metadata`` keys and values must both be strings - that's the
    safetensors format constraint, not ours.  Encode numeric or
    structured data (e.g. schema versions, layout flags) as strings
    at this boundary; decode on the read side.
    """
    with atomic_output_path(path, temp_suffix=".tmp.safetensors") as temporary:
        mx.save_safetensors(
            str(temporary),
            dict(weights),
            metadata=dict(metadata) if metadata else None,
        )


def read_metadata(path: Path | str) -> dict[str, str]:
    """Read metadata from a safetensors file without materializing tensors.

    Inspect constructor metadata, provenance, and artifact versions without
    loading weight tensors into memory.

    Reads the safetensors header natively - an 8-byte little-endian length
    prefix followed by a JSON header whose ``__metadata__`` key holds the
    string map - so this stays free of the safetensors and numpy packages.
    ``mx.load`` handles the tensor path and doesn't need them either.
    """
    header = read_header(path)
    raw_metadata = header.get("__metadata__")
    if raw_metadata is None:
        return {}
    if not isinstance(raw_metadata, dict):
        raise ValueError(f"{path}: safetensors metadata must be an object")
    return {str(key): str(value) for key, value in raw_metadata.items()}


__all__ = [
    "BoundTensorDescriptor",
    "SafetensorsBinding",
    "SafetensorsInventory",
    "SafetensorsTensor",
    "inspect_bindings",
    "inspect_safetensors",
    "load_weights",
    "load_weights_with_metadata",
    "read_dtypes",
    "read_header",
    "read_metadata",
    "read_u8_tensor",
    "save_weights",
]
